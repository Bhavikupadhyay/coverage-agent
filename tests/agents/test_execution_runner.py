"""ExecutionRunner — subprocess-based execution and flakiness detection."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from coverage_agent.config import AgentConfig
from coverage_agent.contracts import BranchGap, CoverageGap, DraftTest, ExecutionResult
from coverage_agent.engine.executor import ExecutionRunner, _run_once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap(from_l: int = 10, to_l: int = 12) -> CoverageGap:
    return CoverageGap(
        file_path="pkg/auth.py",
        target_symbol="login",
        branch=BranchGap(from_line=from_l, to_line=to_l),
        surrounding_lines=list(range(from_l - 2, to_l + 3)),
        kind="branch",
        origin="full",
        gap_id=f"pkg/auth.py:{from_l}->{to_l}",
    )


def _make_draft() -> DraftTest:
    return DraftTest(
        test_code="def test_x(): assert True\n",
        mocks_used=[],
        target_branch=BranchGap(from_line=10, to_line=12),
    )


def _make_result(success: bool, branch_hit: bool = True) -> ExecutionResult:
    return ExecutionResult(
        execution_success=success,
        target_branch_hit=branch_hit if success else False,
        targets_hit=1 if success else 0,
        targets_total=1,
        stderr_trace="" if success else "boom",
        flaky=False,
    )


# ---------------------------------------------------------------------------
# Legacy sandbox path (keeps old tests working)
# ---------------------------------------------------------------------------

class _FakeSandbox:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def run_test(self, test_code, **kwargs):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return _make_result(True)


def test_failure_returns_immediately(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(success=False)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox=sandbox)
    assert sandbox.calls == 1
    assert result.execution_success is False


def test_three_passes_returns_non_flaky(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(True), _make_result(True), _make_result(True)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox=sandbox)
    assert sandbox.calls == 3
    assert result.execution_success is True
    assert result.flaky is False
    assert result.target_branch_hit is True


def test_inconsistent_runs_marked_flaky(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(True), _make_result(True), _make_result(False)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox=sandbox)
    assert sandbox.calls == 3
    assert result.flaky is True
    assert result.execution_success is False


# ---------------------------------------------------------------------------
# New subprocess path
# ---------------------------------------------------------------------------

def _mock_cov(arcs=None):
    """Returns a mock coverage.Coverage object with the given arcs."""
    import coverage as coverage_module
    cov_mock = MagicMock(spec=coverage_module.Coverage)
    data_mock = MagicMock()
    data_mock.arcs.return_value = arcs or []
    cov_mock.get_data.return_value = data_mock
    return cov_mock


def test_subprocess_path_passes_when_test_passes(creds, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gap = _make_gap()
    draft = DraftTest(
        test_code="def test_pass(): assert True\n",
        mocks_used=[],
        target_branch=gap.branch,
    )
    cfg = AgentConfig(tests_dir=str(tmp_path / "tests_gen"), flaky_runs=1)

    with patch("coverage_agent.engine.executor._run_once") as mock_run:
        mock_run.return_value = ExecutionResult(
            execution_success=True,
            target_branch_hit=True,
            targets_hit=1,
            targets_total=1,
        )
        result = ExecutionRunner(creds).run(draft, gap, config=cfg)

    assert result.execution_success is True
    assert result.target_branch_hit is True


def test_subprocess_path_flakiness_detected(creds, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gap = _make_gap()
    draft = _make_draft()
    cfg = AgentConfig(tests_dir=str(tmp_path / "tests_gen"), flaky_runs=3)

    call_count = [0]
    def _fake_run_once(*args, **kwargs):
        call_count[0] += 1
        # Third run fails.
        if call_count[0] >= 3:
            return ExecutionResult(execution_success=False, target_branch_hit=False, stderr_trace="flake")
        return ExecutionResult(execution_success=True, target_branch_hit=True, targets_hit=1, targets_total=1)

    with patch("coverage_agent.engine.executor._run_once", side_effect=_fake_run_once):
        result = ExecutionRunner(creds).run(draft, gap, config=cfg)

    assert result.flaky is True
    assert result.execution_success is False


def test_subprocess_first_run_fail_returns_immediately(creds, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gap = _make_gap()
    draft = _make_draft()
    cfg = AgentConfig(tests_dir=str(tmp_path / "tests_gen"), flaky_runs=3)

    call_count = [0]
    def _fake_run_once(*args, **kwargs):
        call_count[0] += 1
        return ExecutionResult(execution_success=False, target_branch_hit=False, stderr_trace="fail")

    with patch("coverage_agent.engine.executor._run_once", side_effect=_fake_run_once):
        result = ExecutionRunner(creds).run(draft, gap, config=cfg)

    assert call_count[0] == 1
    assert result.execution_success is False


# ---------------------------------------------------------------------------
# _check_targets — kind-aware target verification
# ---------------------------------------------------------------------------

from coverage_agent.engine.executor import _check_targets


class _FakeCovData:
    """Minimal stand-in for coverage.CoverageData: arcs() + lines()."""

    def __init__(self, arcs=None, lines=None):
        self._arcs = arcs or []
        self._lines = lines or []

    def arcs(self, _path):
        return self._arcs

    def lines(self, _path):
        return self._lines


def _make_function_gap(def_line: int = 5, body: list[int] | None = None) -> CoverageGap:
    body = body if body is not None else [6, 7, 8, 9]
    return CoverageGap(
        file_path="pkg/scoring.py",
        target_symbol="percentage",
        branch=BranchGap(from_line=def_line, to_line=body[-1] if body else def_line),
        surrounding_lines=[def_line] + body,
        kind="function",
        origin="diff",
        gap_id=f"pkg/scoring.py:{def_line}->{body[-1] if body else def_line}",
    )


def test_branch_gap_arc_present_is_hit():
    gap = _make_gap(10, 12)
    data = _FakeCovData(arcs=[(10, 12), (1, 2)])
    assert _check_targets(gap, data, "pkg/auth.py") == (1, 1)


def test_branch_gap_arc_absent_is_miss():
    gap = _make_gap(10, 12)
    data = _FakeCovData(arcs=[(10, 11), (1, 2)])
    assert _check_targets(gap, data, "pkg/auth.py") == (0, 1)


def test_function_gap_body_executed_is_hit():
    gap = _make_function_gap(def_line=5, body=[6, 7, 8, 9])
    data = _FakeCovData(lines=[5, 6, 7, 8, 9])
    hit, total = _check_targets(gap, data, "pkg/scoring.py")
    assert (hit, total) == (4, 4)
    assert hit >= 1 and hit / total >= 0.5  # accepted under the engine rule


def test_function_gap_import_only_is_rejected():
    # Import executes only the def line; no body line runs.
    gap = _make_function_gap(def_line=5, body=[6, 7, 8, 9])
    data = _FakeCovData(lines=[5])
    hit, total = _check_targets(gap, data, "pkg/scoring.py")
    assert (hit, total) == (0, 4)


def test_function_gap_partial_below_half_is_rejected():
    gap = _make_function_gap(def_line=5, body=[6, 7, 8, 9])
    data = _FakeCovData(lines=[5, 6])
    hit, total = _check_targets(gap, data, "pkg/scoring.py")
    assert (hit, total) == (1, 4)
    assert not (hit >= 1 and hit / total >= 0.5)


def test_function_gap_no_body_lines_is_safe():
    gap = _make_function_gap(def_line=5, body=[])
    data = _FakeCovData(lines=[5])
    assert _check_targets(gap, data, "pkg/scoring.py") == (0, 0)


# ---------------------------------------------------------------------------
# _cluster_results_from_exec — per-gap result derivation
# ---------------------------------------------------------------------------

from coverage_agent.engine.executor import _cluster_results_from_exec, _cluster_arc_store
from coverage_agent.contracts import ExecutionResult


def _make_branch_gap(from_l: int, to_l: int) -> CoverageGap:
    return CoverageGap(
        file_path="pkg/stats.py",
        target_symbol="letter_grade",
        branch=BranchGap(from_line=from_l, to_line=to_l),
        surrounding_lines=list(range(from_l, to_l + 2)),
        kind="branch",
        origin="full",
        gap_id=f"pkg/stats.py:{from_l}->{to_l}",
    )


def test_cluster_results_single_gap_replicates():
    """Single-gap cluster returns the exec_result directly."""
    gap = _make_branch_gap(10, 12)
    er = ExecutionResult(
        execution_success=True,
        target_branch_hit=True,
        targets_hit=1,
        targets_total=1,
    )
    results = _cluster_results_from_exec([gap], er)
    assert len(results) == 1
    assert results[0] is er


def test_cluster_results_partial_acceptance():
    """One arc hit, one missed → first accepted, second not."""
    g1 = _make_branch_gap(35, 37)
    g2 = _make_branch_gap(37, 38)
    cluster = [g1, g2]

    er = ExecutionResult(
        execution_success=True,
        target_branch_hit=True,   # any arc hit
        targets_hit=1,
        targets_total=2,
    )
    # Inject arc-hit data directly into the store.
    _cluster_arc_store[id(er)] = {
        (35, 37): True,
        (37, 38): False,
    }

    results = _cluster_results_from_exec(cluster, er)
    assert len(results) == 2
    assert results[0].target_branch_hit is True
    assert results[1].target_branch_hit is False
    # Store entry is consumed.
    assert id(er) not in _cluster_arc_store


def test_cluster_results_none_exec_all_missed():
    """None exec_result marks every gap as not executed."""
    cluster = [_make_branch_gap(10, 12), _make_branch_gap(14, 16)]
    results = _cluster_results_from_exec(cluster, None)
    assert all(not r.execution_success for r in results)
    assert all(not r.target_branch_hit for r in results)


def test_cluster_results_fallback_when_no_arc_store():
    """Without arc store data, only primary gap inherits the hit."""
    g1 = _make_branch_gap(10, 12)
    g2 = _make_branch_gap(14, 16)
    cluster = [g1, g2]

    er = ExecutionResult(
        execution_success=True,
        target_branch_hit=True,
        targets_hit=1,
        targets_total=1,
    )
    # No entry in _cluster_arc_store.
    results = _cluster_results_from_exec(cluster, er)
    assert results[0].target_branch_hit is True   # primary
    assert results[1].target_branch_hit is False  # sibling — conservative
