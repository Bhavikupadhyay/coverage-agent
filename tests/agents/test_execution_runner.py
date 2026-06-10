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
