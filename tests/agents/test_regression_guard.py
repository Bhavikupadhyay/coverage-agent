"""RegressionGuard — deterministic, no LLM calls."""
from __future__ import annotations

from unittest.mock import MagicMock

from coverage_agent.engine.regression import RegressionGuard, _filename_for
from coverage_agent.contracts import (
    BranchGap,
    CoverageGap,
    ExecutionResult,
    GapResult,
)


def _make_gap_result(committed: bool, test_code: str = "def test_x(): pass") -> GapResult:
    gap = CoverageGap(
        file_path="pkg/auth.py",
        target_symbol="login",
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=list(range(8, 20)),
        kind="branch",
        origin="full",
        gap_id="pkg/auth.py:10->12",
    )
    return GapResult(
        gap=gap,
        skipped=not committed,
        loops_taken=1,
        validation=None,
        execution=(
            ExecutionResult(execution_success=True, target_branch_hit=True, targets_hit=1, targets_total=1)
            if committed else None
        ),
        accepted=committed,
        test_code=test_code if committed else None,
    )


def test_filename_is_stable_and_collision_free():
    a = _make_gap_result(True)
    b = GapResult(
        gap=CoverageGap(
            file_path="pkg/auth.py",
            target_symbol="login",
            branch=BranchGap(from_line=20, to_line=22),
            surrounding_lines=[20, 21, 22],
            kind="branch",
            origin="full",
            gap_id="pkg/auth.py:20->22",
        ),
        skipped=False, loops_taken=1, validation=None, execution=None,
        accepted=True, test_code="x",
    )
    fa = _filename_for(a)
    fb = _filename_for(b)
    assert fa.endswith(".py")
    assert fa != fb


def test_skipped_when_nothing_committed(creds):
    result = RegressionGuard(creds).run(
        sandbox=MagicMock(),
        committed_results=[_make_gap_result(False)],
        baseline_passed=12,
        baseline_failed=0,
    )
    assert result.skipped is True
    assert result.regression_detected is False
    assert result.new_failures == 0


def test_accepted_tests_persisted_and_suite_reruns(creds):
    sandbox = MagicMock()
    sandbox.count_test_outcomes.return_value = (14, 0)
    result = RegressionGuard(creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True), _make_gap_result(True)],
        baseline_passed=12,
        baseline_failed=0,
    )
    assert result.skipped is False
    assert result.regression_detected is False
    assert result.post_passed == 14
    assert sandbox.persist_test.call_count == 2
    sandbox.count_test_outcomes.assert_called_once()


def test_flags_regression_when_post_failures_exceed_baseline(creds):
    sandbox = MagicMock()
    sandbox.count_test_outcomes.return_value = (14, 1)
    result = RegressionGuard(creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True), _make_gap_result(True)],
        baseline_passed=14,
        baseline_failed=0,
    )
    assert sandbox.persist_test.call_count == 2
    assert result.regression_detected is True
    assert result.new_failures == 1
    assert "Regression detected" in result.summary


def test_clean_when_no_new_failures(creds):
    sandbox = MagicMock()
    sandbox.count_test_outcomes.return_value = (15, 0)
    result = RegressionGuard(creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True)],
        baseline_passed=14,
        baseline_failed=0,
    )
    assert result.regression_detected is False
    assert result.new_failures == 0
    assert "clean" in result.summary.lower()


# ---------------------------------------------------------------------------
# New subprocess path
# ---------------------------------------------------------------------------

def test_subprocess_path_writes_tests_and_runs_suite(creds, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from coverage_agent.config import AgentConfig
    from coverage_agent.engine.regression import _run_test_command
    cfg = AgentConfig(tests_dir=str(tmp_path / "tests_gen"), test_command="pytest -q")

    # Two distinct gaps so filenames don't collide.
    gap2 = CoverageGap(
        file_path="pkg/auth.py", target_symbol="logout",
        branch=BranchGap(from_line=20, to_line=22),
        surrounding_lines=[20, 21, 22], kind="branch", origin="full",
        gap_id="pkg/auth.py:20->22",
    )
    result2 = GapResult(gap=gap2, skipped=False, loops_taken=1,
                        validation=None, execution=None,
                        accepted=True, test_code="def test_y(): pass")

    import coverage_agent.engine.regression as reg_mod
    orig = reg_mod._run_test_command
    reg_mod._run_test_command = lambda *a, **kw: (15, 0)
    try:
        result = RegressionGuard(creds).run(
            committed_results=[_make_gap_result(True), result2],
            baseline_passed=13,
            baseline_failed=0,
            config=cfg,
            repo_root=str(tmp_path),
        )
    finally:
        reg_mod._run_test_command = orig

    assert result.post_passed == 15
    assert result.regression_detected is False
    tests_dir = tmp_path / "tests_gen"
    written = list(tests_dir.glob("test_coverageagent_*.py"))
    assert len(written) == 2


def test_subprocess_path_regression_detected(creds, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from coverage_agent.config import AgentConfig
    import coverage_agent.engine.regression as reg_mod
    cfg = AgentConfig(tests_dir=str(tmp_path / "tests_gen"), test_command="pytest -q")

    orig = reg_mod._run_test_command
    reg_mod._run_test_command = lambda *a, **kw: (13, 2)
    try:
        result = RegressionGuard(creds).run(
            committed_results=[_make_gap_result(True)],
            baseline_passed=14,
            baseline_failed=0,
            config=cfg,
            repo_root=str(tmp_path),
        )
    finally:
        reg_mod._run_test_command = orig

    assert result.regression_detected is True
    assert result.new_failures == 2
    assert "Regression detected" in result.summary
