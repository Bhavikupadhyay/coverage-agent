"""RegressionGuard — sandbox-only agent. No LLM calls, all coverage is offline."""
from __future__ import annotations

from unittest.mock import MagicMock

from coverage_agent.agents.regression_guard import RegressionGuard, _filename_for
from coverage_agent.contracts.schemas import (
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
        priority_score=0.9,
        gap_id="pkg/auth.py:10->12",
    )
    return GapResult(
        gap=gap,
        skipped=not committed,
        loops_taken=1,
        phase1_scores=None,
        phase2_scores=(
            ExecutionResult(execution_success=True, target_branch_hit=True, coverage_delta=0.5)
            if committed else None
        ),
        final_test_committed=committed,
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
            priority_score=0.9,
            gap_id="pkg/auth.py:20->22",
        ),
        skipped=False, loops_taken=1, phase1_scores=None, phase2_scores=None,
        final_test_committed=True, test_code="x",
    )
    fa = _filename_for(a)
    fb = _filename_for(b)
    assert fa.endswith(".py")
    assert fa != fb  # different branches → different filenames


def test_skipped_when_nothing_committed(offline_creds):
    result = RegressionGuard(offline_creds).run(
        sandbox=MagicMock(),
        committed_results=[_make_gap_result(False)],
        baseline_passed=12,
        baseline_failed=0,
    )
    assert result.skipped is True
    assert result.regression_detected is False
    assert result.new_failures == 0


def test_offline_path_returns_clean_pass(offline_creds):
    sandbox = MagicMock()  # never called in offline mode
    result = RegressionGuard(offline_creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True), _make_gap_result(True)],
        baseline_passed=12,
        baseline_failed=0,
    )
    assert result.skipped is False
    assert result.regression_detected is False
    assert result.post_passed == 14  # 12 + 2 new
    sandbox.persist_test.assert_not_called()
    sandbox.count_test_outcomes.assert_not_called()


def test_byok_flags_regression_when_post_failures_exceed_baseline(byok_creds):
    sandbox = MagicMock()
    sandbox.count_test_outcomes.return_value = (14, 1)  # one new failure
    result = RegressionGuard(byok_creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True), _make_gap_result(True)],
        baseline_passed=14,
        baseline_failed=0,
    )
    assert sandbox.persist_test.call_count == 2
    sandbox.count_test_outcomes.assert_called_once()
    assert result.regression_detected is True
    assert result.new_failures == 1
    assert "Regression detected" in result.summary


def test_byok_clean_when_no_new_failures(byok_creds):
    sandbox = MagicMock()
    sandbox.count_test_outcomes.return_value = (15, 0)
    result = RegressionGuard(byok_creds).run(
        sandbox=sandbox,
        committed_results=[_make_gap_result(True)],
        baseline_passed=14,
        baseline_failed=0,
    )
    assert result.regression_detected is False
    assert result.new_failures == 0
    assert "clean" in result.summary.lower()
