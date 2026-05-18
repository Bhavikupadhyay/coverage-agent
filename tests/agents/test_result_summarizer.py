"""ResultSummarizer — final-step LLM agent. Tests cover the offline template and the LLM JSON path."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from coverage_agent.agents.result_summarizer import ResultSummarizer
from coverage_agent.contracts.schemas import (
    BranchGap,
    CoverageGap,
    ExecutionResult,
    GapResult,
    RegressionResult,
)


def _gap(committed: bool, symbol: str = "login") -> GapResult:
    gap = CoverageGap(
        file_path=f"pkg/{symbol}.py",
        target_symbol=symbol,
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=[10, 11, 12],
        priority_score=0.9,
        gap_id=f"pkg/{symbol}.py:10->12",
    )
    return GapResult(
        gap=gap,
        skipped=not committed,
        loops_taken=1,
        phase1_scores=None,
        phase2_scores=(
            ExecutionResult(execution_success=True, target_branch_hit=True, coverage_delta=0.4)
            if committed else None
        ),
        final_test_committed=committed,
        test_code="def test_x(): pass" if committed else None,
        skip_reason="" if committed else "eval loop exhausted",
    )


_SCORECARD = {
    "gaps_targeted": 3,
    "tests_committed": 2,
    "skipped": 1,
    "branch_hit_rate": "67%",
    "avg_coverage_delta": "+0.40%",
    "avg_loops": "1.5",
    "llm_cost": "$0.0000",
}


def test_offline_returns_template_with_committed_bullets(offline_creds):
    out = ResultSummarizer(offline_creds).run(
        [_gap(True, "login"), _gap(True, "logout"), _gap(False, "refresh")],
        _SCORECARD,
        regression=None,
    )
    assert "login" in out.pr_description
    assert "logout" in out.pr_description
    assert "2 new pytest" in out.pr_description
    assert "1 gap(s) skipped" in out.pr_description
    assert "per-gap loop" in out.full_summary.lower()


def test_offline_includes_regression_status(offline_creds):
    reg = RegressionResult(
        baseline_passed=14, baseline_failed=0,
        post_passed=15, post_failed=1,
        new_failures=1, regression_detected=True,
        summary="Regression detected: 1 previously-passing test(s) now fail.",
        skipped=False,
    )
    out = ResultSummarizer(offline_creds).run([_gap(True)], _SCORECARD, regression=reg)
    assert "regression" in out.pr_description.lower()


def test_byok_parses_strict_json_from_llm(byok_creds):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = json.dumps({
        "pr_description": "### New tests\n- pkg/login.py::login",
        "full_summary": "The pipeline committed one test for the login flow.",
    })
    with patch("coverage_agent.agents.result_summarizer.litellm.completion", return_value=fake_response):
        out = ResultSummarizer(byok_creds).run([_gap(True)], _SCORECARD, regression=None)
    assert out.pr_description.startswith("### New tests")
    assert "login flow" in out.full_summary


def test_byok_falls_back_when_llm_errors(byok_creds):
    with patch("coverage_agent.agents.result_summarizer.litellm.completion", side_effect=RuntimeError("boom")):
        out = ResultSummarizer(byok_creds).run([_gap(True)], _SCORECARD, regression=None)
    # Falls back to the offline templated path — must still produce content.
    assert out.pr_description and out.full_summary
    assert "1 new pytest" in out.pr_description
