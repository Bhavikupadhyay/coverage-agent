"""Tests for coverage_agent/evals/braintrust_logger.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coverage_agent.contracts import (
    BranchGap,
    CoverageGap,
    GapResult,
    ValidationResult,
    ExecutionResult,
)
from coverage_agent.evals.braintrust_logger import log_gap_result


def _make_gap() -> CoverageGap:
    return CoverageGap(
        file_path="pkg/mod.py",
        target_symbol="process",
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=[10, 11, 12],
        kind="branch",
        origin="full",
        gap_id="pkg/mod.py:10->12",
    )


def _make_gap_result(accepted: bool = True) -> GapResult:
    phase1 = ValidationResult(
        syntax_valid=True,
        unknown_imports=[],
        critique="looks good",
        route="EXECUTE",
    )
    phase2 = ExecutionResult(
        execution_success=accepted,
        target_branch_hit=accepted,
        targets_hit=1 if accepted else 0,
        targets_total=1,
    )
    return GapResult(
        gap=_make_gap(),
        skipped=not accepted,
        loops_taken=1,
        phase1_scores=phase1,
        phase2_scores=phase2,
        accepted=accepted,
        test_code="def test_x(): pass" if accepted else None,
    )


# ---------------------------------------------------------------------------
# No-op when key is absent
# ---------------------------------------------------------------------------

def test_noop_when_key_absent(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    with patch("braintrust.init") as mock_init:
        log_gap_result(_make_gap_result(), context=None, run_id="run-test")
    mock_init.assert_not_called()


# ---------------------------------------------------------------------------
# Calls experiment.log with correct shape when key is present
# ---------------------------------------------------------------------------

def test_logs_when_key_present(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "fake-key")

    mock_experiment = MagicMock()
    with patch("braintrust.init", return_value=mock_experiment) as mock_init:
        log_gap_result(_make_gap_result(accepted=True), context=None, run_id="run-abc")

    mock_init.assert_called_once_with(
        project="coverage-agent", experiment="run-abc", api_key="fake-key"
    )
    mock_experiment.log.assert_called_once()
    kwargs = mock_experiment.log.call_args[1]
    assert kwargs["scores"]["syntax_valid"] == 1.0
    assert kwargs["scores"]["execution_success"] == 1.0
    assert kwargs["scores"]["target_branch_hit"] == 1.0
    assert kwargs["metadata"]["loops_taken"] == 1
    assert kwargs["metadata"]["skipped"] is False
    assert kwargs["id"] == "pkg/mod.py:10->12"


def test_context_tokens_passed_when_context_provided(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "fake-key")

    mock_ctx = MagicMock()
    mock_ctx.tokens_used = 4200

    mock_experiment = MagicMock()
    with patch("braintrust.init", return_value=mock_experiment):
        log_gap_result(_make_gap_result(), context=mock_ctx, run_id="run-xyz")

    kwargs = mock_experiment.log.call_args[1]
    assert kwargs["input"]["context_tokens"] == 4200


def test_no_scores_when_phase_results_absent(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "fake-key")

    gr = GapResult(
        gap=_make_gap(),
        skipped=True,
        loops_taken=3,
        accepted=False,
    )
    mock_experiment = MagicMock()
    with patch("braintrust.init", return_value=mock_experiment):
        log_gap_result(gr, context=None, run_id="run-skip")

    kwargs = mock_experiment.log.call_args[1]
    assert kwargs["scores"] == {}


def test_exception_in_braintrust_does_not_propagate(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "fake-key")

    with patch("braintrust.init", side_effect=RuntimeError("network error")):
        # Should not raise
        log_gap_result(_make_gap_result(), context=None, run_id="run-err")
