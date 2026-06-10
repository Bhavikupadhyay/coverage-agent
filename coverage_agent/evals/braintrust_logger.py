"""
Braintrust dataset logging for gap results.

No-ops silently when BRAINTRUST_API_KEY is absent so the pipeline
works without a Braintrust account.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from coverage_agent.contracts import GapResult

logger = logging.getLogger(__name__)

_PROJECT = "coverage-agent"


def log_gap_result(
    gap_result: GapResult,
    context,  # Optional[ContextPayload] — avoid circular import
    run_id: str,
) -> None:
    """Log one gap attempt to Braintrust. No-ops if BRAINTRUST_API_KEY is not set."""
    api_key = os.environ.get("BRAINTRUST_API_KEY", "")
    if not api_key:
        logger.debug("BRAINTRUST_API_KEY not set — skipping Braintrust logging")
        return

    try:
        import braintrust

        experiment = braintrust.init(project=_PROJECT, experiment=run_id, api_key=api_key)

        phase1 = gap_result.phase1_scores
        phase2 = gap_result.phase2_scores

        scores: dict = {}
        if phase1 is not None:
            scores["syntax_valid"] = 1.0 if phase1.syntax_valid else 0.0
        if phase2 is not None:
            scores["execution_success"] = 1.0 if phase2.execution_success else 0.0
            scores["target_branch_hit"] = 1.0 if phase2.target_branch_hit else 0.0

        experiment.log(
            input={
                "gap": gap_result.gap.model_dump(),
                "context_tokens": context.tokens_used if context is not None else 0,
            },
            output={"test_code": gap_result.test_code},
            scores=scores,
            metadata={
                "loops_taken": gap_result.loops_taken,
                "skipped": gap_result.skipped,
                "gap_difficulty": gap_result.gap_difficulty,
            },
            id=gap_result.gap.gap_id,
        )
        logger.debug("Logged gap %s to Braintrust experiment %s", gap_result.gap.gap_id, run_id)
    except Exception as exc:
        logger.warning("Braintrust logging failed for %s: %s", gap_result.gap.gap_id, exc)
