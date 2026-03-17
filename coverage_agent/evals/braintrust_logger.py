import logging
import os

from coverage_agent.contracts.schemas import GapResult

logger = logging.getLogger(__name__)


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


class BraintrustLogger:
    """
    Logs GapResult rows to a Braintrust dataset after each pipeline run.

    In dry-run mode, rows are printed to the log — no real API call is made and
    no BRAINTRUST_API_KEY is required.

    In live mode, requires BRAINTRUST_API_KEY in the environment. Each row is
    inserted into the "coverage_gaps" dataset in the "coverage-agent" project.
    Scores are embedded in metadata because Braintrust Dataset rows don't have
    a native scores field (that's Experiment-level); the metadata dict makes
    them queryable and exportable.
    """

    def __init__(self, project_name: str = "coverage-agent") -> None:
        self.project_name = project_name
        self._dataset = None

        if _is_dry_run():
            logger.info("[DRY_RUN] BraintrustLogger — no real connection will be made")
            return

        try:
            import braintrust
            self._dataset = braintrust.init_dataset(
                project=project_name,
                name="coverage_gaps",
            )
            logger.info("Braintrust dataset initialized: project=%s", project_name)
        except Exception as exc:
            logger.error("Braintrust init failed: %s", exc)
            raise

    def log(self, gap_result: GapResult, final_state: dict | None = None) -> None:
        """
        Logs one gap attempt.

        final_state is the full PipelineState dict from pipeline.py — used to
        include context_payload and draft_test in the dataset row if present.
        Passing None is safe; those fields will be omitted from the row.
        """
        final_state = final_state or {}

        if _is_dry_run():
            p1 = gap_result.phase1_scores
            p2 = gap_result.phase2_scores
            logger.info(
                "[DRY_RUN] BraintrustLogger.log — gap_id=%s skipped=%s loops=%d "
                "route=%s branch_hit=%s delta=%s",
                gap_result.gap.gap_id,
                gap_result.skipped,
                gap_result.loops_taken,
                p1.route if p1 else "n/a",
                p2.target_branch_hit if p2 else "n/a",
                p2.coverage_delta if p2 else "n/a",
            )
            return

        scores = self._build_scores(gap_result)
        context = final_state.get("context")
        draft = final_state.get("draft_test")

        self._dataset.insert(
            input={
                "gap": gap_result.gap.model_dump(),
                "context_payload": context.model_dump() if context else None,
            },
            output={
                "draft_test": draft.model_dump() if draft else None,
            },
            metadata={
                "scores": scores,
                "loops_taken": gap_result.loops_taken,
                "skipped": gap_result.skipped,
                "final_test_committed": gap_result.final_test_committed,
                "model": "gemini/gemini-2.5-flash",
                "graph_depth_used": context.graph_depth_used if context else None,
            },
            id=gap_result.gap.gap_id,
        )
        logger.info("Logged gap_id=%s to Braintrust", gap_result.gap.gap_id)

    def flush(self) -> None:
        """Flushes the dataset buffer. Call once after all gaps are processed."""
        if _is_dry_run():
            return
        if self._dataset:
            self._dataset.flush()
            logger.info("Braintrust dataset flushed")

    def _build_scores(self, gap_result: GapResult) -> dict:
        scores: dict = {}
        p1 = gap_result.phase1_scores
        if p1:
            scores["syntax_valid"] = 1.0 if p1.syntax_valid else 0.0
            scores["mock_complete"] = 1.0 if p1.mock_complete else 0.0
            # Normalize assertion score from 1–5 to 0.0–1.0
            scores["assertion_score"] = (p1.assertion_score - 1) / 4.0
        p2 = gap_result.phase2_scores
        if p2:
            scores["execution_success"] = 1.0 if p2.execution_success else 0.0
            scores["target_branch_hit"] = 1.0 if p2.target_branch_hit else 0.0
            scores["coverage_delta"] = p2.coverage_delta
        return scores
