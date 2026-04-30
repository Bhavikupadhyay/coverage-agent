import logging

from coverage_agent.config import is_dry_run
from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.contracts.schemas import GapResult
from coverage_agent.pipeline import run_pipeline
from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Outer loop: prepares sandbox, runs baseline, iterates gaps, returns results.

    All repo operations run inside E2B — no local filesystem access.
    Test code for verified gaps is returned in GapResult.test_code for
    display and download via the web UI or CLI.

    Usage:
        scorecard, results = Orchestrator().run(repo_url_or_path, max_gaps=10)
    """

    def run(
        self,
        repo_url_or_path: str,
        max_gaps: int = 10,
        braintrust_logger=None,
    ) -> tuple[dict, list[GapResult]]:
        sandbox = E2BSandbox(repo_url_or_path)
        try:
            sandbox.setup_repo(repo_url_or_path)
            sandbox.install_dependencies()
            self._validate_python_repo(sandbox)
            baseline_coverage = sandbox.run_coverage_baseline()
        except Exception:
            sandbox.close()
            raise

        raw_gaps = sandbox.parse_gaps(baseline_coverage)
        priority_queue = GapPrioritizer().run(raw_gaps)[:max_gaps]
        logger.info("Orchestrator: %d gaps to process (max=%d)", len(priority_queue), max_gaps)

        results: list[GapResult] = []
        try:
            for gap in priority_queue:
                sandbox.pause()
                gap_result, final_state = run_pipeline(
                    gap=gap,
                    sandbox=sandbox,
                    baseline_coverage=baseline_coverage,
                    braintrust_logger=braintrust_logger,
                )
                if (
                    not gap_result.skipped
                    and gap_result.phase2_scores is not None
                    and gap_result.phase2_scores.target_branch_hit
                ):
                    draft = final_state.get("draft_test")
                    if draft is not None:
                        gap_result = gap_result.model_copy(update={
                            "final_test_committed": True,
                            "test_code": draft.test_code,
                        })

                results.append(gap_result)
                logger.info(
                    "gap=%s skipped=%s branch_hit=%s committed=%s",
                    gap.gap_id,
                    gap_result.skipped,
                    gap_result.phase2_scores.target_branch_hit if gap_result.phase2_scores else False,
                    gap_result.final_test_committed,
                )
        finally:
            sandbox.close()

        if braintrust_logger is not None:
            braintrust_logger.flush()

        scorecard = self._build_scorecard(results, repo_url_or_path, llm_cost=0.0)
        return scorecard, results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_python_repo(self, sandbox: E2BSandbox) -> None:
        """Verifies the repo in E2B has Python files. Raises ValueError if not."""
        if is_dry_run():
            return
        result = sandbox._sandbox.commands.run(
            "find /home/user/repo -name '*.py' | wc -l",
            timeout=15,
        )
        count = int(result.stdout.strip() or "0")
        if count == 0:
            raise ValueError(
                "Not a Python repository — no .py files found. "
                "CoverageAgent only supports Python projects."
            )

    def _build_scorecard(
        self, results: list[GapResult], repo: str, llm_cost: float
    ) -> dict:
        targeted = len(results)
        committed = sum(1 for r in results if r.final_test_committed)
        skipped = sum(1 for r in results if r.skipped)
        hits = [r for r in results if r.phase2_scores and r.phase2_scores.target_branch_hit]
        branch_hit_rate = len(hits) / targeted if targeted else 0.0
        avg_delta = (
            sum(r.phase2_scores.coverage_delta for r in hits) / len(hits) if hits else 0.0
        )
        avg_loops = sum(r.loops_taken for r in results) / targeted if targeted else 0.0
        cost_label = f"${llm_cost:.4f}" + (" (DRY_RUN)" if is_dry_run() else "")
        return {
            "repo": repo,
            "gaps_targeted": targeted,
            "tests_committed": committed,
            "skipped": skipped,
            "branch_hit_rate": f"{branch_hit_rate:.0%}",
            "avg_coverage_delta": f"+{avg_delta:.2f}%",
            "avg_loops": f"{avg_loops:.1f}",
            "llm_cost": cost_label,
        }
