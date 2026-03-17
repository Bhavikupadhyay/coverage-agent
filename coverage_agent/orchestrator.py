import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.contracts.schemas import CoverageGap, ExecutionResult, GapResult
from coverage_agent.pipeline import run_pipeline
from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Outer loop: prepares repo, runs baseline, iterates gaps, commits passing tests.

    Usage:
        scorecard, results = Orchestrator().run(repo_url_or_path, max_gaps=10)
    """

    def run(
        self,
        repo_url_or_path: str,
        max_gaps: int = 10,
        braintrust_logger=None,
    ) -> tuple[dict, list[GapResult]]:
        repo_path = self._prepare_repo(repo_url_or_path)

        sandbox = E2BSandbox(repo_path)
        try:
            sandbox.setup_repo(repo_url_or_path)
            sandbox.install_dependencies()
            baseline_coverage = sandbox.run_coverage_baseline()
        except Exception:
            sandbox.close()
            raise

        baseline_path = Path(repo_path) / ".coverage_baseline_agent.json"
        baseline_path.write_text(json.dumps(baseline_coverage), encoding="utf-8")

        priority_queue = GapPrioritizer().run(baseline_coverage, repo_root=repo_path)[:max_gaps]
        logger.info("Orchestrator: %d gaps to process (max=%d)", len(priority_queue), max_gaps)

        results: list[GapResult] = []
        for gap in priority_queue:
            sandbox.pause()
            gap_result, final_state = run_pipeline(
                gap=gap,
                repo_path=repo_path,
                baseline_coverage_path=str(baseline_path),
                sandbox=sandbox,
                braintrust_logger=braintrust_logger,
            )
            if (
                not gap_result.skipped
                and gap_result.phase2_scores is not None
                and gap_result.phase2_scores.target_branch_hit
            ):
                committed = self._write_test(gap, final_state, repo_path)
                if committed:
                    gap_result = gap_result.model_copy(update={"final_test_committed": True})

            results.append(gap_result)
            logger.info(
                "gap=%s skipped=%s branch_hit=%s committed=%s",
                gap.gap_id,
                gap_result.skipped,
                gap_result.phase2_scores.target_branch_hit if gap_result.phase2_scores else False,
                gap_result.final_test_committed,
            )

        sandbox.close()

        if braintrust_logger is not None:
            braintrust_logger.flush()

        scorecard = self._build_scorecard(results, repo_url_or_path, llm_cost=0.0)
        return scorecard, results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_repo(self, repo_url_or_path: str) -> str:
        """Returns a local filesystem path. Clones if given a URL."""
        if repo_url_or_path.startswith(("http://", "https://", "git@")):
            dest = tempfile.mkdtemp(prefix="coverage_agent_")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url_or_path, dest],
                check=True,
            )
            logger.info("Cloned %s to %s", repo_url_or_path, dest)
            return dest
        return repo_url_or_path

    def _write_test(self, gap: CoverageGap, final_state: dict, repo_path: str) -> bool:
        """Writes the passing test to tests/test_auto_<gap_id>.py. Returns True on success."""
        draft = final_state.get("draft_test")
        if draft is None:
            logger.warning("No draft_test in final_state for gap %s — skipping write", gap.gap_id)
            return False
        safe_id = (
            gap.gap_id.replace("/", "_").replace(":", "_").replace("->", "_").replace(".", "_")
        )
        dest = Path(repo_path) / "tests" / f"test_auto_{safe_id}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(draft.test_code, encoding="utf-8")
        logger.info("Committed test: %s", dest)
        return True

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
        dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
        cost_label = f"${llm_cost:.4f}" + (" (DRY_RUN)" if dry_run else "")
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
