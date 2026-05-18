import logging
from typing import Callable, Optional

from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.agents.regression_guard import RegressionGuard
from coverage_agent.agents.result_summarizer import ResultSummarizer
from coverage_agent.contracts.schemas import ContextPayload, GapResult, RegressionResult, RunSummary
from coverage_agent.cost_tracker import CostTracker
from coverage_agent.credentials import Credentials
from coverage_agent.pipeline import run_pipeline
from coverage_agent.recommendations import gap_branch_recommendation
from coverage_agent.sandbox import get_sandbox
from coverage_agent.tpm_throttle import install_litellm_hook

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Outer loop: prepares sandbox, runs baseline, iterates gaps, returns results.

    All repo operations run inside E2B — no local filesystem access.
    Test code for verified gaps is returned in GapResult.test_code for
    display and download via the web UI or CLI.

    Usage:
        creds = Credentials.for_byok(body)
        scorecard, results = Orchestrator(creds).run(repo_url_or_path, max_gaps=10)
    """

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials
        if not credentials.is_offline:
            install_litellm_hook()

    def run(
        self,
        repo_url_or_path: str,
        max_gaps: int = 10,
        braintrust_logger=None,
        event_callback: Optional[Callable] = None,
        cost_tracker: Optional[CostTracker] = None,
        ignore_file: Optional[str] = None,
    ) -> tuple[dict, list[GapResult]]:
        owns_tracker = cost_tracker is None
        if owns_tracker:
            cost_tracker = CostTracker()
            cost_tracker.install()

        sandbox = get_sandbox(self.creds)
        try:
            sandbox.setup_repo(repo_url_or_path)
            sandbox.install_dependencies()
            self._validate_python_repo(sandbox)
            baseline_coverage = sandbox.run_coverage_baseline()
            baseline_passed, baseline_failed = sandbox.count_test_outcomes()
            logger.info(
                "Orchestrator: baseline suite — %d passing, %d failing",
                baseline_passed, baseline_failed,
            )
        except Exception:
            sandbox.close()
            raise

        ignore_patterns: list[str] = []
        if ignore_file:
            from coverage_agent.context.coverage_parser import load_ignore_patterns
            ignore_patterns = load_ignore_patterns(ignore_file)
            logger.info("Orchestrator: loaded %d ignore patterns from %s", len(ignore_patterns), ignore_file)

        raw_gaps = sandbox.parse_gaps(baseline_coverage, ignore_patterns=ignore_patterns)
        priority_queue = GapPrioritizer(self.creds).run(raw_gaps)[:max_gaps]
        logger.info("Orchestrator: %d gaps to process (max=%d)", len(priority_queue), max_gaps)

        results: list[GapResult] = []
        total_gaps = len(priority_queue)
        try:
            for i, gap in enumerate(priority_queue):
                if event_callback:
                    event_callback("gap_start", "orchestrator", 0, gap.gap_id, {
                        "gap_idx": i + 1,
                        "total_gaps": total_gaps,
                    })

                sandbox.pause()
                gap_result, final_state = run_pipeline(
                    gap=gap,
                    sandbox=sandbox,
                    credentials=self.creds,
                    baseline_coverage=baseline_coverage,
                    braintrust_logger=braintrust_logger,
                    event_callback=event_callback,
                )

                draft = final_state.get("draft_test")
                last_test_code = draft.test_code if draft is not None else None

                # Single source of truth lives on Credentials so the pipeline's
                # post-execution routing and this commit decision can't disagree.
                committed = (
                    not gap_result.skipped
                    and self.creds.should_commit(gap_result.phase2_scores)
                )

                ctx: ContextPayload | None = final_state.get("context")
                p2 = gap_result.phase2_scores
                recommendation = gap_branch_recommendation(gap, ctx, p2) if not committed else ""

                skip_reason = ""
                if gap_result.gap_difficulty == "hard":
                    skip_reason = "io_coupled — gap contains direct IO or network calls"
                elif gap_result.skipped:
                    if p2 and p2.execution_success and not p2.target_branch_hit:
                        skip_reason = (
                            "Retry budget exhausted: pytest passed in the sandbox but coverage "
                            f"never recorded branch {gap.branch.from_line}->{gap.branch.to_line} "
                            "as executed (commits require branch proof)."
                        )
                    elif p2 and not p2.execution_success:
                        trace = (p2.stderr_trace or "")[:200]
                        skip_reason = "Retry budget exhausted — last sandbox error: " + (
                            trace if trace else "no stderr captured"
                        )
                    else:
                        skip_reason = "Hit max retry loops — last critique: " + (
                            gap_result.phase1_scores.critique[:200]
                            if gap_result.phase1_scores and gap_result.phase1_scores.critique
                            else "no critique recorded"
                        )
                elif gap_result.phase2_scores is None:
                    skip_reason = "Pipeline did not reach the sandbox execution step."
                elif not gap_result.phase2_scores.execution_success:
                    trace = gap_result.phase2_scores.stderr_trace or ""
                    skip_reason = "Test ran but failed in the sandbox: " + (trace[:200] if trace else "no stderr captured")
                elif not committed and p2 and p2.execution_success and not p2.target_branch_hit:
                    skip_reason = (
                        "Pytest passed but the target branch was not recorded in coverage — "
                        "not committed (branch proof required)."
                    )
                elif gap_result.phase2_scores.flaky:
                    skip_reason = "Test passed inconsistently across three sandbox runs (flaky)."

                gap_result = gap_result.model_copy(update={
                    "final_test_committed": committed,
                    "test_code": last_test_code,
                    "skip_reason": skip_reason,
                    "recommendation": recommendation,
                })

                original_code = ""
                if ctx is not None:
                    original_code = ctx.primary_code

                if event_callback:
                    event_callback("gap_end", "orchestrator", 0, gap.gap_id, {
                        "original_code": original_code,
                        "committed": gap_result.final_test_committed,
                    })

                results.append(gap_result)
                logger.info(
                    "gap=%s skipped=%s branch_hit=%s committed=%s",
                    gap.gap_id,
                    gap_result.skipped,
                    gap_result.phase2_scores.target_branch_hit if gap_result.phase2_scores else False,
                    gap_result.final_test_committed,
                )
            # Final sandbox-side check: did the committed tests break any
            # previously-passing tests in the suite? Runs once, here, while
            # the sandbox is still warm. No-op when nothing was committed.
            if event_callback:
                event_callback("agent_start", "regression_guard", 0, "run", {})
            regression = RegressionGuard(self.creds).run(
                sandbox=sandbox,
                committed_results=results,
                baseline_passed=baseline_passed,
                baseline_failed=baseline_failed,
            )
            if event_callback:
                event_callback("agent_end", "regression_guard", 0, "run", {
                    "regression_detected": regression.regression_detected,
                    "summary": regression.summary,
                })
        finally:
            sandbox.close()
            if owns_tracker:
                cost_tracker.uninstall()

        if braintrust_logger is not None:
            braintrust_logger.flush()

        llm_cost = cost_tracker.total_usd if cost_tracker is not None else 0.0
        scorecard = self._build_scorecard(results, repo_url_or_path, llm_cost=llm_cost)
        scorecard["regression"] = regression.model_dump()

        # Cheap final LLM call (1 per run, not per gap) — renders the run as a
        # PR description + a longer summary. Skipped silently if it errors so a
        # transient LLM hiccup never fails the whole run.
        if event_callback:
            event_callback("agent_start", "result_summarizer", 0, "run", {})
        summary = ResultSummarizer(self.creds).run(results, scorecard, regression)
        scorecard["summary"] = summary.model_dump()
        if event_callback:
            event_callback("agent_end", "result_summarizer", 0, "run", {})

        return scorecard, results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_python_repo(self, sandbox) -> None:
        if self.creds.is_offline:
            return
        sandbox.validate_python_repo()

    def _build_scorecard(
        self, results: list[GapResult], repo: str, llm_cost: float
    ) -> dict:
        targeted = len(results)
        committed = sum(1 for r in results if r.final_test_committed)
        skipped = sum(1 for r in results if r.skipped)
        committed_rows = [r for r in results if r.final_test_committed and r.phase2_scores]
        branch_hit_rate = committed / targeted if targeted else 0.0
        avg_delta = (
            sum(r.phase2_scores.coverage_delta for r in committed_rows) / len(committed_rows)
            if committed_rows else 0.0
        )
        passed_no_branch = sum(
            1 for r in results
            if r.phase2_scores
            and r.phase2_scores.execution_success
            and not r.phase2_scores.target_branch_hit
            and not r.final_test_committed
        )
        avg_loops = sum(r.loops_taken for r in results) / targeted if targeted else 0.0
        cost_label = f"${llm_cost:.4f}" + (" (OFFLINE)" if self.creds.is_offline else "")
        return {
            "repo": repo,
            "gaps_targeted": targeted,
            "tests_committed": committed,
            "skipped": skipped,
            "branch_hit_rate": f"{branch_hit_rate:.0%}",
            "avg_coverage_delta": f"+{avg_delta:.2f}%",
            "avg_loops": f"{avg_loops:.1f}",
            "llm_cost": cost_label,
            "tests_passed_no_branch": passed_no_branch,
        }
