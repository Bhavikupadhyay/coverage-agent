"""
Naive single-shot baseline — the apples-to-apples comparison for the pipeline.

The multi-agent pipeline burns more LLM tokens, more E2B sandbox time, and
more wall-clock time than just asking the LLM "write a test for this function".
That extra cost has to buy something measurable. This baseline is the
measurement.

What "naive" means here:
- ONE LLM call per gap, no system context, no retries, no Eval gate
- Same model, same temperature defaults as TestWriter
- Same sandbox infra (so coverage measurement is comparable)
- Same gap set (so we're scoring identical inputs)
- Same commit predicate (Credentials.should_commit)

If the pipeline doesn't beat this baseline, the multi-agent story is
unjustified and we should ship the baseline instead.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import litellm

from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.contracts.schemas import CoverageGap, ExecutionResult, GapResult
from coverage_agent.cost_tracker import CostTracker
from coverage_agent.credentials import Credentials
from coverage_agent.recommendations import gap_branch_recommendation
from coverage_agent.sandbox.e2b_runner import E2BSandbox
from coverage_agent.tpm_throttle import install_litellm_hook

logger = logging.getLogger(__name__)


_NAIVE_SYSTEM_PROMPT = (
    "You are a Python test writer. Given a function and an uncovered branch, "
    "write a single pytest test file that exercises that branch. Use mocks for "
    "any external IO. Return only Python code, no explanation."
)


def _extract_code_block(content: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _read_function_source(repo_root: str, file_path: str, target_symbol: str) -> str:
    """Reads the function source from the local clone. Falls back to the
    whole file on extraction failure. The naive baseline doesn't get the
    luxury of Jedi — that's part of what makes it naive.
    """
    full_path = Path(repo_root) / file_path
    try:
        text = full_path.read_text(encoding="utf-8")
    except Exception:
        return f"# Could not read {full_path}"

    # Quick-and-dirty: find a `def target_symbol(` or `async def target_symbol(`
    # and grab until the next top-level def/class.
    # Note: [^\S\n] = "whitespace except newline" — `\s*` would greedily consume
    # the preceding `\n`, producing indent='\n' and breaking the indent-based
    # body detection below.
    pattern = re.compile(
        rf"^([^\S\n]*)((?:async\s+)?def\s+{re.escape(target_symbol)}\s*\()",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return text  # fall back to full file

    start = match.start()
    indent = match.group(1)
    lines = text[start:].splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line.startswith(indent + " ") and not line.startswith(indent + "\t"):
            # left the function — stop
            if line.startswith(indent) and (line.lstrip().startswith(("def ", "class ", "async def "))):
                break
            if not line.startswith(indent + " ") and not line.startswith("\t"):
                break
        out.append(line)
    return "\n".join(out)


class NaiveSingleShotRunner:
    """Single-shot baseline: ask the LLM for a test, run it, decide commit.

    Mirrors the public surface of Orchestrator.run so run_benchmark.py can
    swap between them without restructuring the benchmark pipeline.
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
    ) -> tuple[dict, list[GapResult]]:
        owns_tracker = cost_tracker is None
        if owns_tracker:
            cost_tracker = CostTracker()
            cost_tracker.install()

        sandbox = E2BSandbox(
            repo_url_or_path,
            e2b_api_key=self.creds.e2b_api_key,
            offline=self.creds.is_offline,
        )
        try:
            sandbox.setup_repo(repo_url_or_path)
            sandbox.install_dependencies()
            baseline_coverage = sandbox.run_coverage_baseline()
        except Exception:
            sandbox.close()
            raise

        raw_gaps = sandbox.parse_gaps(baseline_coverage)
        # Same prioritizer as the pipeline so input ordering doesn't skew the result.
        priority_queue = GapPrioritizer(self.creds).run(raw_gaps)[:max_gaps]
        logger.info("NaiveSingleShot: %d gaps to process (max=%d)", len(priority_queue), max_gaps)

        repo_root_for_naive = self._derive_repo_root(repo_url_or_path)
        results: list[GapResult] = []
        try:
            for i, gap in enumerate(priority_queue):
                if event_callback:
                    event_callback("gap_start", "naive_baseline", 0, gap.gap_id, {
                        "gap_idx": i + 1, "total_gaps": len(priority_queue),
                    })

                test_code = self._generate_test(gap, repo_root_for_naive)
                exec_result = self._run_in_sandbox(sandbox, gap, test_code, baseline_coverage)

                committed = self.creds.should_commit(exec_result)
                skip_reason = self._explain_skip(exec_result, committed)
                recommendation = gap_branch_recommendation(gap, None, exec_result) if not committed else ""

                gap_result = GapResult(
                    gap=gap,
                    skipped=False,
                    loops_taken=1,
                    phase1_scores=None,  # naive has no Eval phase
                    phase2_scores=exec_result,
                    final_test_committed=committed,
                    test_code=test_code,
                    skip_reason=skip_reason,
                    recommendation=recommendation,
                )
                results.append(gap_result)

                if event_callback:
                    event_callback("gap_end", "naive_baseline", 0, gap.gap_id, {
                        "committed": committed,
                    })
                logger.info(
                    "naive gap=%s success=%s branch_hit=%s committed=%s",
                    gap.gap_id, exec_result.execution_success,
                    exec_result.target_branch_hit, committed,
                )
        finally:
            sandbox.close()
            if owns_tracker:
                cost_tracker.uninstall()

        llm_cost = cost_tracker.total_usd if cost_tracker is not None else 0.0
        scorecard = self._build_scorecard(results, repo_url_or_path, llm_cost=llm_cost)
        # Naive has no regression guard / summarizer — fill in empty stubs so
        # the scorecard shape matches the pipeline's for side-by-side diffing.
        scorecard["regression"] = {"regression_detected": False, "summary": "naive baseline — skipped"}
        scorecard["summary"] = {"pr_description": "", "full_summary": "Naive single-shot baseline — no LLM summary."}
        return scorecard, results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_test(self, gap: CoverageGap, repo_root: str) -> str:
        """One LLM call, no context bundle. Offline mode returns a hard fixture."""
        if self.creds.is_offline:
            # Same fixture TestWriter uses so we exercise the same code paths
            from coverage_agent.agents.test_writer import _FIXTURES_DIR
            return (_FIXTURES_DIR / "sample_test.py").read_text(encoding="utf-8")

        source = _read_function_source(repo_root, gap.file_path, gap.target_symbol)
        user_prompt = (
            f"File: {gap.file_path}\n"
            f"Function: {gap.target_symbol}\n"
            f"Uncovered branch: line {gap.branch.from_line} -> line {gap.branch.to_line}\n\n"
            f"Function source:\n```python\n{source}\n```\n\n"
            "Write a single pytest test file that covers the uncovered branch above. "
            "Return only Python code."
        )
        try:
            response = litellm.completion(
                messages=[
                    {"role": "system", "content": _NAIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **self.creds.litellm_kwargs(),
            )
            content = response.choices[0].message.content or ""
            return _extract_code_block(content)
        except Exception as exc:
            logger.warning("Naive LLM call failed for %s: %s", gap.gap_id, exc)
            # Return a guaranteed-to-fail test so this gap shows as "ran but failed"
            # rather than crashing the whole benchmark
            return "def test_naive_baseline_llm_failed():\n    assert False, 'LLM call failed'\n"

    def _run_in_sandbox(
        self,
        sandbox: E2BSandbox,
        gap: CoverageGap,
        test_code: str,
        baseline_coverage: dict,
    ) -> ExecutionResult:
        file_data = baseline_coverage.get("files", {}).get(gap.file_path, {})
        baseline_missing = file_data.get("missing_branches", None)
        baseline_pct = baseline_coverage.get("totals", {}).get("percent_covered", 0.0)
        return sandbox.run_test(
            test_code,
            gap_id=gap.gap_id,
            baseline_coverage_pct=baseline_pct,
            target_file=gap.file_path,
            target_from_line=gap.branch.from_line,
            target_to_line=gap.branch.to_line,
            baseline_missing_branches=baseline_missing,
        )

    def _explain_skip(self, exec_result: ExecutionResult, committed: bool) -> str:
        if committed:
            return ""
        if not exec_result.execution_success:
            trace = exec_result.stderr_trace or ""
            return "Naive test crashed in the sandbox: " + (trace[:200] if trace else "no stderr captured")
        if not exec_result.target_branch_hit:
            return (
                "Naive: pytest passed but the target branch was not recorded in coverage — "
                "not committed (branch proof required)."
            )
        return ""

    def _derive_repo_root(self, repo_url_or_path: str) -> str:
        """For LLM source reads. If the input is a local path, use it directly.
        For git URLs, the local clone is in the sandbox — fall back to "." so
        _read_function_source returns the whole file as the prompt.
        """
        if "://" in repo_url_or_path:
            return "."  # remote URL — naive prompt will use whole-file fallback
        return repo_url_or_path

    def _build_scorecard(
        self, results: list[GapResult], repo: str, llm_cost: float
    ) -> dict:
        targeted = len(results)
        committed = sum(1 for r in results if r.final_test_committed)
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
        cost_label = f"${llm_cost:.4f}" + (" (OFFLINE)" if self.creds.is_offline else "")
        return {
            "repo": repo,
            "mode": "naive",
            "gaps_targeted": targeted,
            "tests_committed": committed,
            "skipped": 0,
            "branch_hit_rate": f"{branch_hit_rate:.0%}",
            "avg_coverage_delta": f"+{avg_delta:.2f}%",
            "avg_loops": "1.0",
            "llm_cost": cost_label,
            "tests_passed_no_branch": passed_no_branch,
        }
