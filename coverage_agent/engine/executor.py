import logging
from typing import Any

from coverage_agent.credentials import Credentials
from coverage_agent.contracts import CoverageGap, DraftTest, ExecutionResult

logger = logging.getLogger(__name__)

_SYSTEM_ERROR_PATTERNS: tuple[str, ...] = (
    "Can't append to data files in parallel mode",
    "ModuleNotFoundError",
    "coverage: error:",
    "No module named",
)


def _is_system_error(stderr: str) -> bool:
    return any(p in (stderr or "") for p in _SYSTEM_ERROR_PATTERNS)


class ExecutionRunner:
    """Runs a DraftTest and measures real coverage impact.

    Wraps sandbox.run_test() with flakiness detection: if the first run
    succeeds, it is re-run twice more. Inconsistent results → flaky=True.
    """

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        draft: DraftTest,
        gap: CoverageGap,
        sandbox: Any,
        baseline_coverage: dict | None = None,
    ) -> ExecutionResult:
        baseline_coverage = baseline_coverage or {}
        file_data = baseline_coverage.get("files", {}).get(gap.file_path, {})
        baseline_missing = file_data.get("missing_branches", None)
        baseline_pct = baseline_coverage.get("totals", {}).get("percent_covered", 0.0)

        run_kwargs = dict(
            gap_id=gap.gap_id,
            baseline_coverage_pct=baseline_pct,
            target_file=gap.file_path,
            target_from_line=gap.branch.from_line,
            target_to_line=gap.branch.to_line,
            baseline_missing_branches=baseline_missing,
        )

        first = sandbox.run_test(draft.test_code, **run_kwargs)

        if not first.execution_success:
            if _is_system_error(first.stderr_trace):
                return first.model_copy(update={"is_system_error": True})
            return first

        results = [first]
        for _ in range(2):
            result = sandbox.run_test(draft.test_code, **run_kwargs)
            results.append(result)

        successes = sum(1 for r in results if r.execution_success)
        flaky = successes < len(results)

        if flaky:
            logger.warning(
                "Flaky test detected for %s (%d/%d runs passed) — marking as flaky",
                gap.gap_id,
                successes,
                len(results),
            )
            return ExecutionResult(
                execution_success=False,
                target_branch_hit=False,
                coverage_delta=0.0,
                stderr_trace="Test is flaky — inconsistent results across 3 runs.",
                flaky=True,
            )

        return first
