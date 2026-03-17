import logging

from coverage_agent.contracts.schemas import CoverageGap, DraftTest, ExecutionResult
from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


class ExecutionRunner:
    """
    Runs a DraftTest inside the E2B sandbox and measures real coverage impact.

    Wraps E2BSandbox.run_test() with flakiness detection: if the first run
    succeeds, it is re-run twice more. If results are inconsistent the test
    is marked flaky and skipped (execution_success=False, flaky=True).
    """

    def run(self, draft: DraftTest, gap: CoverageGap, sandbox: E2BSandbox) -> ExecutionResult:
        first = sandbox.run_test(draft.test_code, gap_id=gap.gap_id)

        if not first.execution_success:
            # Failed on first attempt — return immediately, no flakiness check needed
            return first

        # Re-run twice to detect flakiness
        results = [first]
        for _ in range(2):
            result = sandbox.run_test(draft.test_code, gap_id=gap.gap_id)
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

        # All 3 runs passed — return the first result (coverage delta is consistent)
        return first
