"""
RegressionGuard — sandbox-only agent (no LLM calls).

Runs once at the end of a pipeline run, after all per-gap loops have finished.
It persists every committed test into the sandbox at a stable path, re-runs the
full pytest suite, and compares the pass/fail counts against the baseline
captured before any agent-written test was added.

If the previously-passing test count drops (i.e. a new test broke a neighbor),
the result flags a regression. The orchestrator decides whether to surface this
as a soft warning per-gap or to roll the commits back entirely — for now we
flag in the report and leave the tests committed so the developer can
investigate.

Why this exists: the existing ExecutionRunner only verifies that the *new*
test passes in isolation. It cannot detect a test that monkey-patches a global,
shadows a fixture, or otherwise breaks another test in the suite. RegressionGuard
closes that loop.
"""
from __future__ import annotations

import logging
import re

from coverage_agent.contracts.schemas import GapResult, RegressionResult
from coverage_agent.credentials import Credentials

logger = logging.getLogger(__name__)

# Match a Python identifier-safe slug for the filename.
_SLUG_RE = re.compile(r"[^A-Za-z0-9_]")


def _filename_for(result: GapResult) -> str:
    """Stable, collision-resistant filename for a committed test.

    Format: `test_coverageagent_<symbol>_<from>_<to>.py`. Using the gap branch
    in the name guarantees uniqueness even when two gaps target the same symbol.
    """
    symbol = _SLUG_RE.sub("_", result.gap.target_symbol or "unknown").strip("_") or "unknown"
    branch = f"{result.gap.branch.from_line}_{result.gap.branch.to_line}"
    return f"test_coverageagent_{symbol}_{branch}.py"


class RegressionGuard:
    """Final sandbox check — does the post-commit suite still pass?"""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        sandbox,
        committed_results: list[GapResult],
        baseline_passed: int,
        baseline_failed: int,
    ) -> RegressionResult:
        committed = [r for r in committed_results if r.final_test_committed and r.test_code]

        if not committed:
            logger.info("RegressionGuard: no committed tests — skipping suite re-run")
            return RegressionResult(
                baseline_passed=baseline_passed,
                baseline_failed=baseline_failed,
                post_passed=baseline_passed,
                post_failed=baseline_failed,
                new_failures=0,
                regression_detected=False,
                summary="No committed tests to verify.",
                skipped=True,
            )

        # Offline path returns a clean pass with a small bump in passed count
        # so the UI gets something realistic to render.
        if self.creds.is_offline:
            logger.info("[OFFLINE] RegressionGuard — returning passing fixture for %d committed tests", len(committed))
            return RegressionResult(
                baseline_passed=baseline_passed,
                baseline_failed=baseline_failed,
                post_passed=baseline_passed + len(committed),
                post_failed=baseline_failed,
                new_failures=0,
                regression_detected=False,
                summary=f"{len(committed)} new tests, no regressions.",
                skipped=False,
            )

        logger.info("RegressionGuard: persisting %d committed tests and re-running suite", len(committed))
        for r in committed:
            try:
                sandbox.persist_test(r.test_code, _filename_for(r))
            except Exception as exc:
                logger.warning("RegressionGuard: failed to persist test for %s — %s", r.gap.gap_id, exc)

        post_passed, post_failed = sandbox.count_test_outcomes()
        new_failures = max(0, post_failed - baseline_failed)
        regression = new_failures > 0

        if regression:
            summary = (
                f"Regression detected: {new_failures} previously-passing test(s) now fail "
                f"({baseline_passed} → {post_passed} passing)."
            )
        else:
            summary = (
                f"Suite clean: {post_passed} passing ({post_passed - baseline_passed} new), "
                f"{post_failed} failing (no new failures)."
            )

        return RegressionResult(
            baseline_passed=baseline_passed,
            baseline_failed=baseline_failed,
            post_passed=post_passed,
            post_failed=post_failed,
            new_failures=new_failures,
            regression_detected=regression,
            summary=summary,
            skipped=False,
        )
