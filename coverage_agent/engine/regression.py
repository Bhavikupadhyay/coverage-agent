"""
RegressionGuard — deterministic final check, no LLM calls.

Runs once after all per-gap loops. Persists every accepted test into the
sandbox, re-runs the full suite, and compares pass/fail counts against the
baseline captured before any engine-written test was added.

A drop in the previously-passing count flags a regression.
"""
from __future__ import annotations

import logging
import re

from coverage_agent.contracts import GapResult, RegressionResult
from coverage_agent.credentials import Credentials

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^A-Za-z0-9_]")


def _filename_for(result: GapResult) -> str:
    """Stable, collision-resistant filename for an accepted test.

    Format: test_coverageagent_<symbol>_<from>_<to>.py
    """
    symbol = _SLUG_RE.sub("_", result.gap.target_symbol or "unknown").strip("_") or "unknown"
    branch = f"{result.gap.branch.from_line}_{result.gap.branch.to_line}"
    return f"test_coverageagent_{symbol}_{branch}.py"


class RegressionGuard:
    """Final check — does the post-acceptance suite still pass?"""

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
            logger.info("RegressionGuard: no accepted tests — skipping suite re-run")
            return RegressionResult(
                baseline_passed=baseline_passed,
                baseline_failed=baseline_failed,
                post_passed=baseline_passed,
                post_failed=baseline_failed,
                new_failures=0,
                regression_detected=False,
                summary="No accepted tests to verify.",
                skipped=True,
            )

        logger.info("RegressionGuard: persisting %d accepted tests and re-running suite", len(committed))
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
