"""
RegressionGuard — deterministic final check, no LLM calls.

Writes accepted tests to tests_dir, re-runs the full suite, and compares
pass/fail counts against the pre-run baseline. A drop in previously-passing
tests flags a regression.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from coverage_agent.config import AgentConfig
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


def _parse_junit_counts(junit_path: Path) -> tuple[int, int]:
    """Returns (passed, failed) from a junit XML file."""
    if not junit_path.exists():
        return 0, 0
    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            return 0, 0
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        passed = tests - failures - errors - skipped
        return max(0, passed), failures + errors
    except Exception as exc:
        logger.debug("junit parse error: %s", exc)
        return 0, 0


def _run_test_command(
    test_command: str,
    junit_xml: Path,
    cwd: str,
    timeout: int = 300,
) -> tuple[int, int]:
    """Runs test_command and returns (passed, failed) from junit output."""
    cmd = test_command.split() + [f"--junit-xml={junit_xml}", "-q"]
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("RegressionGuard: test command timed out after %ds", timeout)
        return 0, 1
    return _parse_junit_counts(junit_xml)


class RegressionGuard:
    """Final check — does the post-acceptance suite still pass?"""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        committed_results: list[GapResult],
        baseline_passed: int,
        baseline_failed: int,
        config: AgentConfig | None = None,
        repo_root: str = ".",
        # Legacy sandbox parameter — kept for backwards compat with old tests only.
        sandbox=None,
    ) -> RegressionResult:
        if sandbox is not None:
            return self._run_sandbox(committed_results, baseline_passed, baseline_failed, sandbox)

        cfg = config or AgentConfig()
        return self._run_subprocess(committed_results, baseline_passed, baseline_failed, cfg, repo_root)

    def _run_subprocess(
        self,
        committed_results: list[GapResult],
        baseline_passed: int,
        baseline_failed: int,
        cfg: AgentConfig,
        repo_root: str,
    ) -> RegressionResult:
        committed = [r for r in committed_results if r.accepted and r.test_code]

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

        # Write accepted tests to tests_dir.
        tests_dir = Path(repo_root) / cfg.tests_dir
        tests_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for r in committed:
            dest = tests_dir / _filename_for(r)
            try:
                dest.write_text(r.test_code, encoding="utf-8")
                written.append(dest)
                logger.debug("RegressionGuard: wrote %s", dest)
            except Exception as exc:
                logger.warning("RegressionGuard: failed to write %s — %s", dest, exc)

        logger.info(
            "RegressionGuard: wrote %d tests, re-running suite: %s",
            len(written), cfg.test_command,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            junit_xml = Path(tmpdir) / "regression_junit.xml"
            post_passed, post_failed = _run_test_command(
                cfg.test_command,
                junit_xml,
                cwd=repo_root,
            )

        new_failures = max(0, post_failed - baseline_failed)
        regression = new_failures > 0

        if regression:
            summary = (
                f"Regression detected: {new_failures} previously-passing test(s) now fail "
                f"({baseline_passed} → {post_passed} passing)."
            )
        else:
            summary = (
                f"Suite clean: {post_passed} passing ({post_passed - baseline_passed:+d} new), "
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

    def _run_sandbox(
        self,
        committed_results: list[GapResult],
        baseline_passed: int,
        baseline_failed: int,
        sandbox,
    ) -> RegressionResult:
        """Legacy sandbox path — used by old tests only."""
        committed = [r for r in committed_results if r.accepted and r.test_code]

        if not committed:
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
