"""
ExecutionRunner — deterministic acceptance gate.

Writes a draft test to a temp file, runs it under coverage, verifies the
target arc was hit, then repeats for flakiness detection. No sandbox — tests
run in the caller's environment (the repo's venv on PATH).

Three-run flakiness check: run once; if it passes, run twice more. Inconsistent
results → flaky=True (not accepted).

Arc verification uses coverage.Coverage(data_file=...) — never regex on stdout.
junit XML + exit code is the pass/fail signal.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from coverage_agent.config import AgentConfig
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


def _parse_junit(junit_path: Path) -> tuple[int, int]:
    """Returns (passed, failed) counts from a junit XML file."""
    if not junit_path.exists():
        return 0, 1
    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            return 0, 1
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        passed = tests - failures - errors
        return max(0, passed), failures + errors
    except Exception as exc:
        logger.debug("junit parse error: %s", exc)
        return 0, 1


def _run_once(
    test_file: Path,
    gap: CoverageGap,
    cov_data_file: Path,
    cwd: str,
    timeout: int,
) -> ExecutionResult:
    """Runs one pytest + coverage pass and returns an ExecutionResult."""
    junit_xml = test_file.with_suffix(".xml")

    result = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            "--branch",
            "--append",
            f"--data-file={cov_data_file}",
            "-m", "pytest",
            str(test_file),
            "--tb=short", "-q",
            f"--junit-xml={junit_xml}",
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )

    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        # Combine stdout tail + stderr for the critique.
        stdout_tail = (result.stdout or "")[-800:]
        combined = (stdout_tail + "\n" + stderr).strip()
        is_sys = _is_system_error(combined)
        return ExecutionResult(
            execution_success=False,
            target_branch_hit=False,
            stderr_trace=combined[-2000:],
            is_system_error=is_sys,
        )

    # pytest passed — check arc.
    targets_hit = 0
    targets_total = 1
    try:
        import coverage as coverage_module
        cov = coverage_module.Coverage(data_file=str(cov_data_file))
        cov.load()
        data = cov.get_data()
        abs_target = str(Path(cwd) / gap.file_path)
        arcs = set(data.arcs(abs_target) or [])
        if (gap.branch.from_line, gap.branch.to_line) in arcs:
            targets_hit = 1
    except Exception as exc:
        logger.debug("Arc check failed for %s: %s", gap.gap_id, exc)

    return ExecutionResult(
        execution_success=True,
        target_branch_hit=targets_hit > 0,
        targets_hit=targets_hit,
        targets_total=targets_total,
        stderr_trace=stderr,
    )


class ExecutionRunner:
    """Runs a DraftTest and verifies it covers the target arc."""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        draft: DraftTest,
        gap: CoverageGap,
        config: AgentConfig | None = None,
        baseline_coverage: dict | None = None,
        # Legacy sandbox parameter — kept for backwards compat with old tests only.
        sandbox=None,
    ) -> ExecutionResult:
        if sandbox is not None:
            return self._run_sandbox(draft, gap, sandbox, baseline_coverage)

        cfg = config or AgentConfig()
        return self._run_subprocess(draft, gap, cfg)

    def _run_subprocess(
        self,
        draft: DraftTest,
        gap: CoverageGap,
        cfg: AgentConfig,
    ) -> ExecutionResult:
        tests_dir = Path(cfg.tests_dir)
        tests_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_test = Path(tmpdir) / "test_candidate.py"
            tmp_test.write_text(draft.test_code, encoding="utf-8")
            cov_data_file = Path(tmpdir) / ".coverage_exec"
            cwd = str(Path.cwd())
            timeout = cfg.test_timeout

            try:
                first = _run_once(tmp_test, gap, cov_data_file, cwd, timeout)
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    execution_success=False,
                    target_branch_hit=False,
                    stderr_trace=f"Test timed out after {timeout}s.",
                    is_system_error=True,
                )

            if not first.execution_success:
                if first.is_system_error:
                    return first
                return first

            # Flakiness check: run twice more.
            results = [first]
            for _ in range(cfg.flaky_runs - 1):
                try:
                    r = _run_once(tmp_test, gap, cov_data_file, cwd, timeout)
                    results.append(r)
                except subprocess.TimeoutExpired:
                    results.append(ExecutionResult(
                        execution_success=False,
                        target_branch_hit=False,
                        stderr_trace="Timeout on flakiness run.",
                    ))

            successes = sum(1 for r in results if r.execution_success)
            flaky = successes < len(results)

            if flaky:
                logger.warning(
                    "Flaky test for %s (%d/%d runs passed)",
                    gap.gap_id, successes, len(results),
                )
                return ExecutionResult(
                    execution_success=False,
                    target_branch_hit=False,
                    stderr_trace="Test is flaky — inconsistent results across runs.",
                    flaky=True,
                )

            return first

    def _run_sandbox(
        self,
        draft: DraftTest,
        gap: CoverageGap,
        sandbox,
        baseline_coverage: dict | None,
    ) -> ExecutionResult:
        """Legacy sandbox path — used by old tests only."""
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
                "Flaky test for %s (%d/%d runs passed)",
                gap.gap_id, successes, len(results),
            )
            return ExecutionResult(
                execution_success=False,
                target_branch_hit=False,
                stderr_trace="Test is flaky — inconsistent results across 3 runs.",
                flaky=True,
            )

        return first
