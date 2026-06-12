"""
ExecutionRunner — deterministic acceptance gate.

Writes a draft test to a temp file, runs it under coverage, verifies the
target arc was hit, then repeats for flakiness detection. No sandbox — tests
run in the caller's environment (the repo's venv on PATH).

Three-run flakiness check: run once; if it passes, run twice more. Inconsistent
results → flaky=True (not accepted).

Arc verification uses coverage.Coverage(data_file=...) — never regex on stdout.
junit XML + exit code is the pass/fail signal.

Target-hit logic is gap-kind-aware (see _check_targets). Both the outer
Executor and the ReAct agent's inner run_candidate use this same helper so
the acceptance rule is defined exactly once.
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

# Per-run arc-hit data for cluster verification.  Keyed by id(ExecutionResult).
# Entries are written by _run_once and consumed (and removed) by
# _cluster_results_from_exec.  Private implementation detail — never part of
# any external contract.
_cluster_arc_store: dict[int, dict] = {}


def _check_targets(
    gap: CoverageGap,
    cov_data,
    abs_target: str,
) -> tuple[int, int]:
    """Returns (targets_hit, targets_total) based on gap.kind.

    kind="branch":
        Exact arc membership — unchanged from the original logic.
        targets_total=1; targets_hit=1 iff the arc is executed.

    kind="function" or kind="line":
        Verified by executed lines, not arcs.  Import alone executes the
        'def' line (gap.branch.from_line) without running any body code, so
        the def line is excluded from the count — only body lines matter.
        targets_total = len(body lines); targets_hit = body lines executed.
        Accept when targets_hit >= 1 AND targets_hit/targets_total >= 0.5.
        Rationale: a real test that calls the function executes the main path
        (well over half the body); an import-only or trivially-passing test
        doesn't.  Avoids false accepts while not penalising short functions.

    If gap.surrounding_lines is empty for a function/line gap, treats as not
    hit (targets_total=0) and logs a debug message — never divides by zero.
    """
    if gap.kind == "branch":
        arcs = set(cov_data.arcs(abs_target) or [])
        hit = 1 if (gap.branch.from_line, gap.branch.to_line) in arcs else 0
        return hit, 1

    # kind == "function" or kind == "line"
    def_line = gap.branch.from_line
    body_lines = [ln for ln in gap.surrounding_lines if ln != def_line]
    if not body_lines:
        logger.debug(
            "gap %s has no body lines in surrounding_lines — treating as not hit",
            gap.gap_id,
        )
        return 0, 0

    executed = set(cov_data.lines(abs_target) or [])
    hit = sum(1 for ln in body_lines if ln in executed)
    total = len(body_lines)
    return hit, total


def _cluster_results_from_exec(
    cluster: list,
    exec_result: "ExecutionResult | None",
) -> list:
    """Returns one ExecutionResult per gap in cluster based on exec_result.

    When exec_result is None (pipeline was skipped or errored before execution),
    every gap gets a not-hit result.  Otherwise each gap's arc-hit status is read
    independently from exec_result's coverage data — but because the executor ran
    only once we derive per-gap hit from the stored target arcs via the same
    _check_targets helper, re-loading the coverage data from exec_result._cov_data
    when available, or falling back to target_branch_hit for single-gap clusters.
    """
    if exec_result is None:
        return [
            ExecutionResult(
                execution_success=False,
                target_branch_hit=False,
            )
            for _ in cluster
        ]

    # Fast path: single-gap cluster — just replicate the result.
    if len(cluster) <= 1:
        return [exec_result]

    # For multi-gap clusters we need to check each arc individually.
    # _run_once stored per-arc hit data in _cluster_arc_store keyed by the
    # result's id().  Consume and remove the entry to avoid unbounded growth.
    arc_hits: dict = _cluster_arc_store.pop(id(exec_result), {})

    results = []
    for gap in cluster:
        arc_key = (gap.branch.from_line, gap.branch.to_line)
        if arc_hits:
            hit = arc_hits.get(arc_key, False)
        else:
            # No per-gap data — conservatively, only the primary gap inherits the hit.
            hit = (gap is cluster[0]) and exec_result.target_branch_hit
        results.append(ExecutionResult(
            execution_success=exec_result.execution_success,
            target_branch_hit=hit,
            targets_hit=1 if hit else 0,
            targets_total=1,
            stderr_trace=exec_result.stderr_trace,
            flaky=exec_result.flaky,
            is_system_error=exec_result.is_system_error,
        ))
    return results


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
    python_executable: str = "",
    cluster: list | None = None,
) -> ExecutionResult:
    """Runs one pytest + coverage pass and returns an ExecutionResult.

    When cluster has >1 gaps, checks every arc in the cluster and stores the
    per-arc hit map in the result's _cluster_arc_hits attribute so callers can
    build individual GapResults without re-reading coverage data.
    """
    junit_xml = test_file.with_suffix(".xml")
    python = python_executable or sys.executable

    result = subprocess.run(
        [
            python, "-m", "coverage", "run",
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

    # pytest passed — check target coverage by gap kind.
    targets_hit = 0
    targets_total = 1
    arc_hits: dict = {}
    try:
        import coverage as coverage_module
        cov = coverage_module.Coverage(data_file=str(cov_data_file))
        cov.load()
        data = cov.get_data()
        abs_target = str(Path(cwd) / gap.file_path)
        targets_hit, targets_total = _check_targets(gap, data, abs_target)

        # Build per-arc hit map for every gap in the cluster.
        effective_cluster = cluster if cluster and len(cluster) > 1 else None
        if effective_cluster:
            for g in effective_cluster:
                h, _ = _check_targets(g, data, abs_target)
                # For branch gaps (h, t) = (0 or 1, 1); for others use >=1 and >=0.5 rule.
                if g.kind == "branch":
                    arc_hits[(g.branch.from_line, g.branch.to_line)] = h >= 1
                else:
                    t_h, t_t = _check_targets(g, data, abs_target)
                    arc_hits[(g.branch.from_line, g.branch.to_line)] = (
                        t_h >= 1 and t_t > 0 and t_h / t_t >= 0.5
                    )
    except Exception as exc:
        logger.debug("Coverage check failed for %s: %s", gap.gap_id, exc)

    # Primary acceptance: >=1 arc hit (for single-gap this is the usual rule).
    # For clusters, target_branch_hit=True if ANY arc in the cluster was hit —
    # this drives the should_commit gate so the test is kept.
    if arc_hits:
        any_hit = any(arc_hits.values())
        primary_hit = arc_hits.get((gap.branch.from_line, gap.branch.to_line), False)
        accepted = any_hit
    else:
        primary_hit = (
            targets_hit >= 1
            and targets_total > 0
            and targets_hit / targets_total >= 0.5
        )
        accepted = primary_hit

    exec_result = ExecutionResult(
        execution_success=True,
        target_branch_hit=accepted,
        targets_hit=targets_hit,
        targets_total=targets_total,
        stderr_trace=stderr,
    )
    if arc_hits:
        _cluster_arc_store[id(exec_result)] = arc_hits
    return exec_result


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
        cluster: list | None = None,
    ) -> ExecutionResult:
        if sandbox is not None:
            return self._run_sandbox(draft, gap, sandbox, baseline_coverage)

        cfg = config or AgentConfig()
        return self._run_subprocess(draft, gap, cfg, cluster=cluster)

    def _run_subprocess(
        self,
        draft: DraftTest,
        gap: CoverageGap,
        cfg: AgentConfig,
        cluster: list | None = None,
    ) -> ExecutionResult:
        tests_dir = Path(cfg.tests_dir)
        tests_dir.mkdir(parents=True, exist_ok=True)

        python_executable = cfg.python_executable or sys.executable

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_test = Path(tmpdir) / "test_candidate.py"
            tmp_test.write_text(draft.test_code, encoding="utf-8")
            cov_data_file = Path(tmpdir) / ".coverage_exec"
            cwd = str(Path.cwd())
            timeout = cfg.test_timeout

            try:
                first = _run_once(tmp_test, gap, cov_data_file, cwd, timeout, python_executable, cluster=cluster)
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
                    r = _run_once(tmp_test, gap, cov_data_file, cwd, timeout, python_executable, cluster=cluster)
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
