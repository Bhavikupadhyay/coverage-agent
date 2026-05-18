from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from coverage_agent.contracts.schemas import CoverageGap, ExecutionResult
from coverage_agent.sandbox.e2b_runner import _parse_pytest_counts, coverage_source_cli_flag

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class LocalSandbox:
    """
    Subprocess-based sandbox that mirrors the E2BSandbox interface.

    Clones or copies the target repo into a temp dir, creates a fresh
    venv, and runs all coverage commands locally. Uses a COVERAGE_RCFILE
    override to force parallel=false, which sidesteps Bug #1 (coverage.py
    refusing to write to a non-parallel-named file when the repo's own
    config has parallel=true).

    This is the correct backend for benchmarks on trusted repos — zero cost,
    no external APIs, and faster than E2B for iterative local runs.
    """

    def __init__(self, offline: bool = False) -> None:
        self._offline = offline
        self._run_dir = Path(tempfile.mkdtemp(prefix="cov_agent_"))
        self._repo_dir = self._run_dir / "repo"
        self._venv_dir = self._run_dir / "venv"
        self._pip = str(self._venv_dir / "bin" / "pip")
        self._python = str(self._venv_dir / "bin" / "python")
        self._coverage_bin = str(self._venv_dir / "bin" / "coverage")
        self._pytest_bin = str(self._venv_dir / "bin" / "pytest")
        self._coverage_rc = self._run_dir / "coverage_override.ini"
        self._coverage_rc.write_text("[run]\nparallel = false\nbranch = true\n")
        self._paused_id: Optional[str] = None  # interface compat only

    # ------------------------------------------------------------------
    # Repo setup
    # ------------------------------------------------------------------

    def setup_repo(self, repo_url_or_path: str) -> None:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.setup_repo — skipping")
            return
        if repo_url_or_path.startswith(("http://", "https://", "git@")):
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url_or_path, str(self._repo_dir)],
                timeout=120,
                check=True,
            )
            logger.info("Cloned %s → %s", repo_url_or_path, self._repo_dir)
        else:
            shutil.copytree(repo_url_or_path, str(self._repo_dir))
            logger.info("Copied %s → %s", repo_url_or_path, self._repo_dir)

        # Ensure the tests directory exists so run_test() can write there
        (self._repo_dir / "tests").mkdir(exist_ok=True)

    def install_dependencies(self) -> None:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.install_dependencies — skipping")
            return
        subprocess.run(
            [sys.executable, "-m", "venv", str(self._venv_dir)],
            check=True,
            timeout=60,
        )
        # Try extras first; fall back to bare editable install
        result = subprocess.run(
            [self._pip, "install", "-e", ".[dev,test,tests]",
             "pytest", "pytest-cov", "coverage", "jedi", "-q",
             "--no-warn-script-location"],
            cwd=str(self._repo_dir),
            timeout=300,
        )
        if result.returncode != 0:
            subprocess.run(
                [self._pip, "install", "-e", ".",
                 "pytest", "pytest-cov", "coverage", "jedi", "-q",
                 "--no-warn-script-location"],
                cwd=str(self._repo_dir),
                timeout=300,
                check=True,
            )
        logger.info("LocalSandbox: dependencies installed in %s", self._venv_dir)

    def validate_python_repo(self) -> None:
        if self._offline:
            return
        count = sum(1 for _ in self._repo_dir.rglob("*.py"))
        if count == 0:
            raise ValueError(
                "Not a Python repository — no .py files found. "
                "CoverageAgent only supports Python projects."
            )

    # ------------------------------------------------------------------
    # Coverage baseline
    # ------------------------------------------------------------------

    def run_coverage_baseline(self) -> dict:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.run_coverage_baseline — returning fixture")
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))

        env = {**os.environ, "COVERAGE_RCFILE": str(self._coverage_rc)}
        # Baseline: allow test failures (exit 1 is fine — some repos have pre-existing failures)
        subprocess.run(
            [self._coverage_bin, "run", "--branch", "-m", "pytest", "-q", "--tb=no"],
            cwd=str(self._repo_dir),
            env=env,
            timeout=300,
        )
        subprocess.run(
            [self._coverage_bin, "json", "-o", "coverage_baseline.json"],
            cwd=str(self._repo_dir),
            env=env,
            timeout=60,
            check=True,
        )
        result = json.loads((self._repo_dir / "coverage_baseline.json").read_text())
        logger.info(
            "LocalSandbox: baseline coverage %.1f%% (%d files)",
            result.get("totals", {}).get("percent_covered", 0.0),
            len(result.get("files", {})),
        )
        return result

    # ------------------------------------------------------------------
    # Per-test execution
    # ------------------------------------------------------------------

    def run_test(
        self,
        test_code: str,
        gap_id: str = "unknown",
        baseline_coverage_pct: float = 0.0,
        target_file: str = "",
        target_from_line: int = 0,
        target_to_line: int = 0,
        baseline_missing_branches: list | None = None,
    ) -> ExecutionResult:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.run_test — returning fixture for %s", gap_id)
            return ExecutionResult(
                execution_success=True,
                target_branch_hit=True,
                coverage_delta=0.4,
                stderr_trace="",
                flaky=False,
            )

        test_path = self._repo_dir / "tests" / "test_coverageagent_auto.py"
        cov_file = str(self._repo_dir / ".cov_agent_per_test")
        src_flag = coverage_source_cli_flag(target_file)
        env = {
            **os.environ,
            "COVERAGE_RCFILE": str(self._coverage_rc),
            "COVERAGE_FILE": cov_file,
        }

        try:
            test_path.write_text(test_code, encoding="utf-8")

            execution_success = False
            stderr_trace = ""
            try:
                cmd = [self._coverage_bin, "run", "--branch"]
                if src_flag:
                    cmd.append(src_flag.strip())
                cmd += ["-m", "pytest", str(test_path), "-q"]
                result = subprocess.run(
                    cmd,
                    cwd=str(self._repo_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                stdout_out = result.stdout or ""
                stderr_out = result.stderr or ""
                stderr_trace = (stderr_out + "\n" + stdout_out).strip() if (stderr_out or stdout_out) else ""
                execution_success = result.returncode == 0
            except subprocess.TimeoutExpired:
                execution_success = False
                stderr_trace = "Test timed out after 120s"
            except Exception as exc:
                execution_success = False
                stderr_trace = str(exc)

            # Generate per-test coverage JSON (no combine needed — parallel=false)
            if execution_success:
                try:
                    subprocess.run(
                        [self._coverage_bin, "json", "-o", "coverage_after.json"],
                        cwd=str(self._repo_dir),
                        env=env,
                        timeout=60,
                    )
                except Exception:
                    pass

            coverage_delta = 0.0
            target_branch_hit = False

            if execution_success:
                after_path = self._repo_dir / "coverage_after.json"
                try:
                    after = json.loads(after_path.read_text())
                    if target_file and target_file in after.get("files", {}):
                        newly_executed = after["files"][target_file].get("executed_branches", [])
                        target_branch = [target_from_line, target_to_line]
                        was_missing = (
                            baseline_missing_branches is None
                            or target_branch in baseline_missing_branches
                        )
                        target_branch_hit = was_missing and target_branch in newly_executed

                        if baseline_missing_branches:
                            newly_covered_missing = [
                                b for b in newly_executed if b in baseline_missing_branches
                            ]
                            denom = max(len(baseline_missing_branches), 1)
                            coverage_delta = round(
                                len(newly_covered_missing) / denom * 100.0, 2
                            )
                        elif target_branch_hit:
                            coverage_delta = 1.0
                except Exception as exc:
                    logger.warning("LocalSandbox: could not parse post-run coverage: %s", exc)

            return ExecutionResult(
                execution_success=execution_success,
                target_branch_hit=target_branch_hit,
                coverage_delta=coverage_delta,
                stderr_trace=stderr_trace,
                flaky=False,
            )
        finally:
            try:
                test_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Test suite outcome counting
    # ------------------------------------------------------------------

    def count_test_outcomes(self) -> tuple[int, int]:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.count_test_outcomes — returning fixture (12, 0)")
            return 12, 0
        result = subprocess.run(
            [self._pytest_bin, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=str(self._repo_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return _parse_pytest_counts(result.stdout + "\n" + result.stderr)

    # ------------------------------------------------------------------
    # Committed test persistence
    # ------------------------------------------------------------------

    def persist_test(self, test_code: str, filename: str) -> None:
        dest = self._repo_dir / "tests" / filename
        dest.write_text(test_code, encoding="utf-8")
        logger.info("LocalSandbox: persisted test → %s", dest)

    # ------------------------------------------------------------------
    # Gap parsing and context building (delegates to local modules)
    # ------------------------------------------------------------------

    def parse_gaps(self, coverage_json: dict, ignore_patterns: list[str] | None = None) -> list[CoverageGap]:
        from coverage_agent.context.coverage_parser import load_ignore_patterns, parse_coverage
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.parse_gaps — returning fixture gaps")
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            fixture_json = json.loads(fixture_path.read_text(encoding="utf-8"))
            return parse_coverage(fixture_json, repo_root=".")

        patterns = ignore_patterns if ignore_patterns is not None else self._load_repo_ignore()
        return parse_coverage(coverage_json, repo_root=str(self._repo_dir), ignore_patterns=patterns)

    def _load_repo_ignore(self) -> list[str]:
        """Auto-discovers .coverageagentignore in the cloned repo root."""
        from coverage_agent.context.coverage_parser import load_ignore_patterns
        ignore_file = self._repo_dir / ".coverageagentignore"
        if ignore_file.exists():
            patterns = load_ignore_patterns(str(ignore_file))
            logger.info("Loaded %d ignore patterns from %s", len(patterns), ignore_file)
            return patterns
        return []

    def build_context(
        self,
        file_path: str,
        target_symbol: str,
        depth: int,
        from_line: int | None = None,
    ) -> dict:
        if self._offline:
            logger.info("[OFFLINE] LocalSandbox.build_context — returning fixture context")
            fixture_path = _FIXTURES_DIR / "sample_context.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))
        from coverage_agent.context.jedi_graph import build_context
        result = build_context(
            file_path,
            target_symbol,
            depth=depth,
            repo_root=str(self._repo_dir),
            from_line=from_line,
        )
        return result.model_dump()

    # ------------------------------------------------------------------
    # Lifecycle — pause/resume are no-ops (interface compat with E2BSandbox)
    # ------------------------------------------------------------------

    def pause(self) -> str:
        return "local-sandbox-id"

    def resume(self, sandbox_id: str) -> None:
        pass

    def close(self) -> None:
        shutil.rmtree(self._run_dir, ignore_errors=True)
        logger.info("LocalSandbox: cleaned up %s", self._run_dir)
