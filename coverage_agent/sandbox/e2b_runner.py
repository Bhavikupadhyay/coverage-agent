import json
import logging
import os
from pathlib import Path

from coverage_agent.contracts.schemas import ExecutionResult

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


class E2BSandbox:
    """
    Manages the E2B sandbox lifecycle for a single repo benchmark run.

    The sandbox is created once per repo by the Orchestrator and reused
    across all gap iterations. All methods check DRY_RUN before making
    any real E2B calls.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self._sandbox = None

        if _is_dry_run():
            logger.info("[DRY_RUN] E2BSandbox.__init__ — skipping real sandbox creation for %s", repo_path)
            return

        try:
            from e2b import Sandbox
            self._sandbox = Sandbox()
            logger.info("E2B sandbox created for %s", repo_path)
        except Exception as exc:
            logger.error("Failed to create E2B sandbox: %s", exc)
            raise

    def install_dependencies(self) -> None:
        """Runs pip install -e .[dev] inside the sandbox. Called once per repo."""
        if _is_dry_run():
            logger.info("[DRY_RUN] install_dependencies — skipping")
            return

        result = self._sandbox.process.start_and_wait(
            f"cd /repo && pip install -e '.[dev]' -q"
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Dependency installation failed (exit {result.exit_code}):\n{result.stderr}"
            )
        logger.info("Dependencies installed successfully")

    def run_coverage_baseline(self) -> dict:
        """
        Runs the full test suite with branch coverage once per repo.
        Returns the parsed coverage.json dict.
        """
        if _is_dry_run():
            logger.info("[DRY_RUN] run_coverage_baseline — returning fixture")
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))

        result = self._sandbox.process.start_and_wait(
            "cd /repo && coverage run --branch -m pytest -q && coverage json -o coverage.json"
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Coverage baseline failed (exit {result.exit_code}):\n{result.stderr}"
            )

        coverage_raw = self._sandbox.filesystem.read("/repo/coverage.json")
        return json.loads(coverage_raw)

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
        """
        Writes the test to the sandbox, runs it, measures coverage delta and
        whether the specific target branch was newly covered, then deletes the
        test file. Returns an ExecutionResult.
        """
        if _is_dry_run():
            logger.info("[DRY_RUN] run_test — returning fixture ExecutionResult for gap %s", gap_id)
            return ExecutionResult(
                execution_success=True,
                target_branch_hit=True,
                coverage_delta=0.4,
                stderr_trace="",
                flaky=False,
            )

        test_path = "/repo/tests/test_coverageagent_auto.py"
        try:
            self._sandbox.filesystem.write(test_path, test_code)

            result = self._sandbox.process.start_and_wait(
                f"cd /repo && coverage run --branch -m pytest {test_path} -q "
                f"&& coverage json -o coverage_after.json"
            )

            execution_success = result.exit_code == 0
            stderr_trace = result.stderr or ""

            coverage_delta = 0.0
            target_branch_hit = False

            if execution_success:
                try:
                    after_raw = self._sandbox.filesystem.read("/repo/coverage_after.json")
                    after = json.loads(after_raw)
                    after_pct = after.get("totals", {}).get("percent_covered", 0.0)
                    coverage_delta = round(after_pct - baseline_coverage_pct, 2)

                    if target_file and target_file in after.get("files", {}):
                        newly_executed = after["files"][target_file].get("executed_branches", [])
                        target_branch = [target_from_line, target_to_line]
                        was_missing = (
                            baseline_missing_branches is None
                            or target_branch in baseline_missing_branches
                        )
                        target_branch_hit = was_missing and target_branch in newly_executed
                except Exception as exc:
                    logger.warning("Could not parse post-run coverage: %s", exc)

            return ExecutionResult(
                execution_success=execution_success,
                target_branch_hit=target_branch_hit,
                coverage_delta=coverage_delta,
                stderr_trace=stderr_trace,
                flaky=False,
            )
        finally:
            try:
                self._sandbox.filesystem.remove(test_path)
            except Exception:
                pass

    def close(self) -> None:
        """Kills the E2B sandbox VM. Called by Orchestrator after all gaps are done."""
        if _is_dry_run():
            logger.info("[DRY_RUN] close — skipping")
            return

        if self._sandbox:
            try:
                self._sandbox.close()
                logger.info("E2B sandbox closed")
            except Exception as exc:
                logger.warning("Error closing sandbox: %s", exc)
