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

    Live mode uses the E2B SDK v2+ API:
      - Sandbox.create() to spin up
      - sandbox.commands.run(cmd, cwd=..., timeout=...) to execute shell commands
      - sandbox.files.write/read/remove for filesystem operations
      - sandbox.kill() to terminate
      - sandbox.pause() / Sandbox.connect(sandbox_id) for cost optimization
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self._sandbox = None
        self._paused_id: str | None = None

        if _is_dry_run():
            logger.info("[DRY_RUN] E2BSandbox.__init__ — skipping real sandbox creation for %s", repo_path)
            return

        try:
            from e2b import Sandbox
            template_id = os.environ.get("E2B_TEMPLATE_ID") or None
            self._sandbox = Sandbox.create(template_id) if template_id else Sandbox.create()
            logger.info("E2B sandbox created (id=%s) for %s", self._sandbox.sandbox_id, repo_path)
        except Exception as exc:
            logger.error("Failed to create E2B sandbox: %s", exc)
            raise

    def _ensure_active(self) -> None:
        """Resumes the sandbox if it was paused. No-op in dry-run or if already active."""
        if _is_dry_run() or self._sandbox is not None:
            return
        if self._paused_id is None:
            raise RuntimeError("Sandbox is not active and has no paused ID to resume from")
        self.resume(self._paused_id)
        self._paused_id = None

    def install_dependencies(self) -> None:
        """Runs pip install -e .[dev] inside the sandbox. Called once per repo."""
        if _is_dry_run():
            logger.info("[DRY_RUN] install_dependencies — skipping")
            return

        self._ensure_active()
        result = self._sandbox.commands.run(
            "pip install -e '.[dev]' -q",
            cwd="/repo",
            timeout=300,
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

        self._ensure_active()
        result = self._sandbox.commands.run(
            "coverage run --branch -m pytest -q && coverage json -o coverage.json",
            cwd="/repo",
            timeout=300,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Coverage baseline failed (exit {result.exit_code}):\n{result.stderr}"
            )

        coverage_raw = self._sandbox.files.read("/repo/coverage.json")
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

        self._ensure_active()
        test_path = "/repo/tests/test_coverageagent_auto.py"
        try:
            self._sandbox.files.write(test_path, test_code)

            result = self._sandbox.commands.run(
                f"coverage run --branch -m pytest {test_path} -q "
                f"&& coverage json -o coverage_after.json",
                cwd="/repo",
                timeout=120,
            )

            execution_success = result.exit_code == 0
            stderr_trace = result.stderr or ""

            coverage_delta = 0.0
            target_branch_hit = False

            if execution_success:
                try:
                    after_raw = self._sandbox.files.read("/repo/coverage_after.json")
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
                self._sandbox.files.remove(test_path)
            except Exception:
                pass

    def upload_repo(self, local_path: str) -> None:
        """Uploads a local repo directory to /repo in the sandbox. Skips .git, __pycache__, *.pyc."""
        if _is_dry_run():
            logger.info("[DRY_RUN] upload_repo — skipping")
            return
        self._ensure_active()
        _SKIP = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}
        root = Path(local_path)
        uploaded = 0
        for src in root.rglob("*"):
            if src.is_dir():
                continue
            if any(part in _SKIP for part in src.parts):
                continue
            if src.suffix in (".pyc", ".pyo"):
                continue
            rel = src.relative_to(root)
            dest = f"/repo/{rel}"
            try:
                self._sandbox.files.write(dest, src.read_bytes())
                uploaded += 1
            except Exception as exc:
                logger.warning("Could not upload %s: %s", rel, exc)
        logger.info("Uploaded %d files to sandbox /repo", uploaded)

    def setup_repo(self, repo_url_or_path: str) -> None:
        """Clones a repo URL into /repo, or uploads a local path to /repo."""
        if _is_dry_run():
            logger.info("[DRY_RUN] setup_repo — skipping")
            return
        self._ensure_active()
        if repo_url_or_path.startswith(("http://", "https://", "git@")):
            result = self._sandbox.commands.run(
                f"git clone --depth 1 {repo_url_or_path} /repo",
                timeout=120,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"git clone failed:\n{result.stderr}")
            logger.info("Cloned %s into sandbox /repo", repo_url_or_path)
        else:
            self.upload_repo(repo_url_or_path)

    def pause(self) -> str:
        """
        Pauses the sandbox to stop billing during LLM calls.
        Returns the sandbox_id needed to resume. Sets internal sandbox to None.
        """
        if _is_dry_run():
            logger.info("[DRY_RUN] pause — skipping")
            return "dry-run-sandbox-id"

        sandbox_id = self._sandbox.sandbox_id
        self._sandbox.pause()
        self._sandbox = None
        self._paused_id = sandbox_id
        logger.info("E2B sandbox paused (id=%s)", sandbox_id)
        return sandbox_id

    def resume(self, sandbox_id: str) -> None:
        """Resumes a previously paused sandbox by its ID."""
        if _is_dry_run():
            logger.info("[DRY_RUN] resume — skipping")
            return

        from e2b import Sandbox
        self._sandbox = Sandbox.connect(sandbox_id)
        logger.info("E2B sandbox resumed (id=%s)", sandbox_id)

    def close(self) -> None:
        """Kills the E2B sandbox VM. Called by Orchestrator after all gaps are done."""
        if _is_dry_run():
            logger.info("[DRY_RUN] close — skipping")
            return

        if self._sandbox:
            try:
                self._sandbox.kill()
                logger.info("E2B sandbox killed")
            except Exception as exc:
                logger.warning("Error killing sandbox: %s", exc)
