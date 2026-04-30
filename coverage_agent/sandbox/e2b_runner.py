import json
import logging
from pathlib import Path

from coverage_agent.config import is_dry_run
from coverage_agent.contracts.schemas import CoverageGap, ContextPayload, ExecutionResult

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Embedded scripts — run inside the E2B sandbox where the repo lives.
# No local filesystem access needed.
# ---------------------------------------------------------------------------

_COVERAGE_PARSER_SCRIPT = r"""
import ast, json
from pathlib import Path

coverage_json = json.loads(Path("/tmp/coverage_input.json").read_text())
repo_root = "/home/user/repo"

TRIVIAL_SYMBOLS = {"__init__", "__repr__", "__str__", "__eq__", "__hash__"}

def is_trivial_line(line):
    s = line.strip()
    return s in ("pass", "return", "return None", "...")

def get_surrounding_lines(source, from_line, to_line):
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end = node.end_lineno or node.lineno
            if node.lineno <= from_line <= end:
                return list(range(node.lineno, end + 1))
    except Exception:
        pass
    return list(range(max(1, from_line - 5), to_line + 6))

def find_containing_symbol(source, line):
    try:
        tree = ast.parse(source)
        best = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end = node.end_lineno or node.lineno
            if node.lineno <= line <= end:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.name)
        if best:
            return best[1]
    except Exception:
        pass
    return "unknown"

gaps = []
for file_path, file_data in coverage_json.get("files", {}).items():
    try:
        source = (Path(repo_root) / file_path).read_text(encoding="utf-8")
        lines = source.splitlines()
    except Exception:
        continue
    for branch in file_data.get("missing_branches", []):
        if len(branch) != 2:
            continue
        from_line, to_line = int(branch[0]), int(branch[1])
        symbol = find_containing_symbol(source, from_line)
        if symbol in TRIVIAL_SYMBOLS:
            continue
        if 0 < from_line <= len(lines) and is_trivial_line(lines[from_line - 1]):
            continue
        surrounding = get_surrounding_lines(source, from_line, to_line)
        gaps.append({
            "file_path": file_path,
            "target_symbol": symbol,
            "branch": {"from_line": from_line, "to_line": to_line},
            "surrounding_lines": surrounding,
            "priority_score": 0.0,
            "gap_id": f"{file_path}:{from_line}->{to_line}",
        })

print(json.dumps(gaps))
"""

_JEDI_CONTEXT_SCRIPT = r"""
import ast, json
from pathlib import Path

try:
    import jedi
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "jedi", "-q"], check=True)
    import jedi

MAX_TOKENS = 15000

def count_tokens(text):
    return len(text) // 4

def extract_function_source(resolved_path, target_symbol):
    try:
        source = Path(resolved_path).read_text(encoding="utf-8")
        script = jedi.Script(source, path=resolved_path)
        names = script.get_names(all_scopes=True, definitions=True)
        for name in names:
            if name.name == target_symbol and name.type in ("function", "class"):
                start_line = name.line
                lines = source.splitlines()
                base_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                func_lines = []
                for i, line in enumerate(lines[start_line - 1:], start=start_line):
                    if i > start_line and line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                        break
                    func_lines.append(line)
                return "\n".join(func_lines)
    except Exception:
        pass
    return None

def extract_callees(resolved_path, target_symbol):
    deps = {}
    try:
        source = Path(resolved_path).read_text(encoding="utf-8")
        script = jedi.Script(source, path=resolved_path)
        names = script.get_names(all_scopes=True, references=True)
        for name in names:
            if name.name == target_symbol:
                continue
            try:
                definitions = name.goto()
                for defn in definitions:
                    if defn.name and defn.name not in deps:
                        sig = defn.get_signatures()
                        deps[defn.name] = str(sig[0]) if sig else f"# {defn.type}: {defn.name}"
            except Exception:
                continue
    except Exception:
        pass
    return deps

args = json.loads(Path("/tmp/jedi_input.json").read_text())
file_path = args["file_path"]
target_symbol = args["target_symbol"]
depth = args["depth"]
repo_root = "/home/user/repo"
resolved = str(Path(repo_root) / file_path)

fallback_used = False
primary_code = extract_function_source(resolved, target_symbol)
if primary_code is None:
    try:
        primary_code = Path(resolved).read_text(encoding="utf-8")
    except Exception:
        primary_code = f"# Could not read {resolved}"
    fallback_used = True

dependencies = {}
depth_used = 0

if depth >= 1 and not fallback_used:
    tokens = count_tokens(primary_code)
    for name, source in extract_callees(resolved, target_symbol).items():
        t = count_tokens(source)
        if tokens + t > MAX_TOKENS:
            break
        dependencies[name] = source
        tokens += t
    depth_used = 1

if depth >= 2 and not fallback_used and dependencies:
    depth2 = {}
    for callee in list(dependencies):
        for name, source in extract_callees(resolved, callee).items():
            if name in dependencies or name in depth2:
                continue
            t = count_tokens(source)
            if tokens + t > MAX_TOKENS:
                break
            depth2[name] = source
            tokens += t
    if depth2:
        dependencies.update(depth2)
        depth_used = 2

tokens_used = count_tokens(primary_code) + sum(count_tokens(v) for v in dependencies.values())

print(json.dumps({
    "primary_code": primary_code,
    "dependencies_code": dependencies,
    "graph_depth_used": depth_used,
    "tokens_used": tokens_used,
    "fallback_used": fallback_used,
}))
"""


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

        if is_dry_run():
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
        if is_dry_run() or self._sandbox is not None:
            return
        if self._paused_id is None:
            raise RuntimeError("Sandbox is not active and has no paused ID to resume from")
        self.resume(self._paused_id)
        self._paused_id = None

    def install_dependencies(self) -> None:
        """Runs pip install inside the sandbox. Called once per repo."""
        if is_dry_run():
            logger.info("[DRY_RUN] install_dependencies — skipping")
            return

        self._ensure_active()
        result = self._sandbox.commands.run(
            "pip install -e '.[dev]' pytest pytest-cov coverage jedi -q",
            cwd="/home/user/repo",
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
        if is_dry_run():
            logger.info("[DRY_RUN] run_coverage_baseline — returning fixture")
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))

        self._ensure_active()
        # Allow pytest to fail (exit 1 = some tests failed) — still generate coverage JSON.
        # Exit code 2+ means pytest itself couldn't run (collection error, crash).
        self._sandbox.commands.run(
            "coverage run --branch -m pytest -q --ignore=tests/test_utils.py || true",
            cwd="/home/user/repo",
            timeout=300,
        )
        json_result = self._sandbox.commands.run(
            "coverage json -o coverage.json",
            cwd="/home/user/repo",
            timeout=60,
        )
        if json_result.exit_code != 0:
            raise RuntimeError(
                f"coverage json failed (exit {json_result.exit_code}):\n{json_result.stderr}"
            )

        coverage_raw = self._sandbox.files.read("/home/user/repo/coverage.json")
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
        if is_dry_run():
            logger.info("[DRY_RUN] run_test — returning fixture ExecutionResult for gap %s", gap_id)
            return ExecutionResult(
                execution_success=True,
                target_branch_hit=True,
                coverage_delta=0.4,
                stderr_trace="",
                flaky=False,
            )

        self._ensure_active()
        test_path = "/home/user/repo/tests/test_coverageagent_auto.py"
        try:
            self._sandbox.files.write(test_path, test_code)

            # pytest exits 1 on test failures — catch it and record as execution_success=False
            execution_success = True
            stderr_trace = ""
            try:
                result = self._sandbox.commands.run(
                    f"coverage run --branch --append -m pytest {test_path} -q",
                    cwd="/home/user/repo",
                    timeout=120,
                )
                stderr_trace = result.stderr or ""
            except Exception as exc:
                execution_success = False
                stderr_trace = str(exc)

            # Always generate coverage JSON so we can measure delta regardless
            try:
                self._sandbox.commands.run(
                    "coverage json -o coverage_after.json",
                    cwd="/home/user/repo",
                    timeout=60,
                )
            except Exception:
                pass

            coverage_delta = 0.0
            target_branch_hit = False

            if execution_success:
                try:
                    after_raw = self._sandbox.files.read("/home/user/repo/coverage_after.json")
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
        """Uploads a local repo directory to /home/user/repo in the sandbox. Skips .git, __pycache__, *.pyc."""
        if is_dry_run():
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
            dest = f"/home/user/repo/{rel}"
            try:
                self._sandbox.files.write(dest, src.read_bytes())
                uploaded += 1
            except Exception as exc:
                logger.warning("Could not upload %s: %s", rel, exc)
        logger.info("Uploaded %d files to sandbox /home/user/repo", uploaded)

    def setup_repo(self, repo_url_or_path: str) -> None:
        """Clones a repo URL into /home/user/repo, or uploads a local path to /home/user/repo."""
        if is_dry_run():
            logger.info("[DRY_RUN] setup_repo — skipping")
            return
        self._ensure_active()
        if repo_url_or_path.startswith(("http://", "https://", "git@")):
            result = self._sandbox.commands.run(
                f"git clone --depth 1 {repo_url_or_path} /home/user/repo",
                timeout=120,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"git clone failed:\n{result.stderr}")
            logger.info("Cloned %s into sandbox /home/user/repo", repo_url_or_path)
        else:
            self.upload_repo(repo_url_or_path)

    def pause(self) -> str:
        """
        Pauses the sandbox to stop billing during LLM calls.
        Returns the sandbox_id needed to resume. Sets internal sandbox to None.
        """
        if is_dry_run():
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
        if is_dry_run():
            logger.info("[DRY_RUN] resume — skipping")
            return

        from e2b import Sandbox
        self._sandbox = Sandbox.connect(sandbox_id)
        logger.info("E2B sandbox resumed (id=%s)", sandbox_id)

    def parse_gaps(self, coverage_json: dict) -> list[CoverageGap]:
        """Runs coverage gap parsing inside E2B. Returns CoverageGap list with priority_score=0."""
        if is_dry_run():
            logger.info("[DRY_RUN] parse_gaps — returning fixture gaps")
            from coverage_agent.context.coverage_parser import parse_coverage
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            fixture_json = json.loads(fixture_path.read_text(encoding="utf-8"))
            return parse_coverage(fixture_json, repo_root=".")

        self._ensure_active()
        self._sandbox.files.write("/tmp/coverage_input.json", json.dumps(coverage_json))
        self._sandbox.files.write("/tmp/parse_gaps.py", _COVERAGE_PARSER_SCRIPT)
        result = self._sandbox.commands.run(
            "python /tmp/parse_gaps.py",
            cwd="/home/user/repo",
            timeout=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"parse_gaps script failed (exit {result.exit_code}):\n{result.stderr}")
        gaps_data = json.loads(result.stdout)
        return [CoverageGap(**g) for g in gaps_data]

    def build_context(self, file_path: str, target_symbol: str, depth: int) -> dict:
        """Runs Jedi context building inside E2B. Returns a dict matching ContextPayload fields."""
        if is_dry_run():
            logger.info("[DRY_RUN] build_context — returning fixture context")
            fixture_path = _FIXTURES_DIR / "sample_context.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))

        self._ensure_active()
        args = {"file_path": file_path, "target_symbol": target_symbol, "depth": depth}
        self._sandbox.files.write("/tmp/jedi_input.json", json.dumps(args))
        self._sandbox.files.write("/tmp/jedi_context.py", _JEDI_CONTEXT_SCRIPT)
        result = self._sandbox.commands.run(
            "python /tmp/jedi_context.py",
            cwd="/home/user/repo",
            timeout=60,
        )
        if result.exit_code != 0:
            logger.warning(
                "build_context script failed for %s:%s (exit %d) — returning empty context",
                file_path, target_symbol, result.exit_code,
            )
            return {
                "primary_code": f"# Could not read {file_path}:{target_symbol}",
                "dependencies_code": {},
                "graph_depth_used": 0,
                "tokens_used": 0,
                "fallback_used": True,
            }
        return json.loads(result.stdout)

    def close(self) -> None:
        """Kills the E2B sandbox VM. Called by Orchestrator after all gaps are done."""
        if is_dry_run():
            logger.info("[DRY_RUN] close — skipping")
            return

        if self._sandbox:
            try:
                self._sandbox.kill()
                logger.info("E2B sandbox killed")
            except Exception as exc:
                logger.warning("Error killing sandbox: %s", exc)
