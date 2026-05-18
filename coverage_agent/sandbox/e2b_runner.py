import json
import logging
import os
import re
from pathlib import Path

from coverage_agent.contracts.schemas import CoverageGap, ContextPayload, ExecutionResult

logger = logging.getLogger(__name__)

_COVERAGE_SRC_LAYOUT = frozenset({"src", "lib", "python", "source", "packages", "pkg"})


def coverage_source_cli_flag(file_path: str) -> str:
    """Returns a single ` --source=pkg` fragment for `coverage run`, or empty.

    Scopes measurement to the package under test so branch data for that file
    appears in coverage JSON (avoids 'module test was never imported' noise).
    """
    if not file_path:
        return ""
    parts = [p for p in Path(file_path).parts if p not in (".", "..")]
    if not parts:
        return ""
    root = parts[0].removesuffix(".py")
    if root in _COVERAGE_SRC_LAYOUT and len(parts) > 1:
        root = parts[1].removesuffix(".py")
    if not root:
        return ""
    return f" --source={root}"

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Matches a pytest summary tail line:
#   "12 passed, 1 failed, 2 skipped in 1.23s"
#   "12 passed in 0.5s"
#   "1 failed in 0.2s"
_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed")


def _parse_pytest_counts(output: str) -> tuple[int, int]:
    """Extracts (passed, failed) from pytest's terminal summary.

    Returns (0, 0) if the summary can't be parsed — caller should treat that
    as 'unknown' rather than 'definitely zero'.
    """
    passed_m = _PYTEST_PASSED_RE.search(output or "")
    failed_m = _PYTEST_FAILED_RE.search(output or "")
    passed = int(passed_m.group(1)) if passed_m else 0
    failed = int(failed_m.group(1)) if failed_m else 0
    return passed, failed

# ---------------------------------------------------------------------------
# Embedded scripts — run inside the E2B sandbox where the repo lives.
# No local filesystem access needed.
# ---------------------------------------------------------------------------

_COVERAGE_PARSER_SCRIPT = r"""
import ast, fnmatch, json
from pathlib import Path

coverage_json = json.loads(Path("/tmp/coverage_input.json").read_text())
repo_root = "/home/user/repo"

try:
    ignore_patterns = json.loads(Path("/tmp/ignore_patterns.json").read_text())
except Exception:
    ignore_patterns = []

# Also auto-discover .coverageagentignore in the repo root
_ignore_file = Path(repo_root) / ".coverageagentignore"
if _ignore_file.exists():
    for ln in _ignore_file.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            ignore_patterns.append(ln)

TRIVIAL_SYMBOLS = {"__init__", "__repr__", "__str__", "__eq__", "__hash__"}

def file_matches_patterns(fp, patterns):
    fp = fp.replace("\\", "/")
    parts = fp.split("/")
    for raw in patterns:
        is_dir = raw.endswith("/")
        pat = raw.rstrip("/")
        if "/" in pat:
            if fnmatch.fnmatch(fp, pat):
                return True
            if is_dir and fp.startswith(pat + "/"):
                return True
        else:
            if any(fnmatch.fnmatch(part, pat) for part in parts):
                return True
    return False

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
    if ignore_patterns and file_matches_patterns(file_path, ignore_patterns):
        continue
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

def _condition_text(node):
    if isinstance(node, (ast.If, ast.While)):
        try:
            return ast.unparse(node.test)
        except Exception:
            return None
    if isinstance(node, ast.For):
        try:
            return "iteration over `" + ast.unparse(node.iter) + "`"
        except Exception:
            return None
    if isinstance(node, ast.Try):
        return "any exception raised inside the try-block"
    return None

def extract_branch_condition(source, from_line):
    if from_line is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) == from_line:
            cond = _condition_text(node)
            if cond is not None:
                return cond
    smallest = (10**9, None)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None or end is None:
            continue
        if start <= from_line <= end:
            span = end - start
            if span < smallest[0]:
                smallest = (span, node)
    if smallest[1] is not None:
        return _condition_text(smallest[1])
    return None

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
from_line = args.get("from_line")
repo_root = "/home/user/repo"
resolved = str(Path(repo_root) / file_path)

fallback_used = False
primary_code = extract_function_source(resolved, target_symbol)
file_source = ""
try:
    file_source = Path(resolved).read_text(encoding="utf-8")
except Exception:
    pass
if primary_code is None:
    primary_code = file_source if file_source else f"# Could not read {resolved}"
    fallback_used = True

branch_condition_hint = extract_branch_condition(file_source, from_line) if file_source else None

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
    "branch_condition_hint": branch_condition_hint,
}))
"""


class E2BSandbox:
    """
    Manages the E2B sandbox lifecycle for a single repo benchmark run.

    The sandbox is created once per repo by the Orchestrator and reused
    across all gap iterations. All methods check OFFLINE_MODE before making
    any real E2B calls.

    Live mode uses the E2B SDK v2+ API:
      - Sandbox.create() to spin up
      - sandbox.commands.run(cmd, cwd=..., timeout=...) to execute shell commands
      - sandbox.files.write/read/remove for filesystem operations
      - sandbox.kill() to terminate
      - sandbox.pause() / Sandbox.connect(sandbox_id) for cost optimization
    """

    def __init__(
        self,
        repo_path: str,
        e2b_api_key: str = "",
        offline: bool = False,
        template_id: str = "",
    ) -> None:
        self.repo_path = repo_path
        self._sandbox = None
        self._paused_id: str | None = None
        self._offline = offline
        self._api_key = e2b_api_key
        self._template_id = template_id or os.environ.get("E2B_TEMPLATE_ID") or ""

        if self._offline:
            logger.info("[OFFLINE] E2BSandbox.__init__ — skipping real sandbox creation for %s", repo_path)
            return

        try:
            from e2b import Sandbox
            create_kwargs: dict = {}
            if self._api_key:
                create_kwargs["api_key"] = self._api_key
            if self._template_id:
                create_kwargs["template"] = self._template_id
            self._sandbox = Sandbox.create(**create_kwargs)
            logger.info("E2B sandbox created (id=%s) for %s", self._sandbox.sandbox_id, repo_path)
        except Exception as exc:
            logger.error("Failed to create E2B sandbox: %s", exc)
            raise

    def _ensure_active(self) -> None:
        """Resumes the sandbox if it was paused. No-op in offline mode or if already active."""
        if self._offline or self._sandbox is not None:
            return
        if self._paused_id is None:
            raise RuntimeError("Sandbox is not active and has no paused ID to resume from")
        self.resume(self._paused_id)
        self._paused_id = None

    def install_dependencies(self) -> None:
        """Runs pip install inside the sandbox. Called once per repo."""
        if self._offline:
            logger.info("[OFFLINE] install_dependencies — skipping")
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
        if self._offline:
            logger.info("[OFFLINE] run_coverage_baseline — returning fixture")
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
        # Repos with parallel-coverage configs (e.g. .coveragerc parallel=True) leave
        # one .coverage.HOST.PID.RAND file per worker. Subsequent `coverage` commands
        # refuse to read those without combining first. Belt-and-suspenders: try
        # combine, swallow the error if there's nothing to combine.
        self._sandbox.commands.run(
            "coverage combine 2>/dev/null || true",
            cwd="/home/user/repo",
            timeout=30,
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
        if self._offline:
            logger.info("[OFFLINE] run_test — returning fixture ExecutionResult for gap %s", gap_id)
            return ExecutionResult(
                execution_success=True,
                target_branch_hit=True,
                coverage_delta=0.4,
                stderr_trace="",
                flaky=False,
            )

        self._ensure_active()
        test_path = "/home/user/repo/tests/test_coverageagent_auto.py"

        # Write a coverage override file that forces parallel=false and branch=true,
        # ignoring any parallel=true in the target repo's .coveragerc or pyproject.toml.
        # This is the root fix for the "Can't append to data files in parallel mode" error.
        override_path = "/tmp/cov_agent_override.ini"
        self._sandbox.files.write(override_path, "[run]\nparallel = false\nbranch = true\n")
        rc_env = f"COVERAGE_RCFILE={override_path}"

        try:
            self._sandbox.files.write(test_path, test_code)

            test_coverage_env = "COVERAGE_FILE=.coverage_agent_per_test"
            src_flag = coverage_source_cli_flag(target_file)
            execution_success = False
            stderr_trace = ""
            try:
                result = self._sandbox.commands.run(
                    f"rm -f .coverage_agent_per_test* && "
                    f"{test_coverage_env} {rc_env} coverage run --branch{src_flag} -m pytest {test_path} -q",
                    cwd="/home/user/repo",
                    timeout=120,
                )
                stderr_trace = result.stderr or ""
                execution_success = getattr(result, "exit_code", 1) == 0
            except Exception as exc:
                execution_success = False
                stderr_trace = str(exc)

            # Generate JSON for the test-only data. parallel=false means no combine needed.
            try:
                self._sandbox.commands.run(
                    f"{test_coverage_env} {rc_env} coverage json -o coverage_after.json",
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

                    # `after` is the per-test coverage data (in isolation, our
                    # test ran alone). We compute delta as "new branches OUR
                    # test covers in the target file that the baseline missed",
                    # normalized over baseline missing branches in that file.
                    # This is a per-gap contribution measure, not a global
                    # repo-wide percentage — but it's what the UI cares about.
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
                            # rough delta: fraction of previously-missing branches
                            # in this file that our test now covers
                            denom = max(len(baseline_missing_branches), 1)
                            coverage_delta = round(
                                len(newly_covered_missing) / denom * 100.0, 2
                            )
                        elif target_branch_hit:
                            coverage_delta = 1.0  # at least the target branch landed
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

    def count_test_outcomes(self) -> tuple[int, int]:
        """
        Runs the full pytest suite (no coverage, fast) and returns (passed, failed) counts.

        Used by RegressionGuard once before any agent-written tests are added
        (baseline) and once after all committed tests are persisted (post). The
        delta of `failed` tells us whether the agent introduced a regression.
        """
        if self._offline:
            logger.info("[OFFLINE] count_test_outcomes — returning fixture (12, 0)")
            return 12, 0

        self._ensure_active()
        # `pytest -q --no-header` gives a tail line like "12 passed, 1 failed in 1.23s"
        # which is trivial to parse. `-p no:cacheprovider` avoids polluting .pytest_cache.
        result = self._sandbox.commands.run(
            "pytest -q --no-header -p no:cacheprovider --ignore=tests/test_utils.py || true",
            cwd="/home/user/repo",
            timeout=300,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return _parse_pytest_counts(output)

    def persist_test(self, test_code: str, filename: str) -> None:
        """Writes a test file into /home/user/repo/tests/ and leaves it there.

        Used by RegressionGuard so all committed tests are on disk before the
        final full-suite re-run. The orchestrator calls this in a batch.
        """
        if self._offline:
            logger.info("[OFFLINE] persist_test(%s) — skipping", filename)
            return
        self._ensure_active()
        # Ensure tests/ exists; harmless if already present.
        self._sandbox.commands.run("mkdir -p /home/user/repo/tests", timeout=10)
        path = f"/home/user/repo/tests/{filename}"
        self._sandbox.files.write(path, test_code)

    def upload_repo(self, local_path: str) -> None:
        """Uploads a local repo directory to /home/user/repo in the sandbox. Skips .git, __pycache__, *.pyc."""
        if self._offline:
            logger.info("[OFFLINE] upload_repo — skipping")
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
        if self._offline:
            logger.info("[OFFLINE] setup_repo — skipping")
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
        if self._offline:
            logger.info("[OFFLINE] pause — skipping")
            return "offline-sandbox-id"

        sandbox_id = self._sandbox.sandbox_id
        self._sandbox.pause()
        self._sandbox = None
        self._paused_id = sandbox_id
        logger.info("E2B sandbox paused (id=%s)", sandbox_id)
        return sandbox_id

    def resume(self, sandbox_id: str) -> None:
        """Resumes a previously paused sandbox by its ID."""
        if self._offline:
            logger.info("[OFFLINE] resume — skipping")
            return

        from e2b import Sandbox
        connect_kwargs: dict = {}
        if self._api_key:
            connect_kwargs["api_key"] = self._api_key
        self._sandbox = Sandbox.connect(sandbox_id, **connect_kwargs)
        logger.info("E2B sandbox resumed (id=%s)", sandbox_id)

    def parse_gaps(self, coverage_json: dict, ignore_patterns: list[str] | None = None) -> list[CoverageGap]:
        """Runs coverage gap parsing inside E2B. Returns CoverageGap list with priority_score=0."""
        if self._offline:
            logger.info("[OFFLINE] parse_gaps — returning fixture gaps")
            from coverage_agent.context.coverage_parser import parse_coverage
            fixture_path = _FIXTURES_DIR / "sample_coverage.json"
            fixture_json = json.loads(fixture_path.read_text(encoding="utf-8"))
            return parse_coverage(fixture_json, repo_root=".")

        self._ensure_active()

        # Pass ignore patterns into the sandbox script via a sidecar JSON file
        patterns = ignore_patterns or []
        self._sandbox.files.write("/tmp/coverage_input.json", json.dumps(coverage_json))
        self._sandbox.files.write("/tmp/ignore_patterns.json", json.dumps(patterns))
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

    def build_context(
        self,
        file_path: str,
        target_symbol: str,
        depth: int,
        from_line: int | None = None,
    ) -> dict:
        """Runs Jedi context building inside E2B. Returns a dict matching ContextPayload fields.

        `from_line` is the coverage-gap branch's from_line, used to extract a
        branch condition hint via AST. None means skip the hint.
        """
        if self._offline:
            logger.info("[OFFLINE] build_context — returning fixture context")
            fixture_path = _FIXTURES_DIR / "sample_context.json"
            return json.loads(fixture_path.read_text(encoding="utf-8"))

        self._ensure_active()
        args = {
            "file_path": file_path,
            "target_symbol": target_symbol,
            "depth": depth,
            "from_line": from_line,
        }
        self._sandbox.files.write("/tmp/jedi_input.json", json.dumps(args))
        self._sandbox.files.write("/tmp/jedi_context.py", _JEDI_CONTEXT_SCRIPT)
        result = self._sandbox.commands.run(
            "python /tmp/jedi_context.py",
            cwd="/home/user/repo",
            timeout=300,
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
                "branch_condition_hint": None,
            }
        return json.loads(result.stdout)

    def validate_python_repo(self) -> None:
        """Verifies the repo in the sandbox has Python files. Raises ValueError if not."""
        if self._offline:
            return
        self._ensure_active()
        result = self._sandbox.commands.run(
            "find /home/user/repo -name '*.py' | wc -l",
            timeout=15,
        )
        count = int(result.stdout.strip() or "0")
        if count == 0:
            raise ValueError(
                "Not a Python repository — no .py files found. "
                "CoverageAgent only supports Python projects."
            )

    def close(self) -> None:
        """Kills the E2B sandbox VM. Called by Orchestrator after all gaps are done."""
        if self._offline:
            logger.info("[OFFLINE] close — skipping")
            return

        if self._sandbox:
            try:
                self._sandbox.kill()
                logger.info("E2B sandbox killed")
            except Exception as exc:
                logger.warning("Error killing sandbox: %s", exc)
