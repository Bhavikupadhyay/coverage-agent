"""
ReAct tool implementations for the TestWriter agent.

These tools let the LLM probe the repo and execute draft tests before the
outer Executor runs its deterministic 3-run flakiness gate.

run_candidate is the INNER loop execution (agent self-check).
ExecutionRunner is the OUTER deterministic gate (3 runs, arcs verified).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LiteLLM function-calling schema (tools spec)
# ---------------------------------------------------------------------------

TOOLS_SPEC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "Read lines from a source file in the repo. "
                "Use to inspect code beyond the jedi-seeded context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path"},
                    "start": {"type": "integer", "description": "First line (1-indexed)"},
                    "end": {"type": "integer", "description": "Last line (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Return the source of the first definition matching 'name' (jedi-backed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function or class name to find"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_usages",
            "description": "Return a list of call sites for 'name' in the repo (jedi-backed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol to find usages of"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_candidate",
            "description": (
                "Execute a draft test in the caller's environment. "
                "Returns pass/fail, stderr, and whether target branches were hit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_code": {"type": "string", "description": "Complete pytest source code"},
                },
                "required": ["test_code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_source(path: str, start: int = 1, end: int | None = None, repo_root: str = ".") -> str:
    """Read lines [start, end] from a repo-relative file path."""
    try:
        full_path = Path(repo_root) / path
        lines = full_path.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        s = max(1, start) - 1
        e = min(total, end) if end is not None else total
        selected = lines[s:e]
        return "\n".join(f"{s + 1 + i}: {ln}" for i, ln in enumerate(selected))
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def find_symbol(name: str, repo_root: str = ".") -> str:
    """Return source of the first jedi definition matching name."""
    try:
        import jedi
        scripts = []
        for py_file in Path(repo_root).rglob("*.py"):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                script = jedi.Script(source=source, path=str(py_file), project=jedi.Project(repo_root))
                for name_obj in script.get_names(all_scopes=True, definitions=True, references=False):
                    if name_obj.name == name:
                        scripts.append((py_file, name_obj))
            except Exception:
                continue

        if not scripts:
            return f"No definition found for '{name}'."

        py_file, name_obj = scripts[0]
        lines = py_file.read_text(encoding="utf-8").splitlines()
        start = (name_obj.line or 1) - 1
        # Grab up to 40 lines of context.
        snippet = "\n".join(lines[start:start + 40])
        rel = py_file.relative_to(repo_root)
        return f"# {rel}:{name_obj.line}\n{snippet}"

    except Exception as exc:
        return f"Error finding symbol '{name}': {exc}"


def find_usages(name: str, repo_root: str = ".") -> str:
    """Return call sites for name across the repo."""
    try:
        import jedi
        usages: list[str] = []
        for py_file in Path(repo_root).rglob("*.py"):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                for i, line in enumerate(source.splitlines(), 1):
                    if name in line:
                        rel = py_file.relative_to(repo_root)
                        usages.append(f"{rel}:{i}: {line.strip()}")
            except Exception:
                continue

        if not usages:
            return f"No usages found for '{name}'."
        return "\n".join(usages[:30])  # cap at 30 results
    except Exception as exc:
        return f"Error finding usages of '{name}': {exc}"


def run_candidate(
    test_code: str,
    gap_from_line: int = 0,
    gap_to_line: int = 0,
    target_file: str = "",
    repo_root: str = ".",
) -> dict[str, Any]:
    """Execute a draft test in the caller's environment.

    Writes the test to a temp file, runs pytest, and if pytest passes reads
    the coverage arcs to check whether the target branch was hit.

    Returns:
        {
            "passed": bool,
            "stderr": str,
            "targets_hit": int,
            "targets_total": int,
        }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_test = Path(tmpdir) / "test_candidate.py"
        tmp_test.write_text(test_code, encoding="utf-8")
        cov_file = Path(tmpdir) / ".coverage_candidate"
        junit_xml = Path(tmpdir) / "junit.xml"

        # Run pytest (no coverage on the inner run to keep it fast).
        pytest_result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(tmp_test),
                "--tb=short", "-q",
                f"--junit-xml={junit_xml}",
            ],
            capture_output=True, text=True, cwd=repo_root,
        )
        passed = pytest_result.returncode == 0
        stderr = pytest_result.stderr[-2000:] if pytest_result.stderr else ""
        if pytest_result.stdout and not passed:
            stderr = (pytest_result.stdout[-1000:] + "\n" + stderr).strip()

        targets_hit = 0
        targets_total = 1 if (gap_from_line and gap_to_line) else 0

        if passed and target_file and gap_from_line and gap_to_line:
            # Re-run with coverage to check arc hit.
            cov_result = subprocess.run(
                [
                    sys.executable, "-m", "coverage", "run",
                    "--branch", f"--data-file={cov_file}",
                    "-m", "pytest", str(tmp_test), "-q", "--tb=no",
                ],
                capture_output=True, text=True, cwd=repo_root,
            )
            if cov_result.returncode == 0:
                try:
                    import coverage as coverage_module
                    cov = coverage_module.Coverage(data_file=str(cov_file))
                    cov.load()
                    data = cov.get_data()
                    abs_target = str(Path(repo_root) / target_file)
                    arcs = data.arcs(abs_target) or []
                    if (gap_from_line, gap_to_line) in arcs:
                        targets_hit = 1
                except Exception as exc:
                    logger.debug("run_candidate arc check failed: %s", exc)

        return {
            "passed": passed,
            "stderr": stderr,
            "targets_hit": targets_hit,
            "targets_total": targets_total,
        }


# ---------------------------------------------------------------------------
# Dispatcher — called from the writer's ReAct loop
# ---------------------------------------------------------------------------

def dispatch(tool_name: str, tool_input: dict, repo_root: str = ".", **gap_kwargs) -> str:
    """Routes a tool call from the LLM to its implementation.

    Returns the result as a string (tool observation for the next LLM turn).
    """
    try:
        if tool_name == "read_source":
            return read_source(
                path=tool_input["path"],
                start=tool_input.get("start", 1),
                end=tool_input.get("end"),
                repo_root=repo_root,
            )
        elif tool_name == "find_symbol":
            return find_symbol(name=tool_input["name"], repo_root=repo_root)
        elif tool_name == "find_usages":
            return find_usages(name=tool_input["name"], repo_root=repo_root)
        elif tool_name == "run_candidate":
            result = run_candidate(
                test_code=tool_input["test_code"],
                repo_root=repo_root,
                **gap_kwargs,
            )
            return json.dumps(result)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as exc:
        logger.warning("Tool '%s' raised: %s", tool_name, exc)
        return f"Tool error: {exc}"
