"""
EvalAgent — deterministic pre-execution gate.

Catches syntax errors and hallucinated imports before paying for an execution
round-trip. No LLM call. The executor is the ground truth for whether a test
works.

Routes:
  - REWRITE          — syntax error
  - RECONTEXTUALIZE  — unknown imports / undefined names
  - EXECUTE          — clean; defer semantic judgment to the executor
"""
from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from coverage_agent.credentials import Credentials
from coverage_agent.contracts import ContextPayload, CoverageGap, DraftTest, EvalResult

logger = logging.getLogger(__name__)

_STDLIB_MODULES = set(sys.stdlib_module_names)


def _check_syntax(test_code: str) -> bool:
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False


def _ruff_lint(test_code: str) -> tuple[bool, list[str]]:
    """Runs ruff on test_code. Returns (syntax_valid, undefined_names).

    F821 catches hallucinated identifiers. Falls back to ast.parse if ruff
    is not on PATH.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            tmp = f.name
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", "--select=F821", tmp],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(tmp)
        diagnostics = json.loads(result.stdout or "[]")
        syntax_valid = not any(d.get("code") == "invalid-syntax" for d in diagnostics)
        undefined_names = list({
            d["message"].split("`")[1]
            for d in diagnostics
            if d.get("code") == "F821" and "`" in d.get("message", "")
        })
        return syntax_valid, undefined_names
    except FileNotFoundError:
        return _check_syntax(test_code), []
    except Exception as exc:
        logger.debug("ruff lint error (%s) — falling back to ast.parse", exc)
        return _check_syntax(test_code), []


def _extract_imports(test_code: str) -> list[str]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module.split(".")[0])
    return modules


def _find_unknown_imports(
    test_code: str, context: ContextPayload, gap: CoverageGap
) -> list[str]:
    imported = _extract_imports(test_code)

    _SRC_DIRS = {"src", "lib", "python", "source", "packages", "pkg"}
    _parts = Path(gap.file_path).parts if gap.file_path else ()
    target_pkg = next(
        (p.replace(".py", "") for p in _parts if p not in _SRC_DIRS and not p.startswith(".")),
        _parts[0].replace(".py", "") if _parts else "",
    )
    dep_module_roots = {k.split(".")[0] for k in context.dependencies_code}

    known = _STDLIB_MODULES | {"pytest", "unittest", "mock"} | dep_module_roots
    if target_pkg:
        known.add(target_pkg)
    known |= set(_extract_imports(context.primary_code))

    unknown = []
    for mod in imported:
        if mod not in known:
            unknown.append(mod)
    return unknown


_NEUTRAL_ASSERTION_SCORE = 3


class EvalAgent:
    """Deterministic pre-execution gate."""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        draft: DraftTest,
        context: ContextPayload,
        gap: CoverageGap,
    ) -> EvalResult:
        syntax_valid, ruff_undefined = _ruff_lint(draft.test_code)
        if not syntax_valid:
            logger.info("EvalAgent: syntax invalid for %s — routing REWRITE", gap.gap_id)
            return EvalResult(
                syntax_valid=False,
                unknown_imports=[],
                mock_complete=False,
                assertion_score=1,
                critique="Test code has syntax errors. Fix all syntax errors before proceeding.",
                route="REWRITE",
            )

        unknown_imports = _find_unknown_imports(draft.test_code, context, gap)
        for name in ruff_undefined:
            if name not in unknown_imports:
                unknown_imports.append(name)

        if unknown_imports:
            critique = (
                f"Unknown imports / undefined names detected: {unknown_imports}. "
                "These do not appear in the context payload or stdlib. "
                "Either remove them or request more context."
            )
            logger.info(
                "EvalAgent: gap=%s unknown_imports=%s → RECONTEXTUALIZE",
                gap.gap_id, unknown_imports,
            )
            return EvalResult(
                syntax_valid=True,
                unknown_imports=unknown_imports,
                mock_complete=True,
                assertion_score=_NEUTRAL_ASSERTION_SCORE,
                critique=critique,
                route="RECONTEXTUALIZE",
            )

        logger.info("EvalAgent: gap=%s syntax=OK imports=OK → EXECUTE", gap.gap_id)

        return EvalResult(
            syntax_valid=True,
            unknown_imports=[],
            mock_complete=True,
            assertion_score=_NEUTRAL_ASSERTION_SCORE,
            critique="",
            route="EXECUTE",
        )
