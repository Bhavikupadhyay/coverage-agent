"""
EvalAgent — deterministic pre-execution gate.

The sandbox is the ground truth for "does this test work." EvalAgent's only
remaining job is to catch the cheap, obvious failures before we pay for a
sandbox round-trip:

  1. Syntax validity (ruff, falls back to ast.parse)
  2. Import plausibility (ast + context payload + ruff F821)

That's it. No LLM call. No assertion-quality scoring. No mock-completeness
LLM. Those were forms of judgment that the assertion-quality LLM consistently
got wrong on `psf/requests` benchmarks — they were the source of v3's
false-positive 5/5 scores on tests that crashed at runtime.

By design, this means well-formed-but-semantically-wrong tests will reach
the sandbox. That's intentional: the sandbox can tell us they crashed in a
way the LLM cannot, and the new execution_runner → test_writer retry edge
in pipeline.py uses that stderr as the next critique.

Strictness now lives on Credentials (`should_commit`, `max_retry_loops`) and
controls the commit gate after execution, not the LLM gate before it.
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
from coverage_agent.contracts.schemas import ContextPayload, CoverageGap, DraftTest, EvalResult

logger = logging.getLogger(__name__)

_STDLIB_MODULES = set(sys.stdlib_module_names)


def _check_syntax(test_code: str) -> bool:
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False


def _ruff_lint(test_code: str) -> tuple[bool, list[str]]:
    """
    Runs ruff on test_code via subprocess.
    Returns (syntax_valid, undefined_names).

    F821 = undefined name (catches hallucinated identifiers that slip past
    the import-plausibility check). Falls back to ast.parse if ruff isn't
    on PATH.
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
        # F821 messages use backticks: "Undefined name `foo`"
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
    """Returns top-level module names imported in the test code."""
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
    """
    Returns imports that are neither stdlib nor plausibly known given the context.

    Known modules include:
    - All stdlib top-level names
    - pytest / unittest / mock (universal test utilities)
    - The top-level package of the file under test (e.g. "requests" from "requests/auth.py")
    - Module roots extracted from dependency keys (e.g. "re" from "re.compile")
    - Top-level imports appearing in context.primary_code (the excerpt shown to the writer)
    """
    imported = _extract_imports(test_code)

    # Target package root: "requests/auth.py" → "requests"
    # Skip generic src-layout prefixes: "src/requests/auth.py" → "requests"
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
    # Imports already used in the excerpted target code are safe to use in tests
    # (e.g. `certifi` in requests/certs.py) even when Jedi depth=0 omitted them
    # from dependencies_code.
    known |= set(_extract_imports(context.primary_code))

    unknown = []
    for mod in imported:
        if mod not in known:
            unknown.append(mod)
    return unknown


# Constant filled into EvalResult.assertion_score so the schema (which still
# requires 1..=5) doesn't need to change. The number itself is meaningless
# now — the field is a vestige of the old LLM-gate design. The sandbox tells
# us what we actually care about.
_NEUTRAL_ASSERTION_SCORE = 3


class EvalAgent:
    """
    Deterministic pre-execution gate. Catches syntax errors and hallucinated
    imports before we pay for an E2B round-trip. Everything else routes to
    EXECUTE and lets the sandbox tell us the truth.

    Routes:
      - REWRITE        — syntax error (ast.parse / ruff `invalid-syntax`)
      - RECONTEXTUALIZE — unknown imports / undefined names that look like missing context
      - EXECUTE        — clean syntactically, defer semantic judgment to the sandbox

    In offline mode, route=EXECUTE with passing scores so the full pipeline
    can be exercised in tests without an LLM or sandbox.
    """

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        draft: DraftTest,
        context: ContextPayload,
        gap: CoverageGap,
    ) -> EvalResult:
        # --- 1. Syntax check (ruff, falls back to ast.parse) ---
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

        # --- 2. Import plausibility (deterministic) ---
        unknown_imports = _find_unknown_imports(draft.test_code, context, gap)
        # Merge ruff F821 undefined names not already caught by import analysis
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

        # --- 3. Everything else: defer to sandbox ---
        # The sandbox can tell us whether the test runs, whether assertions
        # pass, whether the target branch fires. An LLM cannot do any of
        # that reliably and was the source of v3's false-positive 5/5 scores.
        if self.creds.is_offline:
            logger.info("[OFFLINE] EvalAgent — routing EXECUTE for %s", gap.gap_id)
        else:
            logger.info("EvalAgent: gap=%s syntax=OK imports=OK → EXECUTE", gap.gap_id)

        return EvalResult(
            syntax_valid=True,
            unknown_imports=[],
            mock_complete=True,
            assertion_score=_NEUTRAL_ASSERTION_SCORE,
            critique="",
            route="EXECUTE",
        )
