"""
Branch-condition extractor — the single biggest TestWriter quality lever.

Coverage.py reports uncovered branches as `(from_line, to_line)`. `from_line`
is the line where the branch decision happens (an `if`, `while`, `for`, or
`try` statement). The LLM has historically had to *infer* what condition
controls that branch by reading the surrounding source — and gets it wrong.
The benchmark on `psf/requests.check_compatibility` (v3, 0/3 commits) is a
textbook case: TestWriter mocked `warnings.warn` and asserted `assert_called`
without understanding that `warn` only fires when `version_str < min_version`.

This module extracts the condition text directly via AST and hands it to
TestWriter as a `branch_condition_hint`. With the hint, TestWriter can pick
inputs that actually exercise the target branch instead of guessing.

Robustness:
- Returns None on any failure (syntax error, unmatched line, etc.). The
  pipeline runs fine without the hint — it's a quality boost, not a hard
  dependency.
- Two-pass match: first an exact `lineno == from_line` match (handles the
  common case where coverage reports the conditional's own line), then a
  containing-block fallback (handles multi-line conditionals where the
  reported line falls inside the branch statement).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional


def _condition_text(node: ast.AST) -> Optional[str]:
    """Returns a human-readable description of what triggers this branch."""
    if isinstance(node, (ast.If, ast.While)):
        try:
            return ast.unparse(node.test)
        except Exception:
            return None
    if isinstance(node, ast.For):
        try:
            return f"iteration over `{ast.unparse(node.iter)}`"
        except Exception:
            return None
    if isinstance(node, ast.Try):
        # The branch fires when an exception of the matching type is raised
        # inside the try-block. Without static analysis of every callee that's
        # all we can say cheaply.
        return "any exception raised inside the try-block"
    return None


def extract_branch_condition_from_source(source: str, from_line: int) -> Optional[str]:
    """Pure-function version: takes source text, returns the branch condition.

    Useful when the caller already has the source in memory (sandbox path
    runs this inside a script that already read the file).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    # Pass 1 — exact line match. Coverage typically reports the conditional's
    # own line, so this catches the common case.
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) == from_line:
            cond = _condition_text(node)
            if cond is not None:
                return cond

    # Pass 2 — containing block. Multi-line conditions, or branches reported
    # on inner expressions, fall through pass 1. The smallest enclosing branch
    # statement is the one we want.
    smallest: tuple[int, Optional[ast.AST]] = (10**9, None)
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


def extract_branch_condition(file_path: str, from_line: int) -> Optional[str]:
    """Reads the file and extracts the branch condition for the given line."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return None
    return extract_branch_condition_from_source(source, from_line)
