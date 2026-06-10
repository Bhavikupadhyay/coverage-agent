"""
Diff-gap algorithm: maps changed lines from a git diff to coverage gaps.

Produces CoverageGap objects with origin="diff" so the engine knows these
came from the changed set, not a full-repo scan.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from coverage_agent.contracts import BranchGap, CoverageGap
from coverage_agent.gaps.coverage_data import (
    _find_containing_symbol,
    _get_surrounding_lines,
    _is_trivial_gap,
)

logger = logging.getLogger(__name__)

_DIFF_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_TEST_PATH_PARTS = {"tests", "test", "spec", "specs"}


def _is_test_file(path: str) -> bool:
    parts = set(Path(path).parts)
    return bool(parts & _TEST_PATH_PARTS) or Path(path).name.startswith("test_")


def _merge_base(base_ref: str, repo_root: str) -> str:
    """Returns the merge-base commit SHA between base_ref and HEAD."""
    result = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git merge-base failed: {result.stderr.strip()}\n"
            f"Make sure '{base_ref}' exists as a local or remote ref."
        )
    return result.stdout.strip()


def _changed_lines(base_commit: str, repo_root: str, exclude: Sequence[str] = ()) -> dict[str, set[int]]:
    """Returns {repo_relative_path: {changed_line_numbers}} for .py files.

    Uses git diff -U0 so only changed lines appear (no context). Excludes
    test files, pure renames, and pure deletions.
    """
    result = subprocess.run(
        ["git", "diff", "-U0", "--diff-filter=ACMR", f"{base_commit}...HEAD", "--", "*.py"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")

    changed: dict[str, set[int]] = {}
    current_file: Optional[str] = None

    for line in result.stdout.splitlines():
        m = _DIFF_HEADER_RE.match(line)
        if m:
            path = m.group(1)
            if _is_test_file(path) or _matches_any(path, exclude):
                current_file = None
            else:
                current_file = path
                changed.setdefault(current_file, set())
            continue

        if current_file is None:
            continue

        hm = _HUNK_RE.match(line)
        if hm:
            start = int(hm.group(1))
            count = int(hm.group(2)) if hm.group(2) is not None else 1
            if count == 0:
                continue
            changed[current_file].update(range(start, start + count))

    return changed


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    import fnmatch
    fp = path.replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(fp, pat) or fnmatch.fnmatch(Path(fp).name, pat):
            return True
    return False


def _function_gaps_for_new_file(file_path: str, repo_root: str) -> list[CoverageGap]:
    """Produces one CoverageGap per public function/method in a new, uncovered file."""
    gaps: list[CoverageGap] = []
    try:
        source = (Path(repo_root) / file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as exc:
        logger.debug("AST parse failed for %s: %s", file_path, exc)
        return gaps

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        start = node.lineno
        end = node.end_lineno or start
        gap_id = f"{file_path}:{start}->{end}"
        gaps.append(CoverageGap(
            file_path=file_path,
            target_symbol=node.name,
            branch=BranchGap(from_line=start, to_line=end),
            surrounding_lines=list(range(start, end + 1)),
            kind="function",
            origin="diff",
            gap_id=gap_id,
        ))

    return gaps


def compute_diff_gaps(
    coverage_data: dict,
    repo_root: str = ".",
    base_ref: str = "",
    exclude: Sequence[str] = (),
) -> list[CoverageGap]:
    """Main entry point: maps git diff to coverage gaps.

    Args:
        coverage_data: Normalized coverage dict from load_coverage_file().
        repo_root: Absolute or relative path to the repo checkout.
        base_ref: Git ref to diff against. Defaults to merge-base with
                  origin/main or origin/master (tried in that order).
        exclude: Glob patterns for files to skip.

    Returns:
        List of CoverageGap with origin="diff".
    """
    if not base_ref:
        base_ref = _detect_base_ref(repo_root)

    try:
        base_commit = _merge_base(base_ref, repo_root)
    except RuntimeError as exc:
        logger.error("diff_gaps: %s", exc)
        return []

    try:
        changed = _changed_lines(base_commit, repo_root, exclude)
    except RuntimeError as exc:
        logger.error("diff_gaps: %s", exc)
        return []

    covered_files = set(coverage_data.get("files", {}).keys())
    gaps: list[CoverageGap] = []

    for file_path, changed_line_set in changed.items():
        if not changed_line_set:
            continue

        if file_path not in covered_files:
            # New file entirely absent from coverage — produce function gaps via AST.
            new_gaps = _function_gaps_for_new_file(file_path, repo_root)
            gaps.extend(new_gaps)
            logger.debug("diff_gaps: new file %s → %d function gaps", file_path, len(new_gaps))
            continue

        file_data = coverage_data["files"][file_path]
        missing_branches: list = file_data.get("missing_branches", [])

        for branch in missing_branches:
            if len(branch) != 2:
                continue
            from_line, to_line = int(branch[0]), int(branch[1])
            # Include this gap if the branch originates in changed code.
            if from_line not in changed_line_set and to_line not in changed_line_set:
                continue

            containing_symbol = _find_containing_symbol(file_path, from_line, repo_root)
            if _is_trivial_gap(file_path, from_line, containing_symbol, repo_root):
                continue

            gap_id = f"{file_path}:{from_line}->{to_line}"
            surrounding = _get_surrounding_lines(file_path, from_line, to_line, repo_root)
            gaps.append(CoverageGap(
                file_path=file_path,
                target_symbol=containing_symbol,
                branch=BranchGap(from_line=from_line, to_line=to_line),
                surrounding_lines=surrounding,
                kind="branch",
                origin="diff",
                gap_id=gap_id,
            ))

    logger.info("diff_gaps: %d gaps from %d changed files", len(gaps), len(changed))
    return gaps


def _detect_base_ref(repo_root: str) -> str:
    """Tries origin/main, then origin/master, then falls back to HEAD~1."""
    for ref in ("origin/main", "origin/master"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            capture_output=True, text=True, cwd=repo_root,
        )
        if result.returncode == 0:
            return ref
    logger.warning("diff_gaps: no origin/main or origin/master found — diffing against HEAD~1")
    return "HEAD~1"
