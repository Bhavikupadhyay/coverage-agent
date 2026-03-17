import ast
import logging
from pathlib import Path
from typing import Optional

from coverage_agent.contracts.schemas import BranchGap, CoverageGap

logger = logging.getLogger(__name__)

# Trivial symbols that are not worth targeting
_TRIVIAL_SYMBOLS = {"__init__", "__repr__", "__str__", "__eq__", "__hash__"}


def _is_trivial_line(source_line: str) -> bool:
    """Returns True for lines that represent no meaningful logic."""
    stripped = source_line.strip()
    return stripped in ("pass", "return", "return None", "...")


def _get_surrounding_lines(file_path: str, from_line: int, to_line: int) -> list[int]:
    """Returns the line numbers of the enclosing function body."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            func_start = node.lineno
            func_end = node.end_lineno or func_start
            if func_start <= from_line <= func_end:
                return list(range(func_start, func_end + 1))
    except Exception:
        pass

    # Fallback: return a window around the branch
    start = max(1, from_line - 5)
    end = to_line + 5
    return list(range(start, end + 1))


def _find_containing_symbol(file_path: str, line: int) -> str:
    """Returns the name of the function/method containing the given line."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        best: Optional[tuple[int, str]] = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            func_start = node.lineno
            func_end = node.end_lineno or func_start
            if func_start <= line <= func_end:
                if best is None or func_start > best[0]:
                    best = (func_start, node.name)

        if best:
            return best[1]
    except Exception:
        pass
    return "unknown"


def _is_trivial_gap(file_path: str, from_line: int, containing_symbol: str) -> bool:
    """Filters out gaps that aren't worth testing."""
    if containing_symbol in _TRIVIAL_SYMBOLS:
        return True

    try:
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        if 0 < from_line <= len(lines):
            if _is_trivial_line(lines[from_line - 1]):
                return True
    except Exception:
        pass

    return False


def parse_coverage(coverage_json: dict) -> list[CoverageGap]:
    """
    Parses a coverage.py --branch --json report into a list of CoverageGap objects.

    Each missing branch becomes one CoverageGap. Trivial gaps (inside __init__,
    pass statements, bare returns) are filtered out.

    Gap Prioritizer is responsible for filling in priority_score and refining
    target_symbol — both are set to placeholder values here.
    """
    gaps: list[CoverageGap] = []
    files: dict = coverage_json.get("files", {})

    for file_path, file_data in files.items():
        missing_branches: list = file_data.get("missing_branches", [])

        for branch in missing_branches:
            if len(branch) != 2:
                logger.warning("Unexpected branch format in %s: %r", file_path, branch)
                continue

            from_line, to_line = int(branch[0]), int(branch[1])
            containing_symbol = _find_containing_symbol(file_path, from_line)

            if _is_trivial_gap(file_path, from_line, containing_symbol):
                logger.debug("Skipping trivial gap %s:%d->%d", file_path, from_line, to_line)
                continue

            gap_id = f"{file_path}:{from_line}->{to_line}"
            surrounding = _get_surrounding_lines(file_path, from_line, to_line)

            gap = CoverageGap(
                file_path=file_path,
                target_symbol=containing_symbol,
                branch=BranchGap(from_line=from_line, to_line=to_line),
                surrounding_lines=surrounding,
                priority_score=0.0,
                gap_id=gap_id,
            )
            gaps.append(gap)

    logger.info("Parsed %d coverage gaps from %d files", len(gaps), len(files))
    return gaps
