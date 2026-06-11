"""
Coverage data loaders.

Supports three formats:
  - coverage.py JSON export (coverage json -o coverage.json)
  - .coverage binary data-file (via the coverage Python API)
  - Cobertura XML (coverage xml -o coverage.xml)

`load_coverage_file` auto-detects the format from the file extension and
returns a normalized dict in coverage.py JSON shape so the rest of the
pipeline has one data model.
"""
import ast
import fnmatch
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence

from coverage_agent.contracts import BranchGap, CoverageGap

logger = logging.getLogger(__name__)

_TRIVIAL_SYMBOLS = {"__init__", "__repr__", "__str__", "__eq__", "__hash__"}


# ---------------------------------------------------------------------------
# Coverage file loading
# ---------------------------------------------------------------------------

def load_coverage_file(path: str) -> dict:
    """Loads a coverage report and returns a normalized coverage.py JSON dict.

    Auto-detects format from extension:
      .json        → parse directly
      .coverage    → use coverage Python API
      .xml         → parse Cobertura XML
    """
    p = Path(path)
    if p.suffix == ".json":
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix == ".xml":
        return _load_cobertura_xml(path)
    else:
        # Treat as .coverage binary data-file
        return _load_coverage_datafile(path)


def _load_coverage_datafile(path: str) -> dict:
    """Loads a .coverage binary file by delegating to 'coverage json'.

    Converts the binary .coverage data file to the normalized JSON shape by
    running 'coverage json' in a subprocess. This correctly computes
    missing_branches via coverage.py's own analysis logic rather than
    attempting to reconstruct missing arcs from executed-arc data.
    """
    import json
    import subprocess
    import sys
    import tempfile

    p = Path(path)
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_json = tmp.name

        result = subprocess.run(
            [sys.executable, "-m", "coverage", "json",
             f"--data-file={path}", "-o", tmp_json],
            capture_output=True,
            text=True,
            cwd=str(p.parent),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"coverage json failed for {path}: {result.stderr.strip()}"
            )
        data = json.loads(Path(tmp_json).read_text(encoding="utf-8"))
        Path(tmp_json).unlink(missing_ok=True)
        return data
    except Exception as exc:
        logger.error("Failed to load .coverage data file %s: %s", path, exc)
        raise


def _load_cobertura_xml(path: str) -> dict:
    """Parses a Cobertura XML coverage report into the normalized dict shape."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        files: dict = {}

        for cls in root.iter("class"):
            filename = cls.get("filename", "")
            if not filename:
                continue
            missing_branches: list = []
            for line in cls.iter("line"):
                branch = line.get("branch", "false").lower() == "true"
                if not branch:
                    continue
                number = int(line.get("number", 0))
                conditions_covered = int(line.get("condition-coverage", "0%").split("%")[0])
                # If not fully covered, record a synthetic branch gap
                if conditions_covered < 100:
                    missing_branches.append([number, number + 1])
            files[filename] = {"missing_branches": missing_branches}

        line_rate = float(root.get("line-rate", "0"))
        return {"files": files, "totals": {"percent_covered": line_rate * 100}}
    except Exception as exc:
        logger.error("Failed to parse Cobertura XML %s: %s", path, exc)
        raise


# ---------------------------------------------------------------------------
# Ignore pattern engine (gitignore semantics)
# ---------------------------------------------------------------------------

def load_ignore_patterns(path: str) -> list[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _file_matches_patterns(file_path: str, patterns: Sequence[str]) -> bool:
    fp = file_path.replace("\\", "/")
    parts = fp.split("/")

    for raw in patterns:
        is_dir = raw.endswith("/")
        pat = raw.rstrip("/")

        if "/" in pat:
            if fnmatch.fnmatch(fp, pat):
                return True
            if is_dir and (fp.startswith(pat + "/") or fnmatch.fnmatch(fp, pat + "/*")):
                return True
        else:
            for part in parts:
                if fnmatch.fnmatch(part, pat):
                    return True

    return False


# ---------------------------------------------------------------------------
# Gap extraction helpers
# ---------------------------------------------------------------------------

def _is_trivial_line(source_line: str) -> bool:
    stripped = source_line.strip()
    return stripped in ("pass", "return", "return None", "...")


def _get_surrounding_lines(file_path: str, from_line: int, to_line: int, repo_root: str = ".") -> list[int]:
    try:
        source = (Path(repo_root) / file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            func_start = node.lineno
            func_end = node.end_lineno or func_start
            if func_start <= from_line <= func_end:
                return list(range(func_start, func_end + 1))
    except Exception:
        pass

    start = max(1, from_line - 5)
    end = to_line + 5
    return list(range(start, end + 1))


def _find_containing_symbol(file_path: str, line: int, repo_root: str = ".") -> str:
    try:
        source = (Path(repo_root) / file_path).read_text(encoding="utf-8")
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


def _is_trivial_gap(file_path: str, from_line: int, containing_symbol: str, repo_root: str = ".") -> bool:
    if containing_symbol in _TRIVIAL_SYMBOLS:
        return True

    try:
        lines = (Path(repo_root) / file_path).read_text(encoding="utf-8").splitlines()
        if 0 < from_line <= len(lines):
            if _is_trivial_line(lines[from_line - 1]):
                return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _normalize_file_path(raw_path: str, repo_root: str) -> Optional[str]:
    """Converts an absolute or relative path to a repo-relative path.

    Returns None if the path is outside the repo root (e.g. site-packages).
    """
    try:
        p = Path(raw_path)
        root = Path(repo_root).resolve()
        abs_p = p if p.is_absolute() else (root / p)
        abs_p = abs_p.resolve()
        return str(abs_p.relative_to(root))
    except ValueError:
        return None


def parse_coverage(
    coverage_json: dict,
    repo_root: str = ".",
    ignore_patterns: Sequence[str] = (),
) -> list[CoverageGap]:
    """Parses a normalized coverage dict into CoverageGap objects.

    Accepts the dict returned by load_coverage_file() or a raw
    coverage.py --branch --json export.

    Absolute paths (as stored by the coverage binary) are normalized to
    repo-relative paths. Files outside the repo root are skipped.
    """
    gaps: list[CoverageGap] = []
    files: dict = coverage_json.get("files", {})

    for raw_path, file_data in files.items():
        file_path = _normalize_file_path(raw_path, repo_root)
        if file_path is None:
            logger.debug("Skipping out-of-repo file: %s", raw_path)
            continue

        if ignore_patterns and _file_matches_patterns(file_path, ignore_patterns):
            logger.debug("Ignoring %s (matched ignore pattern)", file_path)
            continue

        missing_branches: list = file_data.get("missing_branches", [])

        for branch in missing_branches:
            if len(branch) != 2:
                logger.warning("Unexpected branch format in %s: %r", file_path, branch)
                continue

            from_line, to_line = int(branch[0]), int(branch[1])
            containing_symbol = _find_containing_symbol(file_path, from_line, repo_root)

            if _is_trivial_gap(file_path, from_line, containing_symbol, repo_root):
                logger.debug("Skipping trivial gap %s:%d->%d", file_path, from_line, to_line)
                continue

            gap_id = f"{file_path}:{from_line}->{to_line}"
            surrounding = _get_surrounding_lines(file_path, from_line, to_line, repo_root)

            gap = CoverageGap(
                file_path=file_path,
                target_symbol=containing_symbol,
                branch=BranchGap(from_line=from_line, to_line=to_line),
                surrounding_lines=surrounding,
                kind="branch",
                origin="full",
                gap_id=gap_id,
            )
            gaps.append(gap)

    logger.info("Parsed %d coverage gaps from %d files", len(gaps), len(files))
    return gaps
