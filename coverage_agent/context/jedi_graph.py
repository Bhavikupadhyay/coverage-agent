import inspect
import json
import logging
import os
import textwrap
from pathlib import Path

import jedi
import litellm

from coverage_agent.contracts.schemas import ContextPayload

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 15000

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


def _count_tokens(text: str) -> int:
    try:
        return litellm.token_counter(model="gemini/gemini-2.5-flash", text=text)
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def _extract_function_source(file_path: str, target_symbol: str) -> str | None:
    """Uses Jedi to find and return the source of the target function."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        script = jedi.Script(source, path=file_path)
        names = script.get_names(all_scopes=True, definitions=True)

        for name in names:
            if name.name == target_symbol and name.type in ("function", "class"):
                start_line = name.line
                lines = source.splitlines()
                # Collect lines until indentation returns to base level or EOF
                base_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                func_lines = []
                for i, line in enumerate(lines[start_line - 1:], start=start_line):
                    if i > start_line and line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                        break
                    func_lines.append(line)
                return "\n".join(func_lines)
    except Exception as exc:
        logger.debug("Jedi source extraction failed for %s in %s: %s", target_symbol, file_path, exc)
    return None


def _extract_callees(file_path: str, target_symbol: str) -> dict[str, str]:
    """
    Resolves immediate callees of the target function using Jedi goto.
    Returns a map of {callee_name: source_or_signature}.
    """
    dependencies: dict[str, str] = {}
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        script = jedi.Script(source, path=file_path)
        names = script.get_names(all_scopes=True, references=True)

        for name in names:
            if name.name == target_symbol:
                continue
            try:
                definitions = name.goto()
                for defn in definitions:
                    if defn.name and defn.name not in dependencies:
                        sig = defn.get_signatures()
                        if sig:
                            dependencies[defn.name] = str(sig[0])
                        else:
                            dependencies[defn.name] = f"# {defn.type}: {defn.name}"
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Jedi callee extraction failed for %s: %s", target_symbol, exc)
    return dependencies


def build_context(file_path: str, target_symbol: str, depth: int = 1) -> ContextPayload:
    """
    Constructs a ContextPayload for the given target symbol.

    In DRY_RUN mode, returns the fixture from fixtures/sample_context.json.

    Depth 0: target function source only.
    Depth 1: target + immediate callees resolved via Jedi goto().

    Token budget is enforced at MAX_CONTEXT_TOKENS = 15000. If adding the next
    depth level would exceed the budget, traversal stops at the current depth.
    If Jedi resolution fails entirely, falls back to local file scope with
    fallback_used=True.
    """
    if _is_dry_run():
        fixture_path = _FIXTURES_DIR / "sample_context.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        logger.info("[DRY_RUN] build_context returning fixture for %s:%s", file_path, target_symbol)
        return ContextPayload(**data)

    fallback_used = False
    primary_code = _extract_function_source(file_path, target_symbol)

    if primary_code is None:
        # Fallback: return entire file as primary_code
        logger.warning("Jedi could not locate %s in %s — falling back to file scope", target_symbol, file_path)
        try:
            primary_code = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            primary_code = f"# Could not read {file_path}"
        fallback_used = True

    dependencies: dict[str, str] = {}
    depth_used = 0

    if depth >= 1 and not fallback_used:
        tokens_so_far = _count_tokens(primary_code)
        callees = _extract_callees(file_path, target_symbol)

        for name, source in callees.items():
            candidate_tokens = _count_tokens(source)
            if tokens_so_far + candidate_tokens > MAX_CONTEXT_TOKENS:
                logger.debug("Token budget reached at depth 1 — stopping before %s", name)
                break
            dependencies[name] = source
            tokens_so_far += candidate_tokens

        depth_used = 1

    tokens_used = _count_tokens(primary_code) + sum(_count_tokens(v) for v in dependencies.values())

    return ContextPayload(
        primary_code=primary_code,
        dependencies_code=dependencies,
        graph_depth_used=depth_used,
        tokens_used=tokens_used,
        fallback_used=fallback_used,
    )
