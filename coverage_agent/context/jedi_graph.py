import inspect
import json
import logging
import textwrap
from pathlib import Path

import jedi
import litellm

from coverage_agent.context.branch_conditions import extract_branch_condition_from_source
from coverage_agent.contracts.schemas import ContextPayload

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 15000

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _count_tokens(text: str) -> int:
    try:
        return litellm.token_counter(model="gemini/gemini-2.5-flash", text=text)
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def _extract_function_source(file_path: str, target_symbol: str) -> str | None:
    """Uses Jedi to find and return the source of the target function."""
    try:
        resolved = Path(file_path)
        source = resolved.read_text(encoding="utf-8")
        script = jedi.Script(source, path=str(resolved))
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
        resolved = Path(file_path)
        source = resolved.read_text(encoding="utf-8")
        script = jedi.Script(source, path=str(resolved))
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


def build_context(
    file_path: str,
    target_symbol: str,
    depth: int = 1,
    repo_root: str = ".",
    offline: bool = False,
    from_line: int | None = None,
) -> ContextPayload:
    """
    Constructs a ContextPayload for the given target symbol.

    file_path is repo-relative (e.g. "requests/auth.py"). repo_root is the
    local path to the cloned repository. All file reads are resolved as
    Path(repo_root) / file_path.

    In offline mode, returns the fixture from fixtures/sample_context.json.

    Depth 0: target function source only.
    Depth 1: target + immediate callees resolved via Jedi goto().
    Depth 2: target + callees + their callees (within token budget).

    Token budget is enforced at MAX_CONTEXT_TOKENS = 15000. Traversal stops
    when adding the next level would exceed the budget.
    Falls back to local file scope with fallback_used=True if Jedi fails.
    """
    if offline:
        fixture_path = _FIXTURES_DIR / "sample_context.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        logger.info("[OFFLINE] build_context returning fixture for %s:%s", file_path, target_symbol)
        return ContextPayload(**data)

    resolved_path = str(Path(repo_root) / file_path)
    fallback_used = False
    primary_code = _extract_function_source(resolved_path, target_symbol)

    if primary_code is None:
        # Fallback: return entire file as primary_code
        logger.warning("Jedi could not locate %s in %s — falling back to file scope", target_symbol, resolved_path)
        try:
            primary_code = Path(resolved_path).read_text(encoding="utf-8")
        except Exception:
            primary_code = f"# Could not read {resolved_path}"
        fallback_used = True

    dependencies: dict[str, str] = {}
    depth_used = 0

    if depth >= 1 and not fallback_used:
        tokens_so_far = _count_tokens(primary_code)
        callees = _extract_callees(resolved_path, target_symbol)

        for name, source in callees.items():
            candidate_tokens = _count_tokens(source)
            if tokens_so_far + candidate_tokens > MAX_CONTEXT_TOKENS:
                logger.debug("Token budget reached at depth 1 — stopping before %s", name)
                break
            dependencies[name] = source
            tokens_so_far += candidate_tokens

        depth_used = 1

    if depth >= 2 and not fallback_used and dependencies:
        depth2: dict[str, str] = {}
        for callee_name in list(dependencies):
            sub_callees = _extract_callees(resolved_path, callee_name)
            for name, source in sub_callees.items():
                if name in dependencies or name in depth2:
                    continue
                candidate_tokens = _count_tokens(source)
                if tokens_so_far + candidate_tokens > MAX_CONTEXT_TOKENS:
                    logger.debug("Token budget reached at depth 2 — stopping before %s", name)
                    break
                depth2[name] = source
                tokens_so_far += candidate_tokens
        if depth2:
            dependencies.update(depth2)
            depth_used = 2

    tokens_used = _count_tokens(primary_code) + sum(_count_tokens(v) for v in dependencies.values())

    # Best-effort branch condition extraction (Phase C). Hands TestWriter the
    # condition controlling the target branch so it can pick inputs that
    # actually trigger it. Silently None on extraction failure — the pipeline
    # works fine without the hint, it's a quality boost.
    branch_hint: str | None = None
    if from_line is not None:
        try:
            file_source = Path(resolved_path).read_text(encoding="utf-8")
            branch_hint = extract_branch_condition_from_source(file_source, from_line)
        except Exception as exc:
            logger.debug("Branch condition extraction failed for %s:L%d: %s", resolved_path, from_line, exc)

    return ContextPayload(
        primary_code=primary_code,
        dependencies_code=dependencies,
        graph_depth_used=depth_used,
        tokens_used=tokens_used,
        fallback_used=fallback_used,
        branch_condition_hint=branch_hint,
    )
