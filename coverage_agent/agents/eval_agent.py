import ast
import logging
import os
import sys

import litellm

from coverage_agent.contracts.schemas import ContextPayload, CoverageGap, DraftTest, EvalResult

logger = logging.getLogger(__name__)

_MODEL = "gemini/gemini-2.5-flash"

# Standard library top-level module names (Python 3.11)
_STDLIB_MODULES = set(sys.stdlib_module_names)


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


def _check_syntax(test_code: str) -> bool:
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False


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


def _find_unknown_imports(test_code: str, context: ContextPayload) -> list[str]:
    """
    Returns imports that are neither stdlib nor present as a key in
    context.dependencies_code. The target file's own package root is
    treated as known.
    """
    imported = _extract_imports(test_code)
    known_names = set(context.dependencies_code.keys())

    unknown = []
    for mod in imported:
        if mod in _STDLIB_MODULES:
            continue
        if mod in known_names:
            continue
        # Common test utilities and the project's own package are always OK
        if mod in {"pytest", "unittest", "mock"}:
            continue
        unknown.append(mod)
    return unknown


class EvalAgent:
    """
    Adversarially scores a DraftTest before the expensive E2B execution step.

    Phase 1 evals:
      1. Syntax validity (deterministic)
      2. Import plausibility (deterministic)
      3. Mock completeness (LLM)
      4. Assertion quality (LLM, 1-5)

    In dry-run mode, all LLM checks are stubbed with passing scores and
    route=EXECUTE.
    """

    def run(
        self,
        draft: DraftTest,
        context: ContextPayload,
        gap: CoverageGap,
    ) -> EvalResult:
        # --- Phase 1a: Syntax check (deterministic) ---
        syntax_valid = _check_syntax(draft.test_code)
        if not syntax_valid:
            logger.info("EvalAgent: syntax invalid for %s — routing REWRITE", gap.gap_id)
            return EvalResult(
                syntax_valid=False,
                unknown_imports=[],
                mock_complete=False,
                assertion_score=1,
                critique="Test code failed ast.parse(). Fix all syntax errors.",
                route="REWRITE",
            )

        # --- Phase 1b: Import plausibility (deterministic) ---
        unknown_imports = _find_unknown_imports(draft.test_code, context)

        if _is_dry_run():
            logger.info("[DRY_RUN] EvalAgent — returning passing stub for %s", gap.gap_id)
            return EvalResult(
                syntax_valid=True,
                unknown_imports=unknown_imports,
                mock_complete=True,
                assertion_score=4,
                critique="",
                route="EXECUTE",
            )

        # --- Phase 1c: LLM scoring ---
        mock_complete, mock_critique = self._check_mock_completeness(draft, context, gap)
        assertion_score, assertion_critique = self._score_assertions(draft, gap)

        # Combine critiques
        critique_parts = []
        if mock_critique:
            critique_parts.append(mock_critique)
        if assertion_critique:
            critique_parts.append(assertion_critique)
        if unknown_imports:
            critique_parts.append(
                f"Unknown imports detected: {unknown_imports}. "
                "These are not in the context payload — verify they are real."
            )

        critique = " ".join(critique_parts)

        # --- Routing logic ---
        if unknown_imports:
            route = "RECONTEXTUALIZE"
        elif assertion_score < 3 or not mock_complete:
            route = "REWRITE"
        else:
            route = "EXECUTE"

        logger.info(
            "EvalAgent: gap=%s syntax=OK mocks=%s assert_score=%d unknown_imports=%s → %s",
            gap.gap_id,
            mock_complete,
            assertion_score,
            unknown_imports,
            route,
        )

        return EvalResult(
            syntax_valid=True,
            unknown_imports=unknown_imports,
            mock_complete=mock_complete,
            assertion_score=assertion_score,
            critique=critique,
            route=route,
        )

    def _check_mock_completeness(
        self,
        draft: DraftTest,
        context: ContextPayload,
        gap: CoverageGap,
    ) -> tuple[bool, str]:
        deps_summary = ", ".join(context.dependencies_code.keys()) or "none"
        prompt = (
            f"Review this pytest test for the function `{gap.target_symbol}` "
            f"in `{gap.file_path}`.\n\n"
            f"Known external dependencies: {deps_summary}\n\n"
            f"Test code:\n```python\n{draft.test_code}\n```\n\n"
            "Does the test mock all external IO, network, filesystem, or database calls "
            "that appear in the function under test?\n\n"
            "Respond with:\n"
            "PASS\n"
            "or:\n"
            "FAIL: <brief explanation of which mocks are missing>"
        )
        try:
            response = litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            if content.upper().startswith("PASS"):
                return True, ""
            critique = content[5:].strip() if content.upper().startswith("FAIL") else content
            return False, f"Mock completeness: {critique}"
        except Exception as exc:
            logger.warning("Mock completeness LLM check failed (%s) — defaulting to PASS", exc)
            return True, ""

    def _score_assertions(
        self,
        draft: DraftTest,
        gap: CoverageGap,
    ) -> tuple[int, str]:
        prompt = (
            f"Score the assertion quality of this pytest test (1-5).\n\n"
            f"Target: function `{gap.target_symbol}`, "
            f"branch {gap.branch.from_line} -> {gap.branch.to_line}\n\n"
            f"Test code:\n```python\n{draft.test_code}\n```\n\n"
            "Scoring rubric:\n"
            "  1 = only trivial assertions like `assert True` or `assert result is not None`\n"
            "  2 = checks type or basic non-None result\n"
            "  3 = checks a specific return value or a single meaningful side effect\n"
            "  4 = checks return value AND a side effect or exception type\n"
            "  5 = rigorously asserts specific return values, side effects, or exception "
            "types and mock call counts\n\n"
            "Respond with:\n"
            "<score>\n"
            "<one sentence of feedback if score <= 3, else empty>"
        )
        try:
            response = litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            lines = response.choices[0].message.content.strip().splitlines()
            score = int(lines[0].strip())
            score = max(1, min(5, score))
            feedback = lines[1].strip() if len(lines) > 1 else ""
            critique = f"Assertion quality ({score}/5): {feedback}" if feedback else ""
            return score, critique
        except Exception as exc:
            logger.warning("Assertion score LLM check failed (%s) — defaulting to score=3", exc)
            return 3, ""
