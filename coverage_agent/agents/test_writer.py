import logging
import re
from pathlib import Path

import litellm

from coverage_agent.config import get_model, is_dry_run
from coverage_agent.contracts.schemas import ContextPayload, CoverageGap, DraftTest

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


_SYSTEM_PROMPT = """\
You are an expert Python test engineer. Your task is to write executable pytest code.

Rules:
- Write pytest functions only (no unittest.TestCase classes unless the target uses them)
- Use unittest.mock.patch for ALL external IO, network, filesystem, or database calls
- Every test must assert something meaningful about state or behavior
- Do NOT use assert result is not None or assert True as the only assertion
- Assert specific return values, side effects, raised exceptions, or mock call counts
- Include all necessary imports at the top of the file
- Target the specific uncovered branch identified in the task
"""


def _extract_code_block(content: str) -> str:
    """Strips markdown code fences if the LLM wrapped the output."""
    match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_mocks(test_code: str) -> list[str]:
    """Extracts all patch targets from @patch decorators and patch() calls."""
    return re.findall(r'patch\(["\']([^"\']+)["\']', test_code)


class TestWriter:
    """
    Generates 1-2 pytest test functions targeting a specific uncovered branch.

    In dry-run mode, returns the sample_test.py fixture wrapped in a DraftTest.
    In live mode, calls the LLM with context and an optional critique from the
    Eval Agent if this is a retry after a REWRITE routing decision.
    """

    def run(
        self,
        gap: CoverageGap,
        context: ContextPayload,
        critique: str | None = None,
    ) -> DraftTest:
        if is_dry_run():
            logger.info("[DRY_RUN] TestWriter — returning fixture test for %s", gap.gap_id)
            test_code = (_FIXTURES_DIR / "sample_test.py").read_text(encoding="utf-8")
            return DraftTest(
                test_code=test_code,
                mocks_used=_extract_mocks(test_code),
                target_branch=gap.branch,
            )

        test_code = self._generate(gap, context, critique)
        return DraftTest(
            test_code=test_code,
            mocks_used=_extract_mocks(test_code),
            target_branch=gap.branch,
        )

    def _generate(
        self,
        gap: CoverageGap,
        context: ContextPayload,
        critique: str | None,
    ) -> str:
        deps_section = ""
        if context.dependencies_code:
            deps_section = "\n\nDependency signatures:\n" + "\n\n".join(
                f"# {name}\n{src}" for name, src in context.dependencies_code.items()
            )

        retry_section = ""
        if critique:
            retry_section = (
                f"\n\nYour previous attempt failed for this reason:\n{critique}\n"
                "Fix the issues described above in your new attempt."
            )

        user_prompt = (
            f"Write 1-2 pytest test functions that cover the following uncovered branch.\n\n"
            f"File: {gap.file_path}\n"
            f"Function: {gap.target_symbol}\n"
            f"Uncovered branch: line {gap.branch.from_line} -> line {gap.branch.to_line}\n\n"
            f"Target function source:\n```python\n{context.primary_code}\n```"
            f"{deps_section}"
            f"{retry_section}\n\n"
            "Return only the complete Python test file (imports + test functions). "
            "No explanation."
        )

        try:
            response = litellm.completion(
                model=get_model(),
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            return _extract_code_block(content)
        except Exception as exc:
            logger.error("TestWriter LLM call failed: %s", exc)
            raise
