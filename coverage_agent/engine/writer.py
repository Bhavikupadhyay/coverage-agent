import logging
import re
from pathlib import Path

import litellm

from coverage_agent.credentials import Credentials
from coverage_agent.contracts import ContextPayload, CoverageGap, DraftTest

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Common src-layout prefixes that wrap an installable package. The prefix is
# repo layout only — tests must import the package underneath, not the prefix.
_SRC_LAYOUT_DIRS = frozenset({"src", "lib", "python", "source", "packages", "pkg"})


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
- Repo paths often include a layout prefix (`src/`, `lib/`, etc.). That prefix is not
  a Python package after `pip install -e .` — import the real top-level package name
  (the directory after the prefix, e.g. `requests`), never `import src` or `from src.…`
"""


def _extract_code_block(content: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_mocks(test_code: str) -> list[str]:
    return re.findall(r'patch\(["\']([^"\']+)["\']', test_code)


def _layout_import_hint(file_path: str) -> str:
    """Extra prompt hint when the gap path uses a src/lib layout."""
    parts = [p for p in Path(file_path).parts if p not in (".", "..")]
    if len(parts) < 2 or parts[0] not in _SRC_LAYOUT_DIRS:
        return ""
    pkg = parts[1].removesuffix(".py")
    if not pkg:
        return ""
    root = parts[0]
    return (
        f"\n\nLayout note: the file path starts with `{root}/` — that folder is repo layout only. "
        f"The installable package is `{pkg}`. Import using `{pkg}` (e.g. `import {pkg}` or "
        f"`from {pkg}.…`), not `{root}`."
    )


class TestWriter:
    """Generates pytest test functions targeting a specific uncovered branch."""

    __test__ = False

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        gap: CoverageGap,
        context: ContextPayload,
        critique: str | None = None,
    ) -> DraftTest:
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

        hint_section = ""
        if context.branch_condition_hint:
            hint_section = (
                f"\n\nBranch trigger condition (line {gap.branch.from_line}): "
                f"`{context.branch_condition_hint}`\n"
                "To hit the target branch, choose test inputs that make this "
                "condition evaluate the right way. Reason carefully about which "
                "direction (truthy vs falsy) leads to the uncovered line "
                f"{gap.branch.to_line}."
            )

        retry_section = ""
        if critique:
            retry_section = (
                f"\n\nThis is a RETRY. Your previous attempt did not succeed:\n\n"
                f"{critique}\n\n"
                "Rewrite the test from scratch addressing the specific problem above. "
                "Do not regenerate the same test code — change inputs, mocks, "
                "assertions, or imports as needed to fix the reported issue."
            )

        user_prompt = (
            f"Write 1-2 pytest test functions that cover the following uncovered branch.\n\n"
            f"File: {gap.file_path}\n"
            f"Function: {gap.target_symbol}\n"
            f"Uncovered branch: line {gap.branch.from_line} -> line {gap.branch.to_line}\n"
            f"{_layout_import_hint(gap.file_path)}"
            f"\nTarget function source:\n```python\n{context.primary_code}\n```"
            f"{deps_section}"
            f"{hint_section}"
            f"{retry_section}\n\n"
            "Return only the complete Python test file (imports + test functions). "
            "No explanation."
        )

        try:
            response = litellm.completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **self.creds.litellm_kwargs(),
            )
            content = response.choices[0].message.content or ""
            return _extract_code_block(content)
        except Exception as exc:
            logger.error("TestWriter LLM call failed: %s", exc)
            raise
