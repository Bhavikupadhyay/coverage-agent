"""
TestWriter — generates pytest tests for a single coverage gap.

Single-shot path: one litellm.completion call, extract code block.
ReAct path: function-calling loop with tools from engine/tools.py.

The ReAct path activates when the model's registry entry has tool_calling=true
AND config.max_tool_calls > 0. It falls back to single-shot on budget
exhaustion or if the model doesn't support tool calling.

Every tool call is appended to the caller-supplied `trace` list as a plain dict
(serialized into AgentTrace by run_pipeline).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import litellm

from coverage_agent.config import AgentConfig
from coverage_agent.credentials import Credentials, list_models
from coverage_agent.contracts import ContextPayload, CoverageGap, DraftTest

logger = logging.getLogger(__name__)

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

When using tools:
- Use read_source to inspect code you need but don't have
- Use run_candidate to verify your test before returning the final answer
- Return your final test as a plain Python code block (```python ... ```) in a text response
"""


def _extract_code_block(content: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_mocks(test_code: str) -> list[str]:
    return re.findall(r'patch\(["\']([^"\']+)["\']', test_code)


def _layout_import_hint(file_path: str) -> str:
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


def _model_supports_tools(model: str) -> bool:
    """Returns True if the registry entry for this model has tool_calling=true."""
    for entry in list_models():
        if entry.get("id") == model:
            return bool(entry.get("tool_calling", False))
    # Unknown model: assume tool calling supported (litellm will surface errors).
    return True


def _token_count(response) -> int:
    try:
        return response.usage.total_tokens or 0
    except Exception:
        return 0


def _build_user_prompt(
    gap: CoverageGap,
    context: ContextPayload,
    critique: str | None,
    react_mode: bool,
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

    tools_hint = (
        "\n\nYou have tools available. Use run_candidate to verify your test works before "
        "submitting the final answer. When you're satisfied, return the final test code in "
        "a ```python ... ``` block in a regular text response (not a tool call)."
        if react_mode else ""
    )

    return (
        f"Write 1-2 pytest test functions that cover the following uncovered branch.\n\n"
        f"File: {gap.file_path}\n"
        f"Function: {gap.target_symbol}\n"
        f"Uncovered branch: line {gap.branch.from_line} -> line {gap.branch.to_line}\n"
        f"{_layout_import_hint(gap.file_path)}"
        f"\nTarget function source:\n```python\n{context.primary_code}\n```"
        f"{deps_section}"
        f"{hint_section}"
        f"{tools_hint}"
        f"{retry_section}\n\n"
        "Return only the complete Python test file (imports + test functions). "
        "No explanation."
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
        config: AgentConfig | None = None,
        trace: list | None = None,
    ) -> DraftTest:
        cfg = config or AgentConfig()
        use_react = (
            cfg.max_tool_calls > 0
            and _model_supports_tools(self.creds.llm_model)
        )

        if use_react:
            test_code = self._generate_react(gap, context, critique, cfg, trace or [])
        else:
            test_code = self._generate_single_shot(gap, context, critique)

        return DraftTest(
            test_code=test_code,
            mocks_used=_extract_mocks(test_code),
            target_branch=gap.branch,
        )

    def _generate_single_shot(
        self,
        gap: CoverageGap,
        context: ContextPayload,
        critique: str | None,
    ) -> str:
        user_prompt = _build_user_prompt(gap, context, critique, react_mode=False)
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
            logger.error("TestWriter single-shot LLM call failed: %s", exc)
            raise

    def _generate_react(
        self,
        gap: CoverageGap,
        context: ContextPayload,
        critique: str | None,
        cfg: AgentConfig,
        trace: list,
    ) -> str:
        from coverage_agent.engine import tools as _tools

        repo_root = "."
        gap_kwargs = {
            "gap_from_line": gap.branch.from_line,
            "gap_to_line": gap.branch.to_line,
            "target_file": gap.file_path,
            "repo_root": repo_root,
        }

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(gap, context, critique, react_mode=True)},
        ]

        total_tokens = 0
        tool_calls_used = 0
        last_code: str | None = None

        for _ in range(cfg.max_tool_calls + 1):
            try:
                response = litellm.completion(
                    messages=messages,
                    tools=_tools.TOOLS_SPEC,
                    tool_choice="auto",
                    **self.creds.litellm_kwargs(),
                )
            except Exception as exc:
                logger.error("TestWriter ReAct LLM call failed: %s", exc)
                # Fall back to whatever we have; if nothing, re-raise.
                if last_code:
                    return last_code
                raise

            total_tokens += _token_count(response)

            # Budget check.
            try:
                cost = float(getattr(response, "cost", 0) or 0)
            except (TypeError, ValueError):
                cost = 0.0
            if cfg.budget_usd > 0 and cost > 0 and cost > cfg.budget_usd:
                logger.warning("TestWriter: budget_usd %.2f exceeded — stopping ReAct loop", cfg.budget_usd)
                break

            msg = response.choices[0].message

            # Text response → extract code and finish.
            if not getattr(msg, "tool_calls", None):
                content = msg.content or ""
                last_code = _extract_code_block(content)
                messages.append({"role": "assistant", "content": content})
                break

            # Tool call(s) → dispatch each.
            messages.append(msg)
            for tc in msg.tool_calls:
                tool_calls_used += 1
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_input = {}

                observation = _tools.dispatch(tool_name, tool_input, **gap_kwargs)

                trace.append({
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": observation[:500],
                    "tokens_used": total_tokens,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

            if tool_calls_used >= cfg.max_tool_calls:
                logger.info("TestWriter: max_tool_calls=%d reached", cfg.max_tool_calls)
                # Ask for a final answer without tools.
                messages.append({
                    "role": "user",
                    "content": "Budget reached. Return your best test code now as a ```python ... ``` block.",
                })
                try:
                    final = litellm.completion(
                        messages=messages,
                        **self.creds.litellm_kwargs(),
                    )
                    last_code = _extract_code_block(final.choices[0].message.content or "")
                except Exception:
                    pass
                break

        if last_code:
            return last_code

        # Safety fallback: single-shot with no tools.
        logger.warning("TestWriter: ReAct loop produced no code — falling back to single-shot")
        return self._generate_single_shot(gap, context, critique)
