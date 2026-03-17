import logging
import os
from pathlib import Path

import litellm

from coverage_agent.contracts.schemas import ContextPayload, CoverageGap
from coverage_agent.context.jedi_graph import build_context

logger = logging.getLogger(__name__)

_MODEL = "gemini/gemini-2.5-flash"


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


class ContextArchitect:
    """
    Constructs the exact subset of codebase context needed to write a test
    for a specific coverage gap.

    In live mode: LLM decides the required graph_depth, then deterministic
    Jedi traversal executes it via build_context().

    In dry-run mode: depth is hardcoded to 1, returning the fixture payload.
    """

    def run(self, gap: CoverageGap, depth_override: int | None = None) -> ContextPayload:
        if depth_override is not None:
            depth = depth_override
        elif _is_dry_run():
            depth = 1
            logger.info("[DRY_RUN] ContextArchitect — using depth=1 for %s", gap.gap_id)
        else:
            depth = self._decide_depth(gap)

        return build_context(gap.file_path, gap.target_symbol, depth=depth)

    def _decide_depth(self, gap: CoverageGap) -> int:
        """
        Asks the LLM what graph_depth is needed to understand the target gap.
        Returns 0, 1, or 2. Falls back to 1 on any failure.
        """
        try:
            source_lines = Path(gap.file_path).read_text(encoding="utf-8").splitlines()
            source_preview = "\n".join(
                f"{ln}: {source_lines[ln - 1]}"
                for ln in gap.surrounding_lines[:20]
                if 0 < ln <= len(source_lines)
            )
        except Exception:
            source_preview = "\n".join(f"line {ln}" for ln in gap.surrounding_lines[:20])

        prompt = (
            f"You are analyzing a Python coverage gap to decide how much context is needed.\n\n"
            f"File: {gap.file_path}\n"
            f"Function: {gap.target_symbol}\n"
            f"Uncovered branch: line {gap.branch.from_line} -> line {gap.branch.to_line}\n"
            f"Function source:\n{source_preview}\n\n"
            "Decide the graph_depth needed:\n"
            "  0 = the target function source alone is sufficient\n"
            "  1 = need target + its immediate callees (most common)\n"
            "  2 = need target + callees + their callees (only for deeply nested logic)\n\n"
            "Respond with a single integer: 0, 1, or 2. Nothing else."
        )
        try:
            response = litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            depth = int(content)
            if depth not in (0, 1, 2):
                raise ValueError(f"depth {depth} out of range")
            logger.info("ContextArchitect decided depth=%d for %s", depth, gap.gap_id)
            return depth
        except Exception as exc:
            logger.warning("LLM depth decision failed (%s) — defaulting to depth=1", exc)
            return 1
