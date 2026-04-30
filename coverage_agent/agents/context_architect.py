import logging
from pathlib import Path
from typing import TYPE_CHECKING

import litellm

from coverage_agent.config import get_model, is_dry_run
from coverage_agent.contracts.schemas import ContextPayload, CoverageGap
from coverage_agent.context.jedi_graph import build_context

if TYPE_CHECKING:
    from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


class ContextArchitect:
    """
    Constructs the context payload needed to write a test for a coverage gap.

    When a sandbox is provided (live/web mode), Jedi runs inside E2B where
    the repo already lives — no local filesystem access. When sandbox is None
    (CLI fallback), the local jedi_graph path is used.
    """

    def run(
        self,
        gap: CoverageGap,
        depth_override: int | None = None,
        repo_root: str = ".",
        sandbox: "E2BSandbox | None" = None,
    ) -> ContextPayload:
        if depth_override is not None:
            depth = depth_override
        elif is_dry_run():
            depth = 1
            logger.info("[DRY_RUN] ContextArchitect — using depth=1 for %s", gap.gap_id)
        else:
            depth = self._decide_depth(gap, repo_root=repo_root, sandbox=sandbox)

        if sandbox is not None:
            context_dict = sandbox.build_context(gap.file_path, gap.target_symbol, depth)
            return ContextPayload(**context_dict)

        return build_context(gap.file_path, gap.target_symbol, depth=depth, repo_root=repo_root)

    def _decide_depth(
        self,
        gap: CoverageGap,
        repo_root: str = ".",
        sandbox: "E2BSandbox | None" = None,
    ) -> int:
        """Asks the LLM what graph_depth is needed. Falls back to 1 on any failure."""
        source_preview = "\n".join(f"line {ln}" for ln in gap.surrounding_lines[:20])
        try:
            if sandbox is not None:
                raw = sandbox._sandbox.files.read(f"/home/user/repo/{gap.file_path}")
                source_lines = raw.splitlines()
            else:
                source_lines = (Path(repo_root) / gap.file_path).read_text(encoding="utf-8").splitlines()
            source_preview = "\n".join(
                f"{ln}: {source_lines[ln - 1]}"
                for ln in gap.surrounding_lines[:20]
                if 0 < ln <= len(source_lines)
            )
        except Exception:
            pass

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
                model=get_model(),
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
