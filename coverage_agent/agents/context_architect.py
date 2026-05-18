import logging
from typing import TYPE_CHECKING

from coverage_agent.credentials import Credentials
from coverage_agent.contracts.schemas import ContextPayload, CoverageGap
from coverage_agent.context.jedi_graph import build_context

if TYPE_CHECKING:
    from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


def _heuristic_depth(gap: CoverageGap) -> int:
    """Picks graph_depth without burning an LLM call.

    Empirical pattern from earlier runs: the LLM picked depth=1 ~90% of the
    time and almost never depth=2. The surrounding_lines count is a decent
    proxy for "how much logic does this gap touch":

      <= 3 lines uncovered  → tiny gap, function probably standalone        → 0
      4-30  lines uncovered → typical case, want immediate callees          → 1
      > 30  lines uncovered → large block of logic, pull in 2-hop neighbors → 2

    Saves one LLM call per gap and matches the LLM's prior decisions on the
    benchmark set within one step.
    """
    n = len(gap.surrounding_lines)
    if n <= 3:
        return 0
    if n > 30:
        return 2
    return 1


class ContextArchitect:
    """
    Constructs the context payload needed to write a test for a coverage gap.

    Depth is chosen by a deterministic heuristic on gap size — no LLM call —
    which saves one round-trip per gap and matched the LLM's prior decisions
    within one step on the benchmark set. Override via `depth_override` if
    a caller wants exact control.

    When a sandbox is provided (live/web mode), Jedi runs inside E2B where
    the repo already lives. When sandbox is None (CLI fallback), the local
    jedi_graph path is used.
    """

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        gap: CoverageGap,
        depth_override: int | None = None,
        repo_root: str = ".",
        sandbox: "E2BSandbox | None" = None,
    ) -> ContextPayload:
        if depth_override is not None:
            depth = depth_override
        elif self.creds.is_offline:
            depth = 1
            logger.info("[OFFLINE] ContextArchitect — using depth=1 for %s", gap.gap_id)
        else:
            depth = _heuristic_depth(gap)
            logger.info(
                "ContextArchitect: depth=%d for %s (heuristic, %d surrounding lines)",
                depth, gap.gap_id, len(gap.surrounding_lines),
            )

        if sandbox is not None:
            context_dict = sandbox.build_context(
                gap.file_path,
                gap.target_symbol,
                depth,
                from_line=gap.branch.from_line,
            )
            return ContextPayload(**context_dict)

        return build_context(
            gap.file_path,
            gap.target_symbol,
            depth=depth,
            repo_root=repo_root,
            offline=self.creds.is_offline,
            from_line=gap.branch.from_line,
        )
