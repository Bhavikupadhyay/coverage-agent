"""
Deterministic gap selection.

Selection order: function gaps (new files) → branch gaps → line gaps.
Within each tier, IO-heavy symbols are demoted to the end of that tier.
Hard cap at max_gaps. This absorbs the old gap_prioritizer and gap_filter agents.
"""
from __future__ import annotations

import fnmatch
import logging
from typing import Sequence

from coverage_agent.contracts import CoverageGap, ContextPayload

logger = logging.getLogger(__name__)

_IO_SUBSTRINGS = frozenset({
    "open", "read", "write", "send", "recv", "socket", "connect",
    "request", "response", "fetch", "download", "upload",
    "db", "database", "query", "execute", "cursor",
    "file", "stream", "pipe",
})

_KIND_ORDER = {"function": 0, "branch": 1, "line": 2}


def _is_io_heavy(symbol: str) -> bool:
    low = symbol.lower()
    return any(sub in low for sub in _IO_SUBSTRINGS)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    fp = path.replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(fp, pat) or fnmatch.fnmatch(fp.split("/")[-1], pat):
            return True
    return False


def select_gaps(
    gaps: Sequence[CoverageGap],
    max_gaps: int = 10,
    exclude: Sequence[str] = (),
) -> list[CoverageGap]:
    """Selects and ranks gaps for the engine pipeline.

    Args:
        gaps: Full list of candidate gaps.
        max_gaps: Hard cap on returned gaps.
        exclude: Glob patterns matching file paths to skip.

    Returns:
        Ordered list of at most max_gaps gaps.
    """
    filtered = [g for g in gaps if not _matches_any(g.file_path, exclude)]

    # Stable sort: kind tier first, then IO difficulty within tier.
    def _sort_key(g: CoverageGap) -> tuple[int, int]:
        tier = _KIND_ORDER.get(g.kind, 1)
        io_penalty = 1 if _is_io_heavy(g.target_symbol) else 0
        return (tier, io_penalty)

    ranked = sorted(filtered, key=_sort_key)
    selected = ranked[:max_gaps]

    logger.info(
        "select_gaps: %d candidates → %d selected (max_gaps=%d)",
        len(filtered), len(selected), max_gaps,
    )
    return selected


def cluster_gaps(selected: list[CoverageGap]) -> list[list[CoverageGap]]:
    """Groups gaps by (file_path, target_symbol) into clusters.

    Cluster ordering follows the position of each cluster's first (best-ranked)
    member in the input list — the selection order from select_gaps is preserved.
    Within a cluster, gaps appear in their original selection order.

    Args:
        selected: Already-ranked list from select_gaps.

    Returns:
        Ordered list of clusters, each cluster an ordered list[CoverageGap].
        Single-gap clusters are included as-is (list of length 1).
    """
    seen: dict[tuple[str, str], int] = {}   # (file, symbol) → cluster index
    clusters: list[list[CoverageGap]] = []

    for gap in selected:
        key = (gap.file_path, gap.target_symbol)
        if key not in seen:
            seen[key] = len(clusters)
            clusters.append([gap])
        else:
            clusters[seen[key]].append(gap)

    return clusters


def io_difficulty_flag(gap: CoverageGap, context: ContextPayload | None = None) -> str:
    """Returns 'hard' if the gap looks IO-heavy, 'easy' otherwise.

    Used by the graph's gap_filter node to mark difficulty without skipping.
    The context token count can bump an easy gap to hard as a secondary signal.
    """
    if _is_io_heavy(gap.target_symbol):
        return "hard"
    if context is not None and context.tokens_used > 8000:
        return "hard"
    return "easy"
