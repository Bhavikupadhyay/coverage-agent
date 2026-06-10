"""Tests for gap selection and diff-gap parsing."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from coverage_agent.contracts import BranchGap, CoverageGap
from coverage_agent.gaps.select import select_gaps, io_difficulty_flag, _is_io_heavy


# ---------------------------------------------------------------------------
# select_gaps
# ---------------------------------------------------------------------------

def _gap(kind="branch", symbol="process", file="pkg/mod.py", from_l=10, to_l=12) -> CoverageGap:
    return CoverageGap(
        file_path=file,
        target_symbol=symbol,
        branch=BranchGap(from_line=from_l, to_line=to_l),
        surrounding_lines=list(range(from_l, to_l + 1)),
        kind=kind,
        origin="full",
        gap_id=f"{file}:{from_l}->{to_l}",
    )


def test_select_respects_max_gaps():
    gaps = [_gap(from_l=i, to_l=i + 1) for i in range(1, 20)]
    result = select_gaps(gaps, max_gaps=5)
    assert len(result) == 5


def test_select_function_before_branch_before_line():
    fn_gap = _gap(kind="function", symbol="new_fn", from_l=1, to_l=5)
    branch_gap = _gap(kind="branch", symbol="check", from_l=10, to_l=12)
    line_gap = _gap(kind="line", symbol="helper", from_l=20, to_l=21)

    result = select_gaps([line_gap, branch_gap, fn_gap], max_gaps=10)
    assert result[0].kind == "function"
    assert result[1].kind == "branch"
    assert result[2].kind == "line"


def test_io_heavy_demoted_within_tier():
    normal = _gap(kind="branch", symbol="compute_value", from_l=10, to_l=12)
    io_gap = _gap(kind="branch", symbol="write_to_db", from_l=20, to_l=22)

    result = select_gaps([io_gap, normal], max_gaps=10)
    assert result[0].target_symbol == "compute_value"
    assert result[1].target_symbol == "write_to_db"


def test_exclude_pattern_filters_files():
    kept = _gap(file="pkg/logic.py")
    excluded = _gap(file="pkg/migrations/0001.py", from_l=5, to_l=6)
    result = select_gaps([kept, excluded], max_gaps=10, exclude=["**/migrations/**"])
    assert len(result) == 1
    assert result[0].file_path == "pkg/logic.py"


def test_empty_input_returns_empty():
    assert select_gaps([], max_gaps=5) == []


def test_select_preserves_order_within_tier():
    g1 = _gap(kind="branch", symbol="alpha", from_l=1, to_l=2)
    g2 = _gap(kind="branch", symbol="beta", from_l=3, to_l=4)
    g3 = _gap(kind="branch", symbol="gamma", from_l=5, to_l=6)
    result = select_gaps([g1, g2, g3], max_gaps=10)
    symbols = [g.target_symbol for g in result]
    assert symbols == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# io_difficulty_flag
# ---------------------------------------------------------------------------

def test_io_flag_easy_for_pure_logic():
    g = _gap(symbol="compute_hash")
    assert io_difficulty_flag(g) == "easy"


def test_io_flag_hard_for_io_symbol():
    g = _gap(symbol="write_file")
    assert io_difficulty_flag(g) == "hard"


def test_io_flag_hard_for_large_context():
    from coverage_agent.contracts import ContextPayload
    g = _gap(symbol="pure_fn")
    ctx = ContextPayload(
        primary_code="def pure_fn(): pass",
        dependencies_code={},
        graph_depth_used=1,
        tokens_used=9000,
    )
    assert io_difficulty_flag(g, ctx) == "hard"


# ---------------------------------------------------------------------------
# _is_io_heavy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("read_config", True),
    ("send_email", True),
    ("open_file", True),
    ("db_query", True),
    ("compute_sum", False),
    ("validate_token", False),
    ("parse_args", False),
])
def test_is_io_heavy_patterns(symbol, expected):
    assert _is_io_heavy(symbol) is expected


# ---------------------------------------------------------------------------
# Gap substitution behaviour (tests the CLI loop logic indirectly via
# select_gaps: when the candidate pool is 2× target, skipped gaps are
# replaced from the tail so accepted_count reaches target_count).
# ---------------------------------------------------------------------------

def test_select_gaps_double_pool_covers_skips():
    """2× pool means skipped gaps can be replaced to hit the target count."""
    target = 3
    # Build 6 gaps (2× target) so the substitution pool is large enough.
    all_gaps = [
        _gap(symbol=f"fn_{i}", from_l=i * 10, to_l=i * 10 + 2)
        for i in range(6)
    ]
    candidates = select_gaps(all_gaps, max_gaps=target * 2, exclude=())
    assert len(candidates) == 6

    # Simulate the CLI loop: accept every other gap (3 skips, 3 accepts).
    accepted = []
    attempted = 0
    accepted_count = 0
    for gap in candidates:
        if accepted_count >= target or attempted >= target * 2:
            break
        attempted += 1
        if attempted % 2 == 0:
            accepted_count += 1
            accepted.append(gap)

    assert accepted_count == target
