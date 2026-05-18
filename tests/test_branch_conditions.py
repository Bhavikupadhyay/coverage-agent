"""Phase C: branch-condition AST extractor.

The whole point is to hand TestWriter the exact condition that gates an
uncovered branch, so it can pick inputs that trigger it. These tests pin
both the happy path and the defensive None-fallback behaviour.
"""
from __future__ import annotations

import pytest

from coverage_agent.context.branch_conditions import (
    extract_branch_condition_from_source,
)


# ---------------------------------------------------------------------------
# Direct line matches — the common case
# ---------------------------------------------------------------------------

def test_simple_if_returns_condition():
    source = (
        "def check(x):\n"     # line 1
        "    if x > 5:\n"     # line 2
        "        return 'big'\n"  # line 3
        "    return 'small'\n"   # line 4
    )
    assert extract_branch_condition_from_source(source, from_line=2) == "x > 5"


def test_compound_condition_returns_full_expression():
    source = (
        "def f(a, b):\n"
        "    if a is not None and b > 0:\n"   # line 2
        "        return a + b\n"
        "    return None\n"
    )
    cond = extract_branch_condition_from_source(source, from_line=2)
    assert cond is not None
    assert "a is not None" in cond
    assert "b > 0" in cond


def test_while_loop_returns_condition():
    source = (
        "def loop(n):\n"
        "    while n > 0:\n"   # line 2
        "        n -= 1\n"
    )
    assert extract_branch_condition_from_source(source, from_line=2) == "n > 0"


def test_for_loop_returns_iteration_description():
    source = (
        "def each(xs):\n"
        "    for item in xs:\n"   # line 2
        "        print(item)\n"
    )
    cond = extract_branch_condition_from_source(source, from_line=2)
    assert cond is not None
    assert "iteration over" in cond
    assert "xs" in cond


def test_try_block_returns_exception_description():
    source = (
        "def safe():\n"
        "    try:\n"   # line 2
        "        risky()\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    cond = extract_branch_condition_from_source(source, from_line=2)
    assert cond is not None
    assert "exception" in cond.lower()


# ---------------------------------------------------------------------------
# Fallback: containing block when from_line doesn't land on the branch itself
# ---------------------------------------------------------------------------

def test_line_inside_if_block_returns_enclosing_condition():
    """Coverage sometimes reports the body line, not the conditional line."""
    source = (
        "def check(x):\n"     # 1
        "    if x > 5:\n"     # 2 — the conditional
        "        y = x * 2\n" # 3 — inside the block
        "        return y\n"  # 4 — also inside
        "    return None\n"   # 5
    )
    assert extract_branch_condition_from_source(source, from_line=3) == "x > 5"


def test_nested_if_picks_smallest_enclosing_block():
    """Multiple nested branches contain the same line — choose the tightest."""
    source = (
        "def outer(x, y):\n"        # 1
        "    if x > 0:\n"            # 2
        "        if y > 0:\n"        # 3 — tightest enclosing for line 4
        "            return x + y\n" # 4
        "    return None\n"          # 5
    )
    assert extract_branch_condition_from_source(source, from_line=4) == "y > 0"


# ---------------------------------------------------------------------------
# Defensive: extractor never raises, always returns None on failure
# ---------------------------------------------------------------------------

def test_syntax_error_returns_none():
    assert extract_branch_condition_from_source("def broken(:\n    pass\n", from_line=1) is None


def test_line_outside_file_returns_none():
    source = "def f():\n    return 1\n"
    assert extract_branch_condition_from_source(source, from_line=999) is None


def test_line_not_inside_any_branch_returns_none():
    """Plain code with no branches at the target line."""
    source = "x = 1\ny = 2\nz = x + y\n"
    assert extract_branch_condition_from_source(source, from_line=2) is None


# ---------------------------------------------------------------------------
# The motivating example: requests.check_compatibility pattern
# ---------------------------------------------------------------------------

def test_motivating_example_check_compatibility_like():
    """The condition shape that confused TestWriter on the v3 benchmark.

    Function only warns conditionally — TestWriter would assert_called for
    every parametrize case and fail. With the hint, it sees the actual
    inequality and can pick inputs accordingly.
    """
    source = (
        "import warnings\n"            # 1
        "def check_compatibility(version_str, min_version):\n"   # 2
        "    if version_str < min_version:\n"                     # 3
        "        warnings.warn('outdated')\n"                     # 4
    )
    hint = extract_branch_condition_from_source(source, from_line=3)
    assert hint == "version_str < min_version"
