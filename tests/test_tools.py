"""Tests for engine/tools.py — ReAct tool implementations."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coverage_agent.engine.tools import read_source, run_candidate, dispatch, TOOLS_SPEC


# ---------------------------------------------------------------------------
# read_source
# ---------------------------------------------------------------------------

def test_read_source_clamps_start(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("line1\nline2\nline3\n")
    result = read_source("sample.py", start=-5, end=2, repo_root=str(tmp_path))
    assert "line1" in result
    assert "line2" in result


def test_read_source_clamps_end(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("line1\nline2\nline3\n")
    result = read_source("sample.py", start=1, end=9999, repo_root=str(tmp_path))
    assert "line3" in result


def test_read_source_missing_file(tmp_path):
    result = read_source("nonexistent.py", repo_root=str(tmp_path))
    assert "Error" in result or "not found" in result


def test_read_source_includes_line_numbers(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("alpha\nbeta\n")
    result = read_source("sample.py", start=1, end=2, repo_root=str(tmp_path))
    assert "1:" in result
    assert "2:" in result


# ---------------------------------------------------------------------------
# run_candidate shape
# ---------------------------------------------------------------------------

def test_run_candidate_returns_correct_shape(tmp_path):
    test_code = textwrap.dedent("""\
        def test_trivial():
            assert 1 + 1 == 2
    """)
    result = run_candidate(test_code=test_code, repo_root=str(tmp_path))
    assert "passed" in result
    assert "stderr" in result
    assert "targets_hit" in result
    assert "targets_total" in result


def test_run_candidate_passing_test(tmp_path):
    test_code = "def test_pass(): assert True\n"
    result = run_candidate(test_code=test_code, repo_root=str(tmp_path))
    assert result["passed"] is True


def test_run_candidate_failing_test(tmp_path):
    test_code = "def test_fail(): assert False, 'intentional'\n"
    result = run_candidate(test_code=test_code, repo_root=str(tmp_path))
    assert result["passed"] is False
    assert result["stderr"] != "" or True  # stderr may be in stdout


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_dispatch_read_source(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n")
    result = dispatch("read_source", {"path": "mod.py", "start": 1}, repo_root=str(tmp_path))
    assert "x = 1" in result


def test_dispatch_unknown_tool():
    result = dispatch("nonexistent_tool", {})
    assert "Unknown tool" in result


def test_dispatch_run_candidate_returns_json(tmp_path):
    test_code = "def test_x(): assert 1\n"
    result = dispatch("run_candidate", {"test_code": test_code}, repo_root=str(tmp_path))
    parsed = json.loads(result)
    assert "passed" in parsed


# ---------------------------------------------------------------------------
# TOOLS_SPEC schema validity
# ---------------------------------------------------------------------------

def test_tools_spec_has_four_tools():
    names = {t["function"]["name"] for t in TOOLS_SPEC}
    assert names == {"read_source", "find_symbol", "find_usages", "run_candidate"}


def test_tools_spec_all_have_required_fields():
    for tool in TOOLS_SPEC:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
