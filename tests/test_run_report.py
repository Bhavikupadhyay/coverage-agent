"""RunReport serialization round-trip tests."""
from __future__ import annotations

import json
from pathlib import Path

from coverage_agent.contracts import RunReport, GapResult, CoverageGap, BranchGap
from coverage_agent.report.run_report import serialize_run_report, load_run_report


def _make_report() -> RunReport:
    gap = CoverageGap(
        file_path="pkg/mod.py",
        target_symbol="process",
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=[10, 11, 12],
        kind="branch",
        origin="full",
        gap_id="pkg/mod.py:10->12",
    )
    gr = GapResult(gap=gap, skipped=False, loops_taken=1, accepted=True,
                   test_code="def test_x(): pass")
    return RunReport(
        scope="full",
        model="gemini/gemini-2.5-flash",
        gaps_found=3,
        gaps_accepted=1,
        tests_accepted=1,
        total_cost_usd=0.002,
        gap_results=[gr],
    )


def test_serialize_returns_valid_json():
    report = _make_report()
    json_str = serialize_run_report(report)
    data = json.loads(json_str)
    assert data["scope"] == "full"
    assert data["tests_accepted"] == 1
    assert len(data["gap_results"]) == 1


def test_serialize_writes_to_file(tmp_path):
    report = _make_report()
    dest = tmp_path / "report.json"
    json_str = serialize_run_report(report, str(dest))
    assert dest.exists()
    assert json.loads(dest.read_text()) == json.loads(json_str)


def test_serialize_creates_parent_dirs(tmp_path):
    report = _make_report()
    dest = tmp_path / "nested" / "dir" / "report.json"
    serialize_run_report(report, str(dest))
    assert dest.exists()


def test_load_round_trip(tmp_path):
    report = _make_report()
    dest = tmp_path / "report.json"
    serialize_run_report(report, str(dest))
    loaded = load_run_report(str(dest))
    assert loaded.scope == report.scope
    assert loaded.tests_accepted == report.tests_accepted
    assert len(loaded.gap_results) == 1
    assert loaded.gap_results[0].accepted is True
