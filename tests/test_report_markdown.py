"""Tests for coverage_agent/report/markdown.py — pure rendering functions."""
from __future__ import annotations

from coverage_agent.contracts import (
    BranchGap,
    CoverageGap,
    ExecutionResult,
    GapResult,
    RunReport,
)
from coverage_agent.report.markdown import (
    COMMENT_MARKER,
    render_comment,
    _render_summary,
    _render_results_table,
    _render_arc_miss_guidance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap(gap_id: str = "pkg/mod.py:10->12", kind: str = "branch") -> CoverageGap:
    return CoverageGap(
        file_path="pkg/mod.py",
        target_symbol="process",
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=[10, 11, 12],
        kind=kind,  # type: ignore[arg-type]
        origin="full",
        gap_id=gap_id,
    )


def _accepted_result(gap: CoverageGap, targets_hit: int = 1, targets_total: int = 1) -> GapResult:
    return GapResult(
        gap=gap,
        skipped=False,
        loops_taken=1,
        accepted=True,
        test_code="def test_process_branch():\n    assert process(None) is None\n",
        execution=ExecutionResult(
            execution_success=True,
            target_branch_hit=True,
            targets_hit=targets_hit,
            targets_total=targets_total,
        ),
    )


def _skipped_result(gap: CoverageGap, reason: str = "") -> GapResult:
    return GapResult(
        gap=gap,
        skipped=True,
        loops_taken=0,
        accepted=False,
        skip_reason=reason,
    )


def _missed_arc_result(gap: CoverageGap) -> GapResult:
    """pytest passed but target arc was not hit."""
    return GapResult(
        gap=gap,
        skipped=False,
        loops_taken=2,
        accepted=False,
        execution=ExecutionResult(
            execution_success=True,
            target_branch_hit=False,
            targets_hit=0,
            targets_total=1,
        ),
    )


# ---------------------------------------------------------------------------
# Tests — marker
# ---------------------------------------------------------------------------

def test_render_comment_starts_with_marker():
    report = RunReport(scope="full", model="test/model", gaps_found=0, gaps_accepted=0)
    body = render_comment(report)
    assert body.startswith(COMMENT_MARKER)


def test_marker_literal_value():
    assert COMMENT_MARKER == "<!-- coverage-agent:report -->"


# ---------------------------------------------------------------------------
# Tests — summary line
# ---------------------------------------------------------------------------

def test_summary_includes_gaps_found():
    report = RunReport(scope="diff", model="gemini/gemini-2.5-flash", gaps_found=5, tests_accepted=3)
    summary = _render_summary(report)
    assert "5" in summary
    assert "3" in summary
    assert "diff" in summary
    assert "gemini/gemini-2.5-flash" in summary


# ---------------------------------------------------------------------------
# Tests — results table
# ---------------------------------------------------------------------------

def test_results_table_accepted_row():
    gap = _make_gap()
    gr = _accepted_result(gap, targets_hit=3, targets_total=4)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=1, tests_accepted=1, gap_results=[gr])
    table = _render_results_table(report)
    assert "accepted" in table
    assert "3/4" in table
    assert gap.gap_id in table


def test_results_table_skipped_row():
    gap = _make_gap()
    gr = _skipped_result(gap, reason="too complex")
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=0, gap_results=[gr])
    table = _render_results_table(report)
    assert "skipped" in table
    assert "too complex" in table


def test_results_table_missed_arc_row():
    gap = _make_gap()
    gr = _missed_arc_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=0, gap_results=[gr])
    table = _render_results_table(report)
    assert "not accepted" in table
    assert "0/1" in table


# ---------------------------------------------------------------------------
# Tests — details block
# ---------------------------------------------------------------------------

def test_details_block_present_for_accepted():
    gap = _make_gap()
    gr = _accepted_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=1, tests_accepted=1, gap_results=[gr])
    body = render_comment(report)
    assert "<details>" in body
    assert "</details>" in body
    # The test code must appear somewhere in a diff block.
    assert "test_process_branch" in body


def test_details_block_absent_when_no_accepted():
    gap = _make_gap()
    gr = _skipped_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=0, gap_results=[gr])
    body = render_comment(report)
    assert "<details>" not in body


# ---------------------------------------------------------------------------
# Tests — arc miss guidance
# ---------------------------------------------------------------------------

def test_arc_miss_guidance_included():
    gap = _make_gap()
    gr = _missed_arc_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=0, gap_results=[gr])
    guidance = _render_arc_miss_guidance(report)
    assert gap.gap_id in guidance
    assert "10" in guidance  # from_line
    assert "12" in guidance  # to_line


def test_arc_miss_guidance_empty_for_accepted_only():
    gap = _make_gap()
    gr = _accepted_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=1, gap_results=[gr])
    guidance = _render_arc_miss_guidance(report)
    assert guidance == ""


def test_arc_miss_guidance_empty_for_skipped():
    gap = _make_gap()
    gr = _skipped_result(gap)
    report = RunReport(scope="full", model="m", gaps_found=1, gaps_accepted=0, gap_results=[gr])
    guidance = _render_arc_miss_guidance(report)
    assert guidance == ""


# ---------------------------------------------------------------------------
# Tests — full render end-to-end
# ---------------------------------------------------------------------------

def test_full_render_mixed_results():
    gap1 = _make_gap("pkg/a.py:5->8")
    gap2 = _make_gap("pkg/b.py:20->25")
    gap3 = _make_gap("pkg/c.py:30->35")
    gr1 = _accepted_result(gap1, 1, 1)
    gr2 = _skipped_result(gap2)
    gr3 = _missed_arc_result(gap3)
    report = RunReport(
        scope="diff",
        model="gemini/gemini-2.5-flash",
        gaps_found=3,
        gaps_accepted=1,
        tests_accepted=1,
        gap_results=[gr1, gr2, gr3],
    )
    body = render_comment(report)
    # Marker always first.
    assert body.startswith(COMMENT_MARKER)
    # All gap ids present.
    assert gap1.gap_id in body
    assert gap2.gap_id in body
    assert gap3.gap_id in body
    # Details block for accepted.
    assert "<details>" in body
    # Arc miss guidance for gap3.
    assert gap3.gap_id in body.split("<details>")[0] or gap3.gap_id in body
