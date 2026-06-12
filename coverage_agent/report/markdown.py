"""
Render a RunReport into a GitHub PR comment body.

Pure functions: RunReport in, string out. No IO, no network.
"""
from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from coverage_agent.contracts import GapResult, RunReport

# Idempotency key — the comment upsert logic searches for this marker.
COMMENT_MARKER = "<!-- coverage-agent:report -->"


def render_comment(report: RunReport) -> str:
    """Render a full PR comment body from a RunReport.

    Returns a string starting with COMMENT_MARKER so the GitHub delivery layer
    can identify and update the comment idempotently.
    """
    parts: list[str] = [COMMENT_MARKER]
    parts.append(_render_summary(report))
    parts.append("")
    parts.append(_render_results_table(report))

    accepted = [r for r in report.gap_results if r.accepted and r.test_code]
    if accepted:
        parts.append("")
        parts.append(_render_details_block(accepted))

    missed_guidance = _render_arc_miss_guidance(report)
    if missed_guidance:
        parts.append("")
        parts.append(missed_guidance)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------

def _render_summary(report: RunReport) -> str:
    scope_label = report.scope
    model_label = report.model or "unknown"
    return (
        f"**coverage-agent** found **{report.gaps_found}** gap(s), "
        f"accepted **{report.tests_accepted}** test(s) — "
        f"scope: `{scope_label}` · model: `{model_label}`"
    )


def _render_results_table(report: RunReport) -> str:
    header = "| Gap | Kind | Result | Targets hit |"
    sep =    "| --- | ---- | ------ | ----------- |"
    rows = [header, sep]

    for gr in report.gap_results:
        kind = gr.gap.kind
        gap_id = gr.gap.gap_id

        if gr.skipped:
            result_cell = f"skipped — {gr.skip_reason}" if gr.skip_reason else "skipped"
            targets_cell = "—"
        elif gr.accepted:
            result_cell = "accepted"
            exec_r = gr.execution
            if exec_r is not None and exec_r.targets_total > 0:
                targets_cell = f"{exec_r.targets_hit}/{exec_r.targets_total}"
            else:
                targets_cell = "—"
        else:
            result_cell = "not accepted"
            exec_r = gr.execution
            if exec_r is not None and exec_r.targets_total > 0:
                targets_cell = f"{exec_r.targets_hit}/{exec_r.targets_total}"
            else:
                targets_cell = "—"

        rows.append(f"| `{gap_id}` | {kind} | {result_cell} | {targets_cell} |")

    return "\n".join(rows)


def _render_details_block(accepted: list[GapResult]) -> str:
    """Accepted test files as unified diffs inside a <details> block."""
    inner_parts: list[str] = []
    for gr in accepted:
        filename = _safe_filename(gr)
        diff_lines = list(difflib.unified_diff(
            [],
            (gr.test_code or "").splitlines(keepends=True),
            fromfile="/dev/null",
            tofile=f"b/{filename}",
        ))
        diff_text = "".join(diff_lines)
        inner_parts.append(f"**`{filename}`**\n\n```diff\n{diff_text}```")

    summary_label = f"{len(accepted)} accepted test file{'s' if len(accepted) != 1 else ''}"
    inner = "\n\n".join(inner_parts)
    return f"<details>\n<summary>{summary_label}</summary>\n\n{inner}\n\n</details>"


def _render_arc_miss_guidance(report: RunReport) -> str:
    """One guidance line per gap where pytest passed but targets were not fully hit."""
    lines: list[str] = []
    for gr in report.gap_results:
        if gr.skipped or gr.accepted:
            continue
        exec_r = gr.execution
        if exec_r is None:
            continue
        if exec_r.execution_success and not exec_r.target_branch_hit:
            gap = gr.gap
            lines.append(
                f"- `{gap.gap_id}`: pytest passed but the target arc "
                f"(`{gap.branch.from_line}` → `{gap.branch.to_line}`) was not reached — "
                "the test inputs likely did not trigger the untaken condition."
            )
    if not lines:
        return ""
    return "\n".join(lines)


def _safe_filename(gr: GapResult) -> str:
    """Derive a safe filename for the diff header from the gap id."""
    import re
    slug = re.sub(r"[^A-Za-z0-9_/.]", "_", gr.gap.gap_id)
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", gr.gap.target_symbol or "gap").strip("_") or "gap"
    return f"tests/generated/test_coverageagent_{symbol}.py"
