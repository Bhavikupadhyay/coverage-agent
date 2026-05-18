"""
ResultSummarizer — final-step LLM agent that turns the run's scorecard + per-gap
results into two narratives:

- `pr_description`: short, markdown, ~120 words, suitable for dropping into a
  pull-request body. Bulleted list of new tests, branches covered, regression
  status.
- `full_summary`: longer paragraph form, ~300 words, covers what the agents
  decided and why, what was skipped, and any caveats.

Runs once per pipeline run. ~1 LLM call total per run, so its rate-limit impact
is negligible (a 5-gap run today does ~25 LLM calls; this adds one).

Offline mode returns a templated fixture so the UI has something to render.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import litellm

from coverage_agent.contracts.schemas import GapResult, RegressionResult, RunSummary
from coverage_agent.credentials import Credentials

logger = logging.getLogger(__name__)


def _gap_one_liner(r: GapResult) -> str:
    branch = f"{r.gap.branch.from_line}->{r.gap.branch.to_line}"
    if r.final_test_committed:
        return f"COMMITTED  {r.gap.file_path}:{branch}  ({r.gap.target_symbol})"
    reason = r.skip_reason[:120] if r.skip_reason else "skipped"
    return f"SKIPPED    {r.gap.file_path}:{branch}  ({r.gap.target_symbol}) — {reason}"


def _build_offline_summary(
    results: list[GapResult],
    scorecard: dict,
    regression: Optional[RegressionResult],
) -> RunSummary:
    committed = [r for r in results if r.final_test_committed]
    skipped = [r for r in results if not r.final_test_committed]

    bullets = "\n".join(
        f"- `{r.gap.file_path}` · `{r.gap.target_symbol}` — covered branch {r.gap.branch.from_line}→{r.gap.branch.to_line}"
        for r in committed[:5]
    ) or "- _(no tests committed)_"

    regression_line = ""
    if regression and not regression.skipped:
        regression_line = (
            "\n\n**Regression check:** "
            + ("⚠ regression detected — investigate before merging." if regression.regression_detected else "clean.")
        )

    pr_description = (
        f"### CoverageAgent — new tests\n\n"
        f"{len(committed)} new pytest test(s) committed, "
        f"{len(skipped)} gap(s) skipped after eval.\n\n"
        f"**New tests:**\n{bullets}"
        f"{regression_line}\n\n"
        f"_Coverage Δ: {scorecard.get('avg_coverage_delta', '+0.00%')} · "
        f"Branch hit rate: {scorecard.get('branch_hit_rate', '0.0%')}._"
    )

    regression_prose = ""
    if regression and not regression.skipped:
        regression_prose = (
            " RegressionGuard then re-ran the full pytest suite with all committed tests in place "
            + ("and detected new failures — see the regression panel for details. "
               if regression.regression_detected
               else "and confirmed no previously-passing test was broken. ")
        )

    full_summary = (
        f"CoverageAgent targeted {scorecard.get('gaps_targeted', len(results))} uncovered branch(es) "
        f"in this run. The eval gate accepted {len(committed)} draft(s) for sandbox execution and "
        f"committed them after they exercised the target branch. {len(skipped)} draft(s) were "
        f"rejected by the eval loop or missed the branch during execution.{regression_prose}\n\n"
        f"The per-gap loop (Prioritizer → Context → Writer → Eval → Runner) ran with the configured "
        f"strictness setting and produced an average of {scorecard.get('avg_loops', '—')} retry loops "
        f"per gap. Overall branch coverage moved by {scorecard.get('avg_coverage_delta', '+0.00%')}."
    )

    return RunSummary(pr_description=pr_description, full_summary=full_summary)


class ResultSummarizer:
    """Final-step agent. Single LLM call per run. Free even on conservative quotas."""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(
        self,
        results: list[GapResult],
        scorecard: dict,
        regression: Optional[RegressionResult] = None,
    ) -> RunSummary:
        if self.creds.is_offline:
            logger.info("[OFFLINE] ResultSummarizer — returning templated fixture")
            return _build_offline_summary(results, scorecard, regression)

        gap_lines = "\n".join(_gap_one_liner(r) for r in results[:25])
        regression_block = ""
        if regression and not regression.skipped:
            regression_block = (
                f"\nRegressionGuard outcome:\n"
                f"  Baseline: {regression.baseline_passed} passing / {regression.baseline_failed} failing\n"
                f"  Post-commit: {regression.post_passed} passing / {regression.post_failed} failing\n"
                f"  Regression detected: {regression.regression_detected}\n"
                f"  Summary: {regression.summary}\n"
            )

        prompt = (
            "You are summarizing a CoverageAgent run for a developer who's about to open a pull request.\n\n"
            f"Scorecard:\n  {json.dumps({k: v for k, v in scorecard.items() if k != 'repo'}, indent=2)}\n\n"
            f"Per-gap outcomes (one line each):\n{gap_lines}\n"
            f"{regression_block}\n"
            "Return STRICT JSON matching this schema (no markdown fences, no commentary):\n"
            '{\n'
            '  "pr_description": "<markdown, ~120 words, suitable to paste into a PR body. '
            'Lead with what was committed; list new tests as bullets with file/symbol; '
            'call out the regression check outcome if relevant.>",\n'
            '  "full_summary": "<plain prose, ~300 words. Explain what the eval gate accepted, '
            'what was skipped and why, average loop count, coverage delta, and any caveats.>"\n'
            '}'
        )

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **self.creds.litellm_kwargs(),
            )
            raw = response.choices[0].message.content
            payload = json.loads(raw)
            return RunSummary(
                pr_description=payload.get("pr_description", "").strip()
                    or "_(model returned no PR description)_",
                full_summary=payload.get("full_summary", "").strip()
                    or "_(model returned no summary)_",
            )
        except Exception as exc:
            logger.warning("ResultSummarizer failed (%s) — falling back to templated summary", exc)
            return _build_offline_summary(results, scorecard, regression)
