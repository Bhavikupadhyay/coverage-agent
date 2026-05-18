"""
GapPrioritizer — picks which uncovered branches to test first.

Phase D rewrite: the deterministic heuristic is the default. The previous
LLM-ranking path is preserved but gated by Credentials.prioritize_with_llm
because empirically it picked nearly the same order as line-count sorting
on benchmark runs while burning one 70B call per run.

The deterministic scorer looks at signals visible on `CoverageGap` alone:

- File path                 — __init__.py / conftest.py / setup-ish locations get penalized
- Target symbol name        — pure side-effect-suggesting names (log_*, warn_*) get penalized
- Gap size                  — preferred range is 4-30 surrounding lines (a real function with logic)
- Branch direction          — branches that jump backward (to_line < from_line) are loops; penalize slightly
- Already-scored heuristics — surface signals to the orchestrator so the UI can show 'why this gap'

Output is always sorted highest-priority first. Score is in [0.0, 1.0].
"""
from __future__ import annotations

import json
import logging
import re

import litellm

from coverage_agent.credentials import Credentials
from coverage_agent.contracts.schemas import CoverageGap

logger = logging.getLogger(__name__)


# Symbol-name patterns that suggest pure-side-effect or setup logic — these
# functions are typically untestable in isolation without exotic mocking.
_NOISY_NAME_PATTERNS = (
    re.compile(r"^_?(log|warn|debug|trace|print|emit)_"),  # noisy logging-style names
    re.compile(r"^_?on_"),                                  # event-handler-style names
    re.compile(r"^_?(setup|teardown|configure)"),           # setup code
    re.compile(r"^__\w+__$"),                               # dunder methods (often metaclass stuff)
)

# File-path patterns that suggest non-test-worthy code. Pure conservatism — we
# don't reject these, just push them down the queue.
_NOISY_FILE_PATTERNS = (
    re.compile(r"(^|/)__init__\.py$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"(^|/)_version\.py$"),
    re.compile(r"(^|/)setup\.py$"),
)


def _score_gap(gap: CoverageGap) -> float:
    """Deterministic testability score in [0.0, 1.0]. Higher is better."""
    score = 0.5  # neutral baseline

    # --- file path signals ---
    if any(p.search(gap.file_path) for p in _NOISY_FILE_PATTERNS):
        score -= 0.2

    # --- symbol name signals ---
    if any(p.match(gap.target_symbol) for p in _NOISY_NAME_PATTERNS):
        score -= 0.25
    # private functions are still testable but slightly less of a target
    elif gap.target_symbol.startswith("_") and not gap.target_symbol.startswith("__"):
        score -= 0.05

    # --- gap size signals ---
    n_lines = len(gap.surrounding_lines)
    if 4 <= n_lines <= 30:
        score += 0.25  # the sweet spot — real logic, not too tangled
    elif n_lines <= 3:
        score -= 0.10  # tiny gap, probably trivial
    elif 30 < n_lines <= 80:
        score -= 0.05  # large function — testable but expensive to context
    else:  # > 80
        score -= 0.20  # likely a god-function; rarely a fair single-gap target

    # --- branch direction ---
    # to_line < from_line means the branch jumps backwards — loop continuation
    # or `else` returning to earlier code. These are typically harder to target
    # reliably with a single test input.
    if gap.branch.to_line < gap.branch.from_line:
        score -= 0.05

    return max(0.0, min(1.0, score))


class GapPrioritizer:
    """Orders uncovered gaps by a testability heuristic. LLM-ranking opt-in."""

    def __init__(self, credentials: Credentials) -> None:
        self.creds = credentials

    def run(self, gaps: list[CoverageGap]) -> list[CoverageGap]:
        if not gaps:
            logger.info("No coverage gaps found — nothing to prioritize")
            return []

        if self.creds.is_offline:
            logger.info("[OFFLINE] GapPrioritizer — assigning index-based priority scores")
            return [
                gap.model_copy(update={"priority_score": round(1.0 - i * 0.1, 2)})
                for i, gap in enumerate(gaps)
            ]

        # Default: deterministic heuristic. No LLM call. Reproducible.
        if not getattr(self.creds, "prioritize_with_llm", False):
            scored = [
                gap.model_copy(update={"priority_score": _score_gap(gap)})
                for gap in gaps
            ]
            ranked = sorted(scored, key=lambda g: g.priority_score, reverse=True)
            logger.info(
                "GapPrioritizer (heuristic): top score=%.2f, bottom score=%.2f, total=%d",
                ranked[0].priority_score, ranked[-1].priority_score, len(ranked),
            )
            return ranked

        # Opt-in: LLM ranking. Kept for completeness; in practice the heuristic
        # picks nearly the same order on benchmark sets while saving a 70B call.
        top_gaps = gaps[:20]
        scored = self._score_with_llm(top_gaps)
        return sorted(scored, key=lambda g: g.priority_score, reverse=True)

    def _score_with_llm(self, gaps: list[CoverageGap]) -> list[CoverageGap]:
        descriptions = "\n".join(
            f"{i}. gap_id={g.gap_id} symbol={g.target_symbol} "
            f"branch={g.branch.from_line}->{g.branch.to_line} "
            f"lines={len(g.surrounding_lines)}"
            for i, g in enumerate(gaps)
        )
        prompt = (
            "Score each Python coverage gap from 0.0 to 1.0 based on:\n"
            "- Logic complexity: is there real branching logic worth testing?\n"
            "- Testability: can it be tested without exotic mocking?\n"
            "- Value: is this a core path or just a trivial edge case?\n\n"
            f"Gaps:\n{descriptions}\n\n"
            "Respond with a JSON array of floats in the same order. Example: [0.9, 0.4, 0.7]\n"
            "Return only the JSON array, nothing else."
        )
        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **self.creds.litellm_kwargs(),
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            scores = json.loads(content)
            result = []
            for i, gap in enumerate(gaps):
                score = float(scores[i]) if i < len(scores) else 0.5
                result.append(gap.model_copy(update={"priority_score": max(0.0, min(1.0, score))}))
            return result
        except Exception as exc:
            logger.warning("LLM scoring failed (%s) — falling back to deterministic heuristic", exc)
            return [
                gap.model_copy(update={"priority_score": _score_gap(gap)})
                for gap in gaps
            ]
