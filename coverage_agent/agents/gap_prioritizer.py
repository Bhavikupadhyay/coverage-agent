import json
import logging
import os
from pathlib import Path

import litellm

from coverage_agent.contracts.schemas import CoverageGap
from coverage_agent.context.coverage_parser import parse_coverage

logger = logging.getLogger(__name__)

_MODEL = "gemini/gemini-2.5-flash"


def _is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


class GapPrioritizer:
    def run(self, coverage_json: dict) -> list[CoverageGap]:
        gaps = parse_coverage(coverage_json)
        if not gaps:
            logger.info("No coverage gaps found — nothing to prioritize")
            return []

        if _is_dry_run():
            logger.info("[DRY_RUN] GapPrioritizer — assigning mock priority scores")
            return [
                gap.model_copy(update={"priority_score": round(1.0 - i * 0.1, 2)})
                for i, gap in enumerate(gaps)
            ]

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
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
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
            logger.warning("LLM scoring failed (%s) — falling back to line-count ordering", exc)
            return [
                gap.model_copy(update={"priority_score": min(1.0, len(gap.surrounding_lines) / 50.0)})
                for gap in gaps
            ]
