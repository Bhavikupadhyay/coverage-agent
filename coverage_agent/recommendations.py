from coverage_agent.contracts.schemas import GapResult


def generate(scorecard: dict, results: list[GapResult]) -> list[str]:
    recs = []
    hit_rate_str = scorecard.get("branch_hit_rate", "0%")
    try:
        hit_rate = float(hit_rate_str.rstrip("%")) / 100
    except ValueError:
        hit_rate = 0.0

    skipped = scorecard.get("skipped", 0)
    committed = scorecard.get("tests_committed", 0)
    try:
        avg_loops = float(scorecard.get("avg_loops", "0"))
    except ValueError:
        avg_loops = 0.0

    if hit_rate == 0.0:
        recs.append(
            "No branches were hit. Verify the repo has a working test suite — "
            "if pytest fails during baseline, coverage data will be empty."
        )
    elif hit_rate < 0.5:
        recs.append(
            f"Branch hit rate is {hit_rate_str}. Consider switching to gemini-2.5-pro "
            "for stronger reasoning, or raising the loop limit to 5 in pipeline.py."
        )

    if skipped > 0:
        skipped_ids = [r.gap.gap_id for r in results if r.skipped]
        recs.append(
            f"{skipped} gap(s) exhausted all retries and were skipped: "
            + ", ".join(skipped_ids)
            + ". These likely need manual test writing."
        )

    if committed > 0:
        recs.append(
            f"{committed} test(s) written — review before merging: tests/test_auto_*.py"
        )

    if avg_loops > 2.0:
        recs.append(
            f"High average loop count ({avg_loops:.1f}). Check Braintrust dataset "
            "for patterns in eval_agent critique — likely a mock completeness or "
            "assertion quality issue."
        )

    return recs
