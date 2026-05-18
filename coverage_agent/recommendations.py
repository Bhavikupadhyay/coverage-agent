from coverage_agent.contracts.schemas import ContextPayload, CoverageGap, ExecutionResult, GapResult


def gap_branch_recommendation(
    gap: CoverageGap,
    ctx: ContextPayload | None,
    p2: ExecutionResult | None,
) -> str:
    """Actionable text when pytest passed but the target branch is still missing from coverage."""
    if p2 is None or not p2.execution_success or p2.target_branch_hit:
        return ""
    hint = ctx.branch_condition_hint if ctx else None
    core = (
        f"Pytest exited 0, but coverage did not record branch "
        f"{gap.branch.from_line}->{gap.branch.to_line} in `{gap.file_path}` "
        "in executed_branches. A commit requires that edge."
    )
    if hint:
        return core + f" Condition hint: `{hint}`. Adjust inputs or mocks so this branch runs."
    return core + " Re-read the implementation and change inputs until the branch executes."


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
            "No commits with branch proof. Check sandbox stderr per gap — "
            "pytest must pass and coverage must list the target branch in executed_branches."
        )
    elif hit_rate < 0.5:
        recs.append(
            f"Branch proof rate is {hit_rate_str}. Try a stronger reasoning model, "
            "or inspect per-gap 'What to try next' — inputs often miss the branch condition."
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
            f"High average loop count ({avg_loops:.1f}). Review per-gap recommendations "
            "and stderr — common causes are wrong mocks, wrong inputs for the branch "
            "condition, or coverage not tracing the package under test."
        )

    return recs
