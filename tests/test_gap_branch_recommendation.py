from coverage_agent.contracts.schemas import BranchGap, ContextPayload, CoverageGap, ExecutionResult
from coverage_agent.recommendations import gap_branch_recommendation


def _gap():
    return CoverageGap(
        gap_id="m.py:1->2",
        file_path="pkg/mod.py",
        target_symbol="f",
        branch=BranchGap(from_line=10, to_line=11),
        surrounding_lines=[1, 2],
        priority_score=0.5,
    )


def test_empty_when_no_phase2():
    assert gap_branch_recommendation(_gap(), None, None) == ""


def test_empty_when_branch_hit():
    p2 = ExecutionResult(
        execution_success=True,
        target_branch_hit=True,
        coverage_delta=1.0,
    )
    assert gap_branch_recommendation(_gap(), None, p2) == ""


def test_includes_hint_when_present():
    p2 = ExecutionResult(
        execution_success=True,
        target_branch_hit=False,
        coverage_delta=0.0,
    )
    ctx = ContextPayload(
        primary_code="def f(): pass",
        dependencies_code={},
        graph_depth_used=0,
        tokens_used=0,
        branch_condition_hint="x > 0",
    )
    out = gap_branch_recommendation(_gap(), ctx, p2)
    assert "10->11" in out
    assert "x > 0" in out


def test_no_hint_still_actionable():
    p2 = ExecutionResult(
        execution_success=True,
        target_branch_hit=False,
        coverage_delta=0.0,
    )
    out = gap_branch_recommendation(_gap(), None, p2)
    assert "executed_branches".lower() in out.lower()
