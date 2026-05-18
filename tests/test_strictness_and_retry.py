"""Commit gate + pipeline routing contracts.

- should_commit requires execution_success AND target_branch_hit for all presets.
- eval_strictness only changes max_retry_loops (loose = 1, else 3).
- Post-execution routing uses should_commit; missing branch routes to TEST_WRITER.
"""
from __future__ import annotations

from coverage_agent.contracts.schemas import ExecutionResult
from coverage_agent.credentials import Credentials


def _exec(*, success: bool, branch_hit: bool, stderr: str = "") -> ExecutionResult:
    return ExecutionResult(
        execution_success=success,
        target_branch_hit=branch_hit,
        coverage_delta=0.0,
        stderr_trace=stderr,
        flaky=False,
    )


def test_should_commit_requires_branch_for_strict():
    creds = Credentials(mode="offline", eval_strictness="strict")
    assert creds.should_commit(_exec(success=True, branch_hit=True)) is True
    assert creds.should_commit(_exec(success=True, branch_hit=False)) is False
    assert creds.should_commit(_exec(success=False, branch_hit=True)) is False


def test_should_commit_requires_branch_for_balanced():
    creds = Credentials(mode="offline", eval_strictness="balanced")
    assert creds.should_commit(_exec(success=True, branch_hit=True)) is True
    assert creds.should_commit(_exec(success=True, branch_hit=False)) is False


def test_should_commit_requires_branch_for_loose():
    creds = Credentials(mode="offline", eval_strictness="loose")
    assert creds.should_commit(_exec(success=True, branch_hit=True)) is True
    assert creds.should_commit(_exec(success=True, branch_hit=False)) is False


def test_should_commit_handles_none_exec_result():
    creds = Credentials(mode="offline", eval_strictness="balanced")
    assert creds.should_commit(None) is False


def test_max_retry_loops_per_strictness():
    assert Credentials(mode="offline", eval_strictness="strict").max_retry_loops() == 3
    assert Credentials(mode="offline", eval_strictness="balanced").max_retry_loops() == 3
    assert Credentials(mode="offline", eval_strictness="loose").max_retry_loops() == 1


def test_commit_requires_branch_hit_always_true():
    for s in ("strict", "balanced", "loose"):
        assert Credentials(mode="offline", eval_strictness=s).commit_requires_branch_hit() is True


def test_pipeline_graph_has_execution_runner_to_test_writer_edge():
    from unittest.mock import MagicMock

    from coverage_agent.pipeline import build_pipeline

    sandbox = MagicMock()
    creds = Credentials(mode="offline", eval_strictness="balanced")
    compiled = build_pipeline(sandbox, creds)

    graph = compiled.get_graph()
    edge_map: dict[str, set[str]] = {}
    for edge in graph.edges:
        edge_map.setdefault(edge.source, set()).add(edge.target)

    exec_targets = edge_map.get("execution_runner", set())
    assert "test_writer" in exec_targets
    assert "commit" in exec_targets
    assert "skip" in exec_targets


def test_pipeline_routes_miss_branch_to_test_writer_all_presets():
    from coverage_agent.pipeline import _make_route_after_execution

    for preset in ("strict", "balanced", "loose"):
        creds = Credentials(mode="offline", eval_strictness=preset)
        router = _make_route_after_execution(creds)
        state = {
            "exec_result": _exec(success=True, branch_hit=False),
            "loop_count": 0,
        }
        assert router(state) == "TEST_WRITER", preset


def test_pipeline_routes_commit_when_branch_proven():
    from coverage_agent.pipeline import _make_route_after_execution

    creds = Credentials(mode="offline", eval_strictness="balanced")
    router = _make_route_after_execution(creds)
    state_ok = {
        "exec_result": _exec(success=True, branch_hit=True),
        "loop_count": 2,
    }
    assert router(state_ok) == "COMMIT"


def test_pipeline_skips_when_retry_budget_exhausted():
    from coverage_agent.pipeline import _make_route_after_execution

    creds = Credentials(mode="offline", eval_strictness="loose")
    router = _make_route_after_execution(creds)

    state = {
        "exec_result": _exec(success=False, branch_hit=False, stderr="boom"),
        "loop_count": 1,
    }
    assert router(state) == "SKIP"
