"""Tests for the cluster-aware pipeline plumbing.

Covers:
- run_pipeline_cluster single-gap regression (behaviour identical to run_pipeline)
- Missed-arc critique content when execution runs but doesn't hit all arcs
- run_pipeline_cluster partial acceptance: some arcs hit → mixed GapResults
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coverage_agent.contracts import (
    BranchGap,
    ContextPayload,
    CoverageGap,
    DraftTest,
    ExecutionResult,
    ValidationResult,
)
from coverage_agent.credentials import Credentials
from coverage_agent.config import AgentConfig
from coverage_agent.engine.executor import _cluster_arc_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREDS = Credentials(llm_api_key="mock-key", llm_model="groq/llama-3.3-70b-versatile")
_CFG = AgentConfig(max_tool_calls=0, flaky_runs=1, max_retries=1, test_timeout=10)


def _gap(symbol: str = "fn", from_l: int = 10, to_l: int = 12) -> CoverageGap:
    return CoverageGap(
        file_path="pkg/stats.py",
        target_symbol=symbol,
        branch=BranchGap(from_line=from_l, to_line=to_l),
        surrounding_lines=list(range(from_l, to_l + 3)),
        kind="branch",
        origin="full",
        gap_id=f"pkg/stats.py:{from_l}->{to_l}",
    )


def _fake_completion(test_code: str = "def test_x(): pass\n"):
    def _mock(*args, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = f"```python\n{test_code}\n```"
        resp.choices[0].message.tool_calls = None
        resp.cost = 0.0
        resp.usage.total_tokens = 50
        return resp
    return _mock


def _mock_exec_result(success: bool, hit: bool) -> ExecutionResult:
    return ExecutionResult(
        execution_success=success,
        target_branch_hit=hit,
        targets_hit=1 if hit else 0,
        targets_total=1,
    )


# ---------------------------------------------------------------------------
# Single-gap regression: run_pipeline_cluster with 1 gap == run_pipeline
# ---------------------------------------------------------------------------

def test_run_pipeline_cluster_single_gap_accepted(tmp_path, monkeypatch):
    """Single-gap cluster accepted → one GapResult with accepted=True."""
    monkeypatch.chdir(tmp_path)
    gap = _gap()

    with patch("litellm.completion", side_effect=_fake_completion()), \
         patch("coverage_agent.engine.executor._run_once",
               return_value=_mock_exec_result(True, True)):

        from coverage_agent.engine.graph import run_pipeline_cluster
        results, _ = run_pipeline_cluster(
            cluster=[gap],
            credentials=_CREDS,
            config=_CFG,
            repo_path=str(tmp_path),
        )

    assert len(results) == 1
    assert results[0].accepted is True


def test_run_pipeline_cluster_single_gap_skipped(tmp_path, monkeypatch):
    """Single-gap cluster that doesn't hit target → accepted=False."""
    monkeypatch.chdir(tmp_path)
    gap = _gap()

    with patch("litellm.completion", side_effect=_fake_completion()), \
         patch("coverage_agent.engine.executor._run_once",
               return_value=_mock_exec_result(True, False)):

        from coverage_agent.engine.graph import run_pipeline_cluster
        results, _ = run_pipeline_cluster(
            cluster=[gap],
            credentials=_CREDS,
            config=_CFG,
            repo_path=str(tmp_path),
        )

    assert len(results) == 1
    assert results[0].accepted is False


# ---------------------------------------------------------------------------
# Multi-gap partial acceptance
# ---------------------------------------------------------------------------

def test_run_pipeline_cluster_partial_acceptance(tmp_path, monkeypatch):
    """Two-gap cluster where one arc is hit and one is missed."""
    monkeypatch.chdir(tmp_path)
    g1 = _gap(symbol="letter_grade", from_l=35, to_l=37)
    g2 = _gap(symbol="letter_grade", from_l=37, to_l=38)
    cluster = [g1, g2]

    base_exec = _mock_exec_result(True, True)   # any-hit → keep test
    # Inject per-arc data: g1 hit, g2 missed.
    _cluster_arc_store[id(base_exec)] = {(35, 37): True, (37, 38): False}

    with patch("litellm.completion", side_effect=_fake_completion()), \
         patch("coverage_agent.engine.executor._run_once", return_value=base_exec):

        from coverage_agent.engine.graph import run_pipeline_cluster
        results, _ = run_pipeline_cluster(
            cluster=cluster,
            credentials=_CREDS,
            config=_CFG,
            repo_path=str(tmp_path),
        )

    assert len(results) == 2
    hit_results = [r for r in results if r.accepted]
    miss_results = [r for r in results if not r.accepted]
    assert len(hit_results) == 1
    assert len(miss_results) == 1
    assert hit_results[0].gap.branch.from_line == 35
    assert miss_results[0].gap.branch.from_line == 37
    assert "37->38" in miss_results[0].skip_reason


# ---------------------------------------------------------------------------
# Missed-arc critique: when execution passes but hits NO arc in the cluster
# ---------------------------------------------------------------------------

def test_execution_runner_critique_names_missed_arcs(tmp_path, monkeypatch):
    """When execution runs but hits none of the cluster's arcs, the retry
    critique lists exactly the missed arcs by line number."""
    monkeypatch.chdir(tmp_path)

    g1 = _gap(symbol="letter_grade", from_l=35, to_l=37)
    g2 = _gap(symbol="letter_grade", from_l=37, to_l=38)
    cluster = [g1, g2]

    # First execution: pytest passes but neither arc was hit.
    exec_no_hit = _mock_exec_result(True, False)

    call_count = [0]
    captured_prompts: list[list] = []

    def _capture_completion(*args, **kwargs):
        call_count[0] += 1
        captured_prompts.append(list(kwargs.get("messages", [])))
        resp = MagicMock()
        resp.choices[0].message.content = "```python\ndef test_x(): pass\n```"
        resp.choices[0].message.tool_calls = None
        resp.cost = 0.0
        resp.usage.total_tokens = 50
        return resp

    run_count = [0]
    def _fake_run_once(*args, **kwargs):
        run_count[0] += 1
        # Always return no-hit so the pipeline goes through the retry branch.
        return exec_no_hit

    with patch("litellm.completion", side_effect=_capture_completion), \
         patch("coverage_agent.engine.executor._run_once", side_effect=_fake_run_once):

        from coverage_agent.engine.graph import run_pipeline_cluster
        results, final_state = run_pipeline_cluster(
            cluster=cluster,
            credentials=_CREDS,
            config=_CFG,
            repo_path=str(tmp_path),
        )

    # The pipeline should have retried (first write + at least one retry).
    assert call_count[0] >= 2, f"Expected retry, got {call_count[0]} LLM calls"

    # The retry prompt must mention RETRY and the missed arcs.
    retry_messages = captured_prompts[1]
    user_msgs = [m.get("content", "") for m in retry_messages if m.get("role") == "user"]
    combined = "\n".join(user_msgs)
    assert "RETRY" in combined
    # Both missed arcs (35→37 and 37→38) must be named in the critique.
    assert "35" in combined
    assert "37" in combined
    assert "38" in combined
