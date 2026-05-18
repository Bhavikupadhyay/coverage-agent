"""Pipeline efficiency guarantees:

1. EvalAgent makes ZERO LLM calls (Phase B: sandbox is the only judge).
2. ContextArchitect picks graph_depth via a deterministic heuristic — no LLM call.

These two changes together remove **all** LLM calls that the pipeline used
to make BEFORE the sandbox-execution step. The only remaining LLM-bound
agents are TestWriter (per-gap, the workhorse) and ResultSummarizer (one
call per whole run).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coverage_agent.agents.context_architect import ContextArchitect, _heuristic_depth
from coverage_agent.agents.eval_agent import EvalAgent
from coverage_agent.contracts.schemas import BranchGap, CoverageGap
from coverage_agent.credentials import Credentials


# ---------------------------------------------------------------------------
# EvalAgent: zero LLM calls regardless of strictness
# ---------------------------------------------------------------------------

def test_eval_makes_zero_llm_calls_balanced(sample_gap, sample_context, sample_draft):
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_x",
        llm_model="groq/llama-3.3-70b-versatile",
        eval_strictness="balanced",
    )
    with patch("litellm.completion") as mock_completion:
        EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    mock_completion.assert_not_called()


def test_eval_makes_zero_llm_calls_strict(sample_gap, sample_context, sample_draft):
    """Even strict strictness gets no LLM gate — strictness lives on commit predicate now."""
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_x",
        llm_model="groq/llama-3.3-70b-versatile",
        eval_strictness="strict",
    )
    with patch("litellm.completion") as mock_completion:
        EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    mock_completion.assert_not_called()


def test_eval_makes_zero_llm_calls_loose(sample_gap, sample_context, sample_draft):
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_x",
        llm_model="groq/llama-3.3-70b-versatile",
        eval_strictness="loose",
    )
    with patch("litellm.completion") as mock_completion:
        EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    mock_completion.assert_not_called()


# ---------------------------------------------------------------------------
# ContextArchitect: deterministic depth heuristic — no LLM call
# ---------------------------------------------------------------------------

def _gap_with_n_lines(n: int) -> CoverageGap:
    return CoverageGap(
        gap_id="g1",
        file_path="src/foo.py",
        target_symbol="bar",
        branch=BranchGap(from_line=10, to_line=15),
        surrounding_lines=list(range(1, n + 1)),
        priority_score=0.5,
    )


@pytest.mark.parametrize(
    "n_lines,expected_depth",
    [
        (1, 0),    # tiny gap
        (3, 0),    # boundary: still tiny
        (4, 1),    # typical
        (15, 1),
        (30, 1),   # boundary: still typical
        (31, 2),   # large block
        (100, 2),
    ],
)
def test_heuristic_depth_matches_pattern(n_lines, expected_depth):
    assert _heuristic_depth(_gap_with_n_lines(n_lines)) == expected_depth


def test_context_architect_makes_zero_llm_calls():
    """The whole point of the heuristic: depth decision must not hit the LLM."""
    creds = Credentials(mode="byok", llm_api_key="gsk_x", llm_model="groq/llama-3.3-70b-versatile")
    arch = ContextArchitect(creds)
    fake_sandbox = MagicMock()
    fake_sandbox.build_context.return_value = {
        "primary_code": "def bar(): pass",
        "dependencies_code": {},
        "module_map": {},
        "graph_depth_used": 1,
        "tokens_used": 50,
    }
    with patch("coverage_agent.agents.context_architect.build_context") as mock_build, \
         patch("litellm.completion") as mock_completion:
        arch.run(_gap_with_n_lines(15), sandbox=fake_sandbox)
        mock_completion.assert_not_called()
        mock_build.assert_not_called()  # sandbox path doesn't hit local build_context


def test_context_architect_respects_depth_override():
    """An explicit depth_override skips the heuristic entirely."""
    creds = Credentials(mode="byok", llm_api_key="gsk_x", llm_model="groq/llama-3.3-70b-versatile")
    fake_sandbox = MagicMock()
    fake_sandbox.build_context.return_value = {
        "primary_code": "x", "dependencies_code": {}, "module_map": {},
        "graph_depth_used": 2, "tokens_used": 10,
    }
    ContextArchitect(creds).run(_gap_with_n_lines(1), depth_override=2, sandbox=fake_sandbox)
    fake_sandbox.build_context.assert_called_once()
    args, _ = fake_sandbox.build_context.call_args
    assert args[2] == 2
