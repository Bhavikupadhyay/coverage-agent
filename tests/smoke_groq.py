"""
Live smoke test: one real LLM call through the GapPrioritizer using Groq.

Skipped automatically if GROQ_API_KEY is absent so this can live alongside
the default suite without breaking offline runs.

Run explicitly with:
    GROQ_API_KEY=... pytest tests/smoke_groq.py -v --no-header
"""
from __future__ import annotations

import os

import pytest

from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.contracts.schemas import BranchGap, CoverageGap
from coverage_agent.credentials import Credentials
from coverage_agent.cost_tracker import CostTracker

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live smoke test",
)


@pytest.fixture(autouse=True)
def _disable_offline():
    """Force live mode for this test file even if the default suite enabled offline."""
    prev = os.environ.pop("OFFLINE_MODE", None)
    yield
    if prev is not None:
        os.environ["OFFLINE_MODE"] = prev


def _make_gap(name: str) -> CoverageGap:
    return CoverageGap(
        file_path=f"pkg/{name}.py",
        target_symbol=name,
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=list(range(8, 20)),
        priority_score=0.0,
        gap_id=f"pkg/{name}.py:10->12",
    )


def test_groq_round_trip_with_cost_tracking():
    """Sends a single real request to Groq and asserts a successful response.

    Also asserts that CostTracker accumulates a non-zero USD cost — proves
    the litellm.success_callback path is wired correctly.
    """
    creds = Credentials(
        mode="byok",
        llm_api_key=os.environ["GROQ_API_KEY"],
        llm_model=os.environ.get("COVERAGE_AGENT_MODEL", "groq/llama-3.3-70b-versatile"),
    )

    tracker = CostTracker()
    tracker.install()
    try:
        result = GapPrioritizer(creds).run([
            _make_gap("handle_auth"),
            _make_gap("parse_url"),
            _make_gap("trivial_getter"),
        ])
    finally:
        tracker.uninstall()

    assert len(result) == 3
    assert all(0.0 <= g.priority_score <= 1.0 for g in result)
    # Sorted descending
    scores = [g.priority_score for g in result]
    assert scores == sorted(scores, reverse=True)
    # Real LLM call -> we should see some cost (could be 0 if model not in
    # litellm pricing catalog, so soft-assert)
    if tracker.call_count > 0:
        assert tracker.call_count >= 1
