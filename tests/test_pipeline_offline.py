"""End-to-end offline pipeline test.

Runs the full Orchestrator against fixture data with no real LLM, no real
E2B, no real network. Asserts the scorecard shape and that committed tests
contain valid Python.
"""
from __future__ import annotations

import ast

from coverage_agent.contracts.schemas import GapResult
from coverage_agent.orchestrator import Orchestrator


def test_orchestrator_offline_end_to_end(offline_creds):
    orch = Orchestrator(credentials=offline_creds)
    scorecard, results = orch.run(
        repo_url_or_path="https://github.com/example/fake",
        max_gaps=2,
    )

    assert isinstance(results, list)
    assert results, "fixture pipeline should produce at least one result"
    assert all(isinstance(r, GapResult) for r in results)

    expected_keys = {
        "repo", "gaps_targeted", "tests_committed", "skipped",
        "branch_hit_rate", "avg_coverage_delta", "avg_loops", "llm_cost",
    }
    assert expected_keys.issubset(set(scorecard.keys()))

    assert scorecard["gaps_targeted"] == len(results)
    assert scorecard["repo"] == "https://github.com/example/fake"
    assert "OFFLINE" in scorecard["llm_cost"]


def test_committed_tests_are_valid_python(offline_creds):
    orch = Orchestrator(credentials=offline_creds)
    _, results = orch.run(
        repo_url_or_path="https://github.com/example/fake",
        max_gaps=1,
    )
    committed = [r for r in results if r.final_test_committed]
    assert committed, "offline fixtures should commit at least one test"
    for r in committed:
        ast.parse(r.test_code)


def test_no_os_environ_pollution(offline_creds):
    import os
    snapshot = dict(os.environ)
    orch = Orchestrator(credentials=offline_creds)
    orch.run(repo_url_or_path="https://github.com/example/fake", max_gaps=1)
    assert dict(os.environ) == snapshot, "Orchestrator must not mutate os.environ"
