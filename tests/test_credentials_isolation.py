"""
Credentials isolation tests.

The whole point of the Credentials refactor is that two concurrent runs
with different keys cannot bleed into each other. These tests cover the
contract: keys flow explicitly, os.environ is never mutated, and litellm
receives the api_key as a per-call kwarg.
"""
from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from coverage_agent.agents.gap_prioritizer import GapPrioritizer
from coverage_agent.contracts.schemas import BranchGap, CoverageGap
from coverage_agent.credentials import Credentials


def _make_gap() -> CoverageGap:
    return CoverageGap(
        file_path="pkg/mod.py",
        target_symbol="fn",
        branch=BranchGap(from_line=10, to_line=12),
        surrounding_lines=[10, 11, 12],
        priority_score=0.0,
        gap_id="pkg/mod.py:10->12",
    )


def test_byok_credentials_carry_keys():
    creds = Credentials.for_byok({
        "llm_api_key": "gsk_test_abc123",
        "e2b_api_key": "e2b_test_xyz",
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.mode == "byok"
    assert creds.llm_api_key == "gsk_test_abc123"
    assert creds.e2b_api_key == "e2b_test_xyz"


def test_byok_rejects_missing_keys():
    with pytest.raises(ValueError):
        Credentials.for_byok({"e2b_api_key": "x"})
    with pytest.raises(ValueError):
        Credentials.for_byok({"llm_api_key": "x"})


def test_offline_has_no_keys():
    creds = Credentials.for_offline()
    assert creds.mode == "offline"
    assert creds.llm_api_key == ""
    assert creds.e2b_api_key == ""
    assert creds.is_offline


def test_redacted_never_exposes_full_key():
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_supersecretkey_xyz",
        e2b_api_key="e2b_anothersecretkey_ab",
    )
    red = creds.redacted()
    assert "supersecret" not in str(red)
    assert "anothersecret" not in str(red)
    assert red["llm_api_key"].startswith("gsk_")
    assert "..." in red["llm_api_key"]


def test_litellm_kwargs_carry_per_call_key():
    creds = Credentials(mode="byok", llm_api_key="gsk_xyz", llm_model="groq/x")
    kwargs = creds.litellm_kwargs()
    assert kwargs == {"model": "groq/x", "api_key": "gsk_xyz"}


def test_litellm_kwargs_omits_key_when_offline():
    creds = Credentials.for_offline()
    kwargs = creds.litellm_kwargs()
    assert "api_key" not in kwargs


def test_run_does_not_mutate_os_environ(offline_creds):
    snapshot = dict(os.environ)
    GapPrioritizer(offline_creds).run([_make_gap()])
    assert dict(os.environ) == snapshot


def test_concurrent_credentials_do_not_leak():
    """Two threads, two different keys — each agent must see only its own.

    We patch litellm.completion to record the api_key it received, then run
    two prioritizers in parallel with distinct credentials.
    """
    seen: dict[int, list[str]] = {0: [], 1: []}
    gate = threading.Event()

    # Phase D made the deterministic heuristic the default — opt into the LLM
    # path explicitly so this isolation contract still exercises real litellm
    # calls (the thing we actually care about isolating between concurrent runs).
    creds_a = Credentials(mode="byok", llm_api_key="gsk_aaaa", llm_model="groq/x",
                          prioritize_with_llm=True)
    creds_b = Credentials(mode="byok", llm_api_key="gsk_bbbb", llm_model="groq/x",
                          prioritize_with_llm=True)
    gaps = [_make_gap()]

    def fake_completion(**kwargs):
        # Tag which thread by api_key
        key = kwargs.get("api_key", "")
        if key == "gsk_aaaa":
            seen[0].append(key)
        elif key == "gsk_bbbb":
            seen[1].append(key)

        class _Choice:
            class message:
                content = "[1.0]"
        class _Resp:
            choices = [_Choice]
        gate.wait(timeout=2.0)
        return _Resp()

    with patch("coverage_agent.agents.gap_prioritizer.litellm.completion", side_effect=fake_completion):

        def run(creds):
            GapPrioritizer(creds).run(gaps)

        t1 = threading.Thread(target=run, args=(creds_a,))
        t2 = threading.Thread(target=run, args=(creds_b,))
        t1.start()
        t2.start()
        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

    assert seen[0] == ["gsk_aaaa"]
    assert seen[1] == ["gsk_bbbb"]
    assert "gsk_aaaa" not in seen[1]
    assert "gsk_bbbb" not in seen[0]
