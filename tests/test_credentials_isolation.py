"""
Credentials isolation tests.

Keys flow explicitly through the engine; os.environ is never mutated;
litellm receives api_key as a per-call kwarg.
"""
from __future__ import annotations

import pytest

from coverage_agent.credentials import Credentials


def test_byok_carries_keys():
    creds = Credentials.for_byok({
        "llm_api_key": "gsk_test_abc123",
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.llm_api_key == "gsk_test_abc123"
    assert creds.llm_model == "groq/llama-3.3-70b-versatile"


def test_byok_rejects_missing_key():
    with pytest.raises(ValueError):
        Credentials.for_byok({})


def test_redacted_never_exposes_full_key():
    creds = Credentials(llm_api_key="gsk_supersecretkey_xyz")
    red = creds.redacted()
    assert "supersecret" not in str(red)
    assert red["llm_api_key"].startswith("gsk_")
    assert "..." in red["llm_api_key"]


def test_litellm_kwargs_carry_per_call_key():
    creds = Credentials(llm_api_key="gsk_xyz", llm_model="groq/x")
    assert creds.litellm_kwargs() == {"model": "groq/x", "api_key": "gsk_xyz"}


def test_litellm_kwargs_omits_key_when_empty():
    creds = Credentials(llm_api_key="", llm_model="groq/x")
    assert "api_key" not in creds.litellm_kwargs()
