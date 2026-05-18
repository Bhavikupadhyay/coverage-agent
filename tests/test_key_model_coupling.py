"""Key↔model coupling — make sure a Groq key + Gemini model never reaches LiteLLM."""
from __future__ import annotations

import pytest

from coverage_agent.credentials import (
    Credentials,
    provider_for_key,
    provider_for_model,
)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("gsk_abc123",          "groq"),
        ("sk-proj-abc",         "openai"),
        ("sk-abc",              "openai"),
        ("sk-ant-abc",          "anthropic"),  # MUST resolve before sk- alone
        ("AIzaSyABC",           "gemini"),
        ("csk-cere-1",          "cerebras"),
        ("",                    "unknown"),
        ("totally-custom-key",  "unknown"),
    ],
)
def test_provider_for_key(key, expected):
    assert provider_for_key(key) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("groq/llama-3.3-70b-versatile", "groq"),
        ("gemini/gemini-2.5-pro",        "gemini"),
        ("openai/gpt-4o-mini",           "openai"),
        ("anthropic/claude-3-haiku",     "anthropic"),
        ("just-a-model-name",            "unknown"),
        ("",                             "unknown"),
    ],
)
def test_provider_for_model(model, expected):
    assert provider_for_model(model) == expected


def test_for_byok_accepts_matching_groq_key_and_model():
    creds = Credentials.for_byok({
        "llm_api_key": "gsk_aaa",
        "e2b_api_key": "e2b_zzz",
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.llm_model.startswith("groq/")


def test_for_byok_rejects_groq_key_with_gemini_model():
    with pytest.raises(ValueError, match=r"groq.*gemini|gemini.*groq"):
        Credentials.for_byok({
            "llm_api_key": "gsk_aaa",
            "e2b_api_key": "e2b_zzz",
            "model": "gemini/gemini-2.5-pro",
        })


def test_for_byok_rejects_openai_key_with_groq_model():
    with pytest.raises(ValueError, match=r"openai.*groq|groq.*openai"):
        Credentials.for_byok({
            "llm_api_key": "sk-proj-zzz",
            "e2b_api_key": "e2b_zzz",
            "model": "groq/llama-3.3-70b-versatile",
        })


def test_for_byok_does_not_reject_when_key_prefix_is_unknown():
    """A custom-prefix key (e.g. proxy) should be accepted — we can't infer the provider."""
    creds = Credentials.for_byok({
        "llm_api_key": "internal-proxy-key-xyz",
        "e2b_api_key": "e2b_zzz",
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.llm_api_key == "internal-proxy-key-xyz"
