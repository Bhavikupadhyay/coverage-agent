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
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.llm_model.startswith("groq/")


def test_for_byok_rejects_groq_key_with_gemini_model():
    with pytest.raises(ValueError, match=r"groq.*gemini|gemini.*groq"):
        Credentials.for_byok({
            "llm_api_key": "gsk_aaa",
            "model": "gemini/gemini-2.5-pro",
        })


def test_for_byok_rejects_openai_key_with_groq_model():
    with pytest.raises(ValueError, match=r"openai.*groq|groq.*openai"):
        Credentials.for_byok({
            "llm_api_key": "sk-proj-zzz",
            "model": "groq/llama-3.3-70b-versatile",
        })


def test_for_byok_does_not_reject_when_key_prefix_is_unknown():
    creds = Credentials.for_byok({
        "llm_api_key": "internal-proxy-key-xyz",
        "model": "groq/llama-3.3-70b-versatile",
    })
    assert creds.llm_api_key == "internal-proxy-key-xyz"


def test_registry_driven_prefix_map_includes_all_vendors():
    """_KEY_PREFIXES is built from models.json, not hardcoded."""
    from coverage_agent.credentials import _KEY_PREFIXES
    providers = {p for _, p in _KEY_PREFIXES}
    # All vendors with a known key prefix must be represented
    assert "gemini" in providers
    assert "anthropic" in providers
    assert "openai" in providers
    assert "groq" in providers
    assert "xai" in providers
    assert "cerebras" in providers


def test_anthropic_key_rejected_for_gemini_model():
    with pytest.raises(ValueError, match=r"anthropic.*gemini|gemini.*anthropic"):
        Credentials.for_byok({
            "llm_api_key": "sk-ant-key",
            "model": "gemini/gemini-2.5-flash",
        })


def test_xai_key_accepted_for_xai_model():
    creds = Credentials.for_byok({
        "llm_api_key": "xai-somekey",
        "model": "xai/grok-3",
    })
    assert creds.llm_api_key == "xai-somekey"
