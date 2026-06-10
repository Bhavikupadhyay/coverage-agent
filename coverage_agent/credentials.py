"""
Credentials — per-run authentication and model configuration.

A Credentials object is built once per run from one of two sources:

- `for_byok()` — key supplied in the request body or config
- `for_cli_env()` — key read from the shell environment (CLI / CI use)

The object is threaded explicitly through the engine so concurrent runs
with different credentials cannot bleed into each other.

Key/model coupling is validated against models.json. Picking a Gemini model
with an Anthropic key is caught at construction time with a clear message.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coverage_agent.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

Strictness = Literal["strict", "balanced", "loose"]

_REGISTRY_PATH = Path(__file__).parent / "models.json"


def _load_registry() -> list[dict]:
    """Returns the full model list from models.json."""
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
    except Exception as exc:
        logger.warning("Could not load models.json: %s", exc)
        return []


def _build_prefix_map(registry: list[dict]) -> tuple[tuple[str, str], ...]:
    """Builds the ordered (prefix, provider) tuple from registry entries.

    Ordering rules:
    - Longer prefixes must come before shorter ones to prevent a shorter
      prefix swallowing a more specific match (e.g. "sk-ant-" before "sk-").
    - Entries with empty key_prefix are skipped (provider can't be inferred).
    """
    seen: dict[str, str] = {}
    for entry in registry:
        prefix = entry.get("key_prefix", "")
        provider = entry.get("provider", "")
        if prefix and provider and prefix not in seen:
            seen[prefix] = provider
    # Sort longest-first to ensure specificity
    return tuple(sorted(seen.items(), key=lambda kv: -len(kv[0])))


_REGISTRY: list[dict] = _load_registry()
_KEY_PREFIXES: tuple[tuple[str, str], ...] = _build_prefix_map(_REGISTRY)


def provider_for_key(key: str) -> str:
    """Detects the provider from an API key's prefix. Returns 'unknown' on no match."""
    k = (key or "").strip()
    for prefix, provider in _KEY_PREFIXES:
        if k.startswith(prefix):
            return provider
    return "unknown"


def provider_for_model(model: str) -> str:
    """Extracts the LiteLLM provider prefix from a model string like 'groq/llama-3.3-70b'."""
    m = (model or "").strip().lower()
    if "/" in m:
        return m.split("/", 1)[0]
    return "unknown"


def validate_model_id(model: str) -> str | None:
    """Checks that model is in the registry.

    Returns None if valid (or unknown — unknown IDs pass through to litellm
    as an escape hatch). Returns a warning string if the model looks wrong
    (e.g. provider mismatch). Never raises.
    """
    if not model:
        return None
    for entry in _REGISTRY:
        if entry["id"] == model:
            return None
    return (
        f"Model '{model}' is not in models.json. It will be passed to litellm as-is. "
        "Run `coverage-agent models` to see supported models."
    )


def _key_env_for_provider(provider: str) -> str | None:
    """Returns the env var name for a provider's API key, or None if unknown."""
    for entry in _REGISTRY:
        if entry.get("provider") == provider:
            key_env = entry.get("key_env", "")
            if key_env:
                return key_env
    return None


def list_models() -> list[dict]:
    """Returns the full model registry for CLI display."""
    return _REGISTRY


@dataclass(frozen=True)
class Credentials:
    llm_api_key: str = ""
    llm_model: str = DEFAULT_MODEL
    eval_strictness: Strictness = "balanced"

    # Gap selection. Default is a deterministic heuristic — free and reproducible.
    # Flip to True for the LLM-ranking path on large, diverse gap sets.
    prioritize_with_llm: bool = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_byok(cls, body: dict) -> "Credentials":
        """BYOK mode — key supplied in a dict (request body, config, or test).

        Validates key/model coupling against the registry. Raises ValueError
        on a clear mismatch (e.g. Groq key + Gemini model). Passes through
        unknown key prefixes — proxy keys and custom deployments are valid.
        """
        llm_key = (body.get("llm_api_key") or "").strip()
        model = (body.get("model") or DEFAULT_MODEL).strip()
        if not llm_key:
            raise ValueError("BYOK mode requires llm_api_key.")

        key_provider = provider_for_key(llm_key)
        model_provider = provider_for_model(model)
        if key_provider != "unknown" and model_provider != "unknown" and key_provider != model_provider:
            raise ValueError(
                f"API key looks like a {key_provider} key, but '{model}' belongs to "
                f"{model_provider}. Supply a {model_provider} key or pick a {key_provider} model.\n"
                f"Run `coverage-agent models` to see all supported models and their key requirements."
            )

        warning = validate_model_id(model)
        if warning:
            logger.warning(warning)

        return cls(
            llm_api_key=llm_key,
            llm_model=model,
            eval_strictness=_coerce_strictness(body.get("eval_strictness")),
        )

    @classmethod
    def for_cli_env(cls) -> "Credentials":
        """CLI / CI mode — reads credentials from the environment.

        Model is read from COVERAGE_AGENT_MODEL (default: gemini/gemini-2.5-flash).
        The matching vendor key env var is resolved from the model registry and
        read directly. A missing key for the chosen model is a hard error — there
        is no fallback to another vendor's key.

        LLM_API_KEY is accepted as a generic override for proxy deployments or
        any model not in the registry.
        """
        model = os.environ.get("COVERAGE_AGENT_MODEL", DEFAULT_MODEL).strip()
        provider = provider_for_model(model)

        # Generic override takes precedence over all vendor-specific vars.
        if generic_key := os.environ.get("LLM_API_KEY", "").strip():
            llm_key = generic_key
        else:
            key_env = _key_env_for_provider(provider)
            if key_env is None:
                raise RuntimeError(
                    f"Model '{model}' has provider '{provider}' which is not in the registry. "
                    "Set LLM_API_KEY as a generic override, or run `coverage-agent models` "
                    "to see supported models."
                )
            llm_key = os.environ.get(key_env, "").strip()
            if not llm_key:
                raise RuntimeError(
                    f"Model '{model}' requires a {provider} API key. "
                    f"Set {key_env} in your environment or .env file.\n"
                    "Run `coverage-agent models` to see all supported models and their key requirements."
                )

        warning = validate_model_id(model)
        if warning:
            logger.warning(warning)

        return cls(
            llm_api_key=llm_key,
            llm_model=model,
            eval_strictness=_coerce_strictness(os.environ.get("EVAL_STRICTNESS")),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def max_retry_loops(self) -> int:
        """Retry budget per gap. strict/balanced allow 3 loops; loose allows 1."""
        return 1 if self.eval_strictness == "loose" else 3

    def should_commit(self, exec_result) -> bool:
        """Single source of truth for the acceptance gate.

        Requires pytest success and proof the target branch was hit.
        """
        if exec_result is None or not exec_result.execution_success:
            return False
        return bool(exec_result.target_branch_hit)

    def litellm_kwargs(self, *, model: str | None = None, max_tokens: int | None = None) -> dict:
        """Returns kwargs to pass to litellm.completion(...).

        litellm picks up api_key per-call without touching os.environ.
        """
        kwargs: dict = {"model": model or self.llm_model}
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def redacted(self) -> dict:
        """Safe-to-log representation. Never include raw keys in logs or errors."""
        return {
            "llm_model": self.llm_model,
            "llm_api_key": _mask(self.llm_api_key),
        }


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


def _coerce_strictness(value) -> Strictness:
    s = (value or "balanced").strip().lower() if isinstance(value, str) else "balanced"
    if s not in ("strict", "balanced", "loose"):
        return "balanced"
    return s  # type: ignore[return-value]
