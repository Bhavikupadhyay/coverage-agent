"""
Credentials — per-run authentication and model configuration.

A Credentials object is built once per run from one of three sources:

- `for_offline()` — no real keys, agents return fixture data
- `for_demo()` — server's DEMO_* keys, capped quotas
- `for_byok()` — keys supplied in the request body

The object is then threaded explicitly through Orchestrator -> agents ->
sandbox -> braintrust. No agent reads os.environ during a run, so concurrent
runs with different credentials cannot bleed into each other.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from coverage_agent.config import DEFAULT_MODEL

Mode = Literal["demo", "byok", "offline"]
Strictness = Literal["strict", "balanced", "loose"]

# Recognised LLM provider prefixes. The first match wins, so order matters:
# `sk-ant-` MUST come before `sk-` to be detected as Anthropic, not OpenAI.
_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gsk_",    "groq"),
    ("sk-ant-", "anthropic"),
    ("sk-proj-", "openai"),
    ("sk-",     "openai"),
    ("AIza",    "gemini"),
    ("csk-",    "cerebras"),
)


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


@dataclass(frozen=True)
class Credentials:
    mode: Mode
    llm_api_key: str = ""
    llm_model: str = DEFAULT_MODEL
    e2b_api_key: str = ""
    braintrust_api_key: str = ""
    # Retry budget only. Commits always need execution_success AND target_branch_hit
    # (see should_commit). Presets differ only in max_retry_loops: loose = 1,
    # strict/balanced = 3.
    eval_strictness: Strictness = "balanced"

    # Phase D: gap selection. Default is a deterministic heuristic — no LLM
    # call, more reproducible, and free. Flip to True for the LLM-ranking
    # path which can be valuable on large, diverse gap sets but burns one
    # 70B call per run and is unverifiable on benchmark deltas.
    prioritize_with_llm: bool = False

    # Sandbox backend: local runs in a subprocess+venv on the caller's machine
    # (free, no external API, correct for benchmarks on trusted repos).
    # e2b spins up a cloud VM (required for the web demo with untrusted repos).
    sandbox_mode: Literal["local", "e2b"] = "local"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_offline(cls) -> "Credentials":
        """Offline mode — agents return fixture data, no real keys needed."""
        return cls(mode="offline", llm_model=DEFAULT_MODEL)

    @classmethod
    def for_demo(cls, eval_strictness: str = "balanced") -> "Credentials":
        """Demo mode — reads dedicated DEMO_* keys from the server's environment.

        Raises if the demo path is misconfigured. Keep DEMO_* keys distinct
        from BYOK env keys so a misconfigured server can't accidentally hand
        the developer's full-power key to a public demo user.
        """
        llm_key = os.environ.get("DEMO_GROQ_API_KEY", "").strip()
        e2b_key = os.environ.get("DEMO_E2B_API_KEY", "").strip()
        bt_key = os.environ.get("DEMO_BRAINTRUST_API_KEY", "").strip()
        if not llm_key or not e2b_key:
            raise RuntimeError(
                "Demo mode is not configured. Set DEMO_GROQ_API_KEY and "
                "DEMO_E2B_API_KEY in the server environment, or disable demo mode."
            )
        return cls(
            mode="demo",
            llm_api_key=llm_key,
            llm_model=os.environ.get("DEMO_MODEL", DEFAULT_MODEL),
            e2b_api_key=e2b_key,
            braintrust_api_key=bt_key,
            eval_strictness=_coerce_strictness(eval_strictness),
            sandbox_mode="e2b",
        )

    @classmethod
    def for_byok(cls, body: dict) -> "Credentials":
        """BYOK mode — keys come from the request body. Raises if missing or mismatched."""
        llm_key = (body.get("llm_api_key") or "").strip()
        e2b_key = (body.get("e2b_api_key") or "").strip()
        model = (body.get("model") or DEFAULT_MODEL).strip()
        if not llm_key:
            raise ValueError("BYOK mode requires llm_api_key in the request body.")
        if not e2b_key:
            raise ValueError("BYOK mode requires e2b_api_key in the request body.")

        # The most common BYOK footgun is pasting a Groq key while leaving the
        # model dropdown on Gemini (or vice-versa). LiteLLM will happily send
        # the request and you get a confusing 401 from the wrong provider.
        # Reject it up front with a clear message.
        key_provider = provider_for_key(llm_key)
        model_provider = provider_for_model(model)
        if key_provider != "unknown" and key_provider != model_provider:
            raise ValueError(
                f"LLM key looks like a {key_provider} key, but the selected model "
                f"is from {model_provider}. Pick a {key_provider} model or supply "
                f"a {model_provider} key."
            )

        return cls(
            mode="byok",
            llm_api_key=llm_key,
            llm_model=model,
            e2b_api_key=e2b_key,
            braintrust_api_key=(body.get("braintrust_api_key") or "").strip(),
            eval_strictness=_coerce_strictness(body.get("eval_strictness")),
            sandbox_mode="e2b",
        )

    @classmethod
    def for_cli_env(cls, sandbox_mode: Literal["local", "e2b"] = "local") -> "Credentials":
        """CLI mode — reads keys from the developer's shell environment.

        Used by run_benchmark.py. Looks at GROQ_API_KEY / E2B_API_KEY /
        BRAINTRUST_API_KEY directly.
        """
        return cls(
            mode="byok",
            llm_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            llm_model=os.environ.get("COVERAGE_AGENT_MODEL", DEFAULT_MODEL),
            e2b_api_key=os.environ.get("E2B_API_KEY", "").strip(),
            braintrust_api_key=os.environ.get("BRAINTRUST_API_KEY", "").strip(),
            eval_strictness=_coerce_strictness(os.environ.get("EVAL_STRICTNESS")),
            sandbox_mode=sandbox_mode,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_offline(self) -> bool:
        return self.mode == "offline"

    def commit_requires_branch_hit(self) -> bool:
        """Always true: a commit is only recorded when sandbox coverage proves the
        target branch was executed. eval_strictness does not lower this bar; it
        only changes max_retry_loops().
        """
        return True

    def max_retry_loops(self) -> int:
        """Retry budget shared across Eval REWRITE/RECONTEXTUALIZE and sandbox failure.

        strict + balanced allow up to 3 loops; loose allows 1.
        """
        return 1 if self.eval_strictness == "loose" else 3

    def should_commit(self, exec_result) -> bool:
        """Single source of truth for 'is this test good enough to commit?'.

        Requires pytest/coverage success and proof the target branch appears in
        executed_branches (target_branch_hit). Used by the pipeline router and
        the orchestrator so they cannot disagree.
        """
        if exec_result is None or not exec_result.execution_success:
            return False
        if not exec_result.target_branch_hit:
            return False
        return True

    def litellm_kwargs(self, *, model: str | None = None, max_tokens: int | None = None) -> dict:
        """Returns kwargs to pass to litellm.completion(...).

        litellm picks up api_key per-call without touching os.environ.
        `model` lets callers override the default reasoning model (rarely
        useful now that Eval is deterministic — left in place for future
        per-agent model routing if it becomes worth the complexity).
        `max_tokens` caps output length so a chatty model can't burn budget.
        """
        kwargs: dict = {"model": model or self.llm_model}
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def redacted(self) -> dict:
        """Safe-to-log representation. Never include raw keys in logs/errors."""
        return {
            "mode": self.mode,
            "llm_model": self.llm_model,
            "sandbox_mode": self.sandbox_mode,
            "llm_api_key": _mask(self.llm_api_key),
            "e2b_api_key": _mask(self.e2b_api_key),
            "braintrust_api_key": _mask(self.braintrust_api_key),
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
