"""
Preflight checks — fast, cheap validation before a run starts.

Three independent checks:

- repo:   HEAD the GitHub API to confirm the repo exists, is public, and is
          (probably) Python. Pure HTTPS, ~150 ms, no auth needed.
- llm:    one 1-token chat completion against the user's chosen model. Validates
          auth + provider availability + model name. ~1 token cost (free on Groq).
- e2b:    list active sandboxes for the API key. Validates auth without
          creating a real sandbox (which would cost ~30 s + sandbox minutes).

Returns a structured result the UI can render as inline status chips so the
user knows everything is green before committing to a 2-minute pipeline run.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Optional

import litellm

from coverage_agent.credentials import (
    Credentials,
    provider_for_key,
    provider_for_model,
)

logger = logging.getLogger(__name__)

_GH_REPO_RE = re.compile(r"^https?://github\.com/([^/\s]+)/([^/\s?#]+?)(?:\.git)?/?$")


@dataclass
class CheckResult:
    ok: bool
    message: str
    detail: str = ""


@dataclass
class PreflightReport:
    ready: bool
    repo: CheckResult
    llm: CheckResult
    e2b: CheckResult

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "repo": asdict(self.repo),
            "llm": asdict(self.llm),
            "e2b": asdict(self.e2b),
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_repo(repo_url: str) -> CheckResult:
    """Validates a GitHub URL by hitting the public API. Public repos only."""
    if not repo_url:
        return CheckResult(ok=False, message="No repo URL provided.")

    match = _GH_REPO_RE.match(repo_url.strip())
    if not match:
        return CheckResult(
            ok=False,
            message="Only public GitHub URLs (https://github.com/owner/repo) are supported.",
        )

    owner, repo = match.group(1), match.group(2)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "CoverageAgent/0.1"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return CheckResult(ok=False, message="Repository not found or private.")
        if exc.code == 403:
            return CheckResult(
                ok=True,
                message="GitHub API rate-limited — proceeding without metadata check.",
                detail="public API rate limit",
            )
        return CheckResult(ok=False, message=f"GitHub returned HTTP {exc.code}.")
    except Exception as exc:
        return CheckResult(
            ok=True,
            message="Couldn't reach GitHub (no network?), but URL format is valid.",
            detail=str(exc)[:120],
        )

    if data.get("private"):
        return CheckResult(ok=False, message="Repository is private. Demo mode supports public repos only.")

    language = (data.get("language") or "").lower()
    if language and language != "python":
        return CheckResult(
            ok=False,
            message=f"Primary language is {data['language']}, not Python.",
            detail="CoverageAgent only supports Python repositories.",
        )

    size_kb = data.get("size", 0)
    return CheckResult(
        ok=True,
        message=f"{owner}/{repo} · Python · {size_kb // 1024 or 1} MB",
        detail=data.get("default_branch", ""),
    )


def check_llm(llm_api_key: str, model: str) -> CheckResult:
    """Sends a 1-token completion to verify the key works with the chosen model."""
    if not llm_api_key:
        return CheckResult(ok=False, message="No LLM key provided.")

    key_provider = provider_for_key(llm_api_key)
    model_provider = provider_for_model(model)
    if key_provider != "unknown" and key_provider != model_provider:
        return CheckResult(
            ok=False,
            message=f"Key is {key_provider}, model is {model_provider}. Pick a {key_provider} model.",
        )

    try:
        # max_tokens=1 keeps this essentially free — Groq's free tier ignores this
        # for usage tracking entirely.
        response = litellm.completion(
            model=model,
            api_key=llm_api_key,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
            timeout=10,
        )
        _ = response.choices[0].message.content  # touch the result so a malformed response surfaces
        return CheckResult(
            ok=True,
            message=f"{model_provider or 'LLM'} reachable",
            detail=model,
        )
    except Exception as exc:
        text = str(exc)[:160]
        # litellm raises BadRequestError for invalid model, AuthenticationError for bad key.
        if "auth" in text.lower() or "api key" in text.lower() or "401" in text:
            return CheckResult(ok=False, message="LLM key rejected by provider.", detail=text)
        if "model" in text.lower() and ("not found" in text.lower() or "404" in text):
            return CheckResult(ok=False, message=f"Model {model} not available on this account.", detail=text)
        return CheckResult(ok=False, message="LLM check failed.", detail=text)


def check_e2b(e2b_api_key: str) -> CheckResult:
    """Lists active sandboxes — fast and validates auth without provisioning."""
    if not e2b_api_key:
        return CheckResult(ok=False, message="No E2B key provided.")

    try:
        # Lazy import: the e2b SDK is large and we don't want to pay its load
        # cost unless preflight is actually exercised. Also lets tests monkey-
        # patch `sys.modules['e2b']` without polluting the global import graph.
        from e2b import Sandbox  # noqa: PLC0415
        # `list` is the cheapest auth-validating call. It returns active sandboxes
        # for this key and raises on bad credentials. We never instantiate one.
        Sandbox.list(api_key=e2b_api_key)
        return CheckResult(ok=True, message="E2B reachable", detail="auth ok")
    except Exception as exc:
        text = str(exc)[:160]
        if "401" in text or "auth" in text.lower() or "unauthorized" in text.lower():
            return CheckResult(ok=False, message="E2B key rejected.", detail=text)
        return CheckResult(ok=False, message="E2B check failed.", detail=text)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_preflight(
    repo_url: str,
    mode: str,
    credentials: Optional[Credentials] = None,
) -> PreflightReport:
    """Runs the three preflight checks. Returns a PreflightReport."""
    repo_result = check_repo(repo_url)

    if mode == "demo":
        # Demo keys live on the server. The web app already validated they exist
        # when Credentials.for_demo() was called. We trust them here.
        llm_result = CheckResult(ok=True, message="Server-managed Groq key", detail="demo mode")
        e2b_result = CheckResult(ok=True, message="Server-managed E2B key", detail="demo mode")
    elif credentials is None:
        return PreflightReport(
            ready=False,
            repo=repo_result,
            llm=CheckResult(ok=False, message="No credentials supplied for BYOK mode."),
            e2b=CheckResult(ok=False, message="No credentials supplied for BYOK mode."),
        )
    elif credentials.is_offline:
        llm_result = CheckResult(ok=True, message="Offline mode — no LLM check needed")
        e2b_result = CheckResult(ok=True, message="Offline mode — no sandbox needed")
    else:
        llm_result = check_llm(credentials.llm_api_key, credentials.llm_model)
        e2b_result = check_e2b(credentials.e2b_api_key)

    return PreflightReport(
        ready=repo_result.ok and llm_result.ok and e2b_result.ok,
        repo=repo_result,
        llm=llm_result,
        e2b=e2b_result,
    )
