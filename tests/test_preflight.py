"""Preflight checks — repo / LLM / E2B validation before a run starts."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from coverage_agent.credentials import Credentials
from coverage_agent.preflight import (
    CheckResult,
    PreflightReport,
    check_e2b,
    check_llm,
    check_repo,
    run_preflight,
)


# ---------------------------------------------------------------------------
# check_repo
# ---------------------------------------------------------------------------

def _fake_gh_response(body: dict, status: int = 200):
    """Builds a context-manager-compatible mock urlopen response."""
    payload = json.dumps(body).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=payload)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_check_repo_rejects_non_github_url():
    r = check_repo("https://gitlab.com/foo/bar")
    assert r.ok is False
    assert "github" in r.message.lower()


def test_check_repo_rejects_empty_url():
    r = check_repo("")
    assert r.ok is False


def test_check_repo_accepts_public_python_repo():
    fake = _fake_gh_response({
        "private": False,
        "language": "Python",
        "size": 5120,
        "default_branch": "main",
    })
    with patch("urllib.request.urlopen", return_value=fake):
        r = check_repo("https://github.com/psf/requests")
    assert r.ok is True
    assert "psf/requests" in r.message
    assert "Python" in r.message


def test_check_repo_rejects_private_repo():
    fake = _fake_gh_response({"private": True, "language": "Python", "size": 100})
    with patch("urllib.request.urlopen", return_value=fake):
        r = check_repo("https://github.com/x/private-repo")
    assert r.ok is False
    assert "private" in r.message.lower()


def test_check_repo_rejects_non_python_repo():
    fake = _fake_gh_response({"private": False, "language": "Go", "size": 100})
    with patch("urllib.request.urlopen", return_value=fake):
        r = check_repo("https://github.com/x/go-repo")
    assert r.ok is False
    assert "go" in r.message.lower()


def test_check_repo_handles_404_as_not_found():
    import urllib.error
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        r = check_repo("https://github.com/x/y")
    assert r.ok is False
    assert "not found" in r.message.lower()


def test_check_repo_handles_403_rate_limit_gracefully():
    import urllib.error
    err = urllib.error.HTTPError("u", 403, "rate limit", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        r = check_repo("https://github.com/x/y")
    # Rate-limited GH means we let the user proceed.
    assert r.ok is True
    assert "rate" in r.message.lower()


# ---------------------------------------------------------------------------
# check_llm
# ---------------------------------------------------------------------------

def test_check_llm_rejects_missing_key():
    r = check_llm("", "groq/llama-3.3-70b-versatile")
    assert r.ok is False


def test_check_llm_rejects_provider_mismatch():
    r = check_llm("gsk_aaa", "gemini/gemini-2.5-pro")
    assert r.ok is False
    assert "groq" in r.message.lower()
    assert "gemini" in r.message.lower()


def test_check_llm_calls_litellm_with_1_token():
    fake = MagicMock()
    fake.choices = [MagicMock()]
    fake.choices[0].message.content = "ok"
    with patch("coverage_agent.preflight.litellm.completion", return_value=fake) as mock:
        r = check_llm("gsk_aaa", "groq/llama-3.3-70b-versatile")
    assert r.ok is True
    _, kwargs = mock.call_args
    assert kwargs.get("max_tokens") == 1
    assert kwargs.get("api_key") == "gsk_aaa"
    assert kwargs.get("model") == "groq/llama-3.3-70b-versatile"


def test_check_llm_auth_error_maps_to_friendly_message():
    with patch(
        "coverage_agent.preflight.litellm.completion",
        side_effect=RuntimeError("AuthenticationError: invalid api key"),
    ):
        r = check_llm("gsk_bad", "groq/llama-3.3-70b-versatile")
    assert r.ok is False
    assert "rejected" in r.message.lower()


def test_check_llm_model_not_found_maps_clearly():
    with patch(
        "coverage_agent.preflight.litellm.completion",
        side_effect=RuntimeError("Model not found: 404"),
    ):
        r = check_llm("gsk_aaa", "groq/imaginary-model")
    assert r.ok is False
    assert "not available" in r.message.lower()


# ---------------------------------------------------------------------------
# check_e2b
# ---------------------------------------------------------------------------

def test_check_e2b_rejects_missing_key():
    r = check_e2b("")
    assert r.ok is False


def test_check_e2b_calls_sandbox_list():
    fake_sandbox_module = MagicMock()
    fake_sandbox_module.Sandbox.list.return_value = []
    with patch.dict("sys.modules", {"e2b": fake_sandbox_module}):
        r = check_e2b("e2b_aaa")
    assert r.ok is True
    fake_sandbox_module.Sandbox.list.assert_called_once_with(api_key="e2b_aaa")


def test_check_e2b_auth_error_maps_to_rejected():
    fake_sandbox_module = MagicMock()
    fake_sandbox_module.Sandbox.list.side_effect = RuntimeError("401 unauthorized")
    with patch.dict("sys.modules", {"e2b": fake_sandbox_module}):
        r = check_e2b("e2b_bad")
    assert r.ok is False
    assert "rejected" in r.message.lower()


# ---------------------------------------------------------------------------
# run_preflight integration
# ---------------------------------------------------------------------------

def test_run_preflight_offline_short_circuits_real_calls(offline_creds):
    fake = _fake_gh_response({"private": False, "language": "Python", "size": 100})
    with patch("urllib.request.urlopen", return_value=fake):
        report = run_preflight(
            repo_url="https://github.com/psf/requests",
            mode="offline",
            credentials=offline_creds,
        )
    assert report.ready is True
    assert report.llm.ok and "offline" in report.llm.message.lower()
    assert report.e2b.ok and "offline" in report.e2b.message.lower()


def test_run_preflight_demo_skips_byok_checks():
    fake = _fake_gh_response({"private": False, "language": "Python", "size": 100})
    demo_creds = Credentials(mode="demo", llm_api_key="server", e2b_api_key="server")
    with patch("urllib.request.urlopen", return_value=fake):
        report = run_preflight(
            repo_url="https://github.com/psf/requests",
            mode="demo",
            credentials=demo_creds,
        )
    assert report.ready is True
    assert "server" in report.llm.message.lower()


def test_run_preflight_returns_not_ready_when_any_check_fails(byok_creds):
    fake = _fake_gh_response({"private": True, "language": "Python", "size": 100})
    fake_sb = MagicMock()
    fake_sb.Sandbox.list.return_value = []
    fake_llm_response = MagicMock()
    fake_llm_response.choices = [MagicMock()]
    fake_llm_response.choices[0].message.content = "ok"
    with patch("urllib.request.urlopen", return_value=fake), \
         patch.dict("sys.modules", {"e2b": fake_sb}), \
         patch("coverage_agent.preflight.litellm.completion", return_value=fake_llm_response):
        report = run_preflight(
            repo_url="https://github.com/x/private",
            mode="byok",
            credentials=byok_creds,
        )
    assert report.ready is False
    assert report.repo.ok is False  # private → not ok
    assert report.llm.ok is True
    assert report.e2b.ok is True


def test_run_preflight_to_dict_shape(offline_creds):
    fake = _fake_gh_response({"private": False, "language": "Python", "size": 100})
    with patch("urllib.request.urlopen", return_value=fake):
        report = run_preflight(
            repo_url="https://github.com/psf/requests",
            mode="offline",
            credentials=offline_creds,
        )
    d = report.to_dict()
    assert set(d.keys()) == {"ready", "repo", "llm", "e2b"}
    for sub in ("repo", "llm", "e2b"):
        assert set(d[sub].keys()) >= {"ok", "message"}
