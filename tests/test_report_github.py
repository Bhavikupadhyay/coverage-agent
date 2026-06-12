"""Tests for coverage_agent/report/github.py — upsert_comment PATCH-vs-POST decision."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call
from io import BytesIO

import pytest

from coverage_agent.report.github import upsert_comment, COMMENT_MARKER
from coverage_agent.report.markdown import COMMENT_MARKER as MARKDOWN_MARKER


# The marker must be the same object in both modules.
def test_marker_import_consistency():
    assert COMMENT_MARKER == MARKDOWN_MARKER


# ---------------------------------------------------------------------------
# Recorded-shape JSON fixtures
# ---------------------------------------------------------------------------

def _make_comment(comment_id: int, body: str) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": "github-actions[bot]"},
        "html_url": f"https://github.com/owner/repo/issues/1#issuecomment-{comment_id}",
    }


_COMMENTS_WITH_MARKER = json.dumps([
    _make_comment(1001, "some unrelated comment"),
    _make_comment(1002, f"{COMMENT_MARKER}\n\n**coverage-agent** found 3 gaps"),
    _make_comment(1003, "another unrelated comment"),
]).encode()

_COMMENTS_WITHOUT_MARKER = json.dumps([
    _make_comment(1001, "some unrelated comment"),
    _make_comment(1003, "another unrelated comment"),
]).encode()

_EMPTY_COMMENTS = json.dumps([]).encode()

_PATCH_RESPONSE = json.dumps({"id": 1002, "body": "updated"}).encode()
_POST_RESPONSE = json.dumps({"id": 9999, "body": "new comment"}).encode()


# ---------------------------------------------------------------------------
# urllib mock helpers
# ---------------------------------------------------------------------------

def _make_urlopen_response(bodies: list[bytes]):
    """Returns a side_effect callable that yields successive response bodies."""
    responses = list(bodies)
    index = [0]

    def _urlopen(req, *args, **kwargs):
        i = index[0]
        index[0] += 1
        body = responses[i] if i < len(responses) else b"{}"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body
        return mock_resp

    return _urlopen


# ---------------------------------------------------------------------------
# PATCH path — marker comment exists
# ---------------------------------------------------------------------------

def test_upsert_patches_existing_comment():
    """When the marker comment exists, must use PATCH, not POST."""
    comment_body = f"{COMMENT_MARKER}\n\nupdated report"

    with patch("urllib.request.urlopen", side_effect=_make_urlopen_response([
        _COMMENTS_WITH_MARKER,  # GET comments list
        _PATCH_RESPONSE,        # PATCH existing comment
    ])) as mock_urlopen:
        upsert_comment("owner/repo", 42, comment_body, token="tok_test")

    calls = mock_urlopen.call_args_list
    assert len(calls) == 2

    # First call: GET the comments list.
    get_req = calls[0][0][0]
    assert get_req.get_method() == "GET"
    assert "/issues/42/comments" in get_req.full_url

    # Second call: PATCH the existing comment (id 1002).
    patch_req = calls[1][0][0]
    assert patch_req.get_method() == "PATCH"
    assert "/issues/comments/1002" in patch_req.full_url

    # Payload must contain the new body.
    payload = json.loads(patch_req.data.decode())
    assert payload["body"] == comment_body


# ---------------------------------------------------------------------------
# POST path — no marker comment present
# ---------------------------------------------------------------------------

def test_upsert_posts_when_no_existing_comment():
    """When no marker comment exists, must use POST, not PATCH."""
    comment_body = f"{COMMENT_MARKER}\n\nbrand new report"

    with patch("urllib.request.urlopen", side_effect=_make_urlopen_response([
        _COMMENTS_WITHOUT_MARKER,  # GET comments list
        _POST_RESPONSE,            # POST new comment
    ])) as mock_urlopen:
        upsert_comment("owner/repo", 7, comment_body, token="tok_test")

    calls = mock_urlopen.call_args_list
    assert len(calls) == 2

    # First call: GET
    get_req = calls[0][0][0]
    assert get_req.get_method() == "GET"

    # Second call: POST to the issues comments endpoint.
    post_req = calls[1][0][0]
    assert post_req.get_method() == "POST"
    assert "/issues/7/comments" in post_req.full_url

    payload = json.loads(post_req.data.decode())
    assert payload["body"] == comment_body


# ---------------------------------------------------------------------------
# POST path — empty comment list
# ---------------------------------------------------------------------------

def test_upsert_posts_to_empty_comment_list():
    comment_body = f"{COMMENT_MARKER}\n\nfirst report"

    with patch("urllib.request.urlopen", side_effect=_make_urlopen_response([
        _EMPTY_COMMENTS,
        _POST_RESPONSE,
    ])) as mock_urlopen:
        upsert_comment("owner/repo", 1, comment_body, token="tok_test")

    calls = mock_urlopen.call_args_list
    assert len(calls) == 2
    post_req = calls[1][0][0]
    assert post_req.get_method() == "POST"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def test_upsert_uses_correct_repo_and_pr():
    comment_body = f"{COMMENT_MARKER}\nbody"
    with patch("urllib.request.urlopen", side_effect=_make_urlopen_response([
        _COMMENTS_WITHOUT_MARKER,
        _POST_RESPONSE,
    ])) as mock_urlopen:
        upsert_comment("myorg/myrepo", 99, comment_body, token="tok_test")

    get_req = mock_urlopen.call_args_list[0][0][0]
    assert "myorg/myrepo" in get_req.full_url
    assert "99" in get_req.full_url

    post_req = mock_urlopen.call_args_list[1][0][0]
    assert "myorg/myrepo" in post_req.full_url
    assert "99" in post_req.full_url


# ---------------------------------------------------------------------------
# Preview mode — no network calls
# ---------------------------------------------------------------------------

def test_upsert_preview_makes_no_network_calls(capsys):
    with patch("urllib.request.urlopen") as mock_urlopen:
        upsert_comment("owner/repo", 1, f"{COMMENT_MARKER}\nbody", token="tok", preview=True)

    mock_urlopen.assert_not_called()
    out = capsys.readouterr().out
    assert "[preview]" in out


# ---------------------------------------------------------------------------
# Auth header — token passed correctly, not logged
# ---------------------------------------------------------------------------

def test_upsert_sends_bearer_token():
    comment_body = f"{COMMENT_MARKER}\nbody"
    with patch("urllib.request.urlopen", side_effect=_make_urlopen_response([
        _COMMENTS_WITHOUT_MARKER,
        _POST_RESPONSE,
    ])) as mock_urlopen:
        upsert_comment("owner/repo", 1, comment_body, token="my_secret_token")

    for call_args in mock_urlopen.call_args_list:
        req = call_args[0][0]
        auth = req.get_header("Authorization")
        assert auth == "Bearer my_secret_token"
