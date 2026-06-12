"""
GitHub REST delivery layer — stdlib urllib.request only, no new dependencies.

Every function accepts an explicit token argument. Token values are never
logged or printed. Call sites read the token from env; this module does not.

preview=True prints what WOULD be done (method, URL, body excerpt / git
commands) instead of executing any network or git operations.
"""
from __future__ import annotations

import json
import logging
import subprocess
import urllib.error
import urllib.request
from typing import Optional

from coverage_agent.report.markdown import COMMENT_MARKER

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_PER_PAGE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_request(
    method: str,
    url: str,
    token: str,
    body: Optional[dict] = None,
) -> dict:
    """Execute a single GitHub REST API call and return the parsed JSON body."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace") if exc.fp else ""
        logger.debug("GitHub API %s %s → %d: %s", method, url, exc.code, body_text[:500])
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_comment(
    repo: str,
    pr_number: int,
    body: str,
    token: str,
    preview: bool = False,
) -> None:
    """Create or update the PR comment that starts with COMMENT_MARKER.

    If an existing comment whose body starts with the marker is found,
    PATCH it. Otherwise POST a new one.

    Args:
        repo:       "owner/repo"
        pr_number:  pull request number
        body:       full comment body (must start with COMMENT_MARKER)
        token:      GitHub token — never read inside; passed from the call site
        preview:    if True, print the action instead of executing it
    """
    list_url = f"{_API_BASE}/repos/{repo}/issues/{pr_number}/comments?per_page={_PER_PAGE}"

    if preview:
        excerpt = body[:120].replace("\n", " ")
        print(f"[preview] GET {list_url}")
        print(f"[preview] would upsert comment on {repo}#{pr_number}: {excerpt!r}…")
        return

    # Fetch existing comments.
    existing_id: Optional[int] = None
    try:
        comments = _api_request("GET", list_url, token)
        for c in comments:
            if isinstance(c, dict) and (c.get("body") or "").startswith(COMMENT_MARKER):
                existing_id = c["id"]
                break
    except Exception as exc:
        logger.warning("upsert_comment: failed to list comments: %s", exc)

    payload = {"body": body}

    if existing_id is not None:
        patch_url = f"{_API_BASE}/repos/{repo}/issues/comments/{existing_id}"
        logger.info("upsert_comment: PATCH %s", patch_url)
        _api_request("PATCH", patch_url, token, payload)
    else:
        post_url = f"{_API_BASE}/repos/{repo}/issues/{pr_number}/comments"
        logger.info("upsert_comment: POST %s", post_url)
        _api_request("POST", post_url, token, payload)


def push_commit(
    repo_root: str,
    tests_dir: str,
    token: str,
    preview: bool = False,
) -> None:
    """Add files under tests_dir to git, commit with a coverage-agent: prefix, and push.

    Implemented via git subprocess, not REST, so the user's git credential
    configuration is honoured (the GITHUB_TOKEN is injected by the Action's
    checkout step, not by this function).

    Args:
        repo_root:  absolute path to the git working tree
        tests_dir:  repo-relative path to the generated tests directory
        token:      GitHub token (unused here — kept for API symmetry with upsert_comment)
        preview:    if True, print the git commands instead of running them
    """
    cmds = [
        ["git", "add", tests_dir],
        ["git", "commit", "-m", "coverage-agent: add generated tests"],
        ["git", "push"],
    ]

    if preview:
        for cmd in cmds:
            print(f"[preview] {' '.join(cmd)}")
        return

    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
        if result.returncode != 0:
            raise RuntimeError(
                f"git command failed: {' '.join(cmd)}\n{result.stderr.strip()}"
            )


def open_pr(
    repo: str,
    pr_number: int,
    head_branch: str,
    tests_dir: str,
    repo_root: str,
    token: str,
    preview: bool = False,
) -> Optional[str]:
    """Create a stacked PR with the generated tests on a new branch.

    Flow:
      1. Create branch coverage-agent/pr-<pr_number> from the current HEAD.
      2. Add and commit files under tests_dir.
      3. Push the branch.
      4. POST /repos/{repo}/pulls against head_branch.

    Returns the new PR URL, or None in preview mode.

    Args:
        repo:        "owner/repo"
        pr_number:   parent PR number (used for the branch name and PR title)
        head_branch: the base branch for the new PR (the parent PR's head branch)
        tests_dir:   repo-relative path to the generated tests directory
        repo_root:   absolute path to the git working tree
        token:       GitHub token — never read inside; passed from the call site
        preview:     if True, print actions instead of executing them
    """
    branch_name = f"coverage-agent/pr-{pr_number}"
    cmds = [
        ["git", "checkout", "-b", branch_name],
        ["git", "add", tests_dir],
        ["git", "commit", "-m", f"coverage-agent: add generated tests for pr-{pr_number}"],
        ["git", "push", "-u", "origin", branch_name],
    ]
    pulls_url = f"{_API_BASE}/repos/{repo}/pulls"
    pull_payload = {
        "title": f"coverage-agent: generated tests for #{pr_number}",
        "head": branch_name,
        "base": head_branch,
        "body": (
            f"{COMMENT_MARKER}\n\n"
            f"Auto-generated tests for #{pr_number}.\n\n"
            "Merge only after verifying the suite is green."
        ),
    }

    if preview:
        for cmd in cmds:
            print(f"[preview] {' '.join(cmd)}")
        print(f"[preview] POST {pulls_url} head={branch_name} base={head_branch}")
        return None

    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
        if result.returncode != 0:
            raise RuntimeError(
                f"git command failed: {' '.join(cmd)}\n{result.stderr.strip()}"
            )

    resp = _api_request("POST", pulls_url, token, pull_payload)
    pr_url: str = resp.get("html_url", "")
    logger.info("open_pr: created %s", pr_url)
    return pr_url
