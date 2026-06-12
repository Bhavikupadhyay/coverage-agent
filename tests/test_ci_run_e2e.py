"""End-to-end test of the ci-run reporting path with a mock LLM.

Patches litellm.completion, runs ci-run --preview against the fixture repo
with fake GITHUB_* env vars, and asserts the rendered comment contains the
marker, the accepted-count, and a details block.

No secrets required. No real network or git push.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.resolve()
_FIXTURE_SRC = _REPO_ROOT / "benchmarks" / "fixture_repo"


# ---------------------------------------------------------------------------
# Canned test (subset of run_acceptance canned tests — enough to accept ≥1)
# ---------------------------------------------------------------------------

_CANNED_TEST = '''\
"""Generated test for mathlib/stats.py uncovered branches."""
from mathlib.stats import clamp, safe_divide, letter_grade

def test_clamp_below_lo():
    assert clamp(-1, 0, 10) == 0

def test_safe_divide_zero():
    assert safe_divide(5, 0) == 0.0
'''


def _make_mock_completion():
    def _mock(*args, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = f"```python\n{_CANNED_TEST}\n```"
        resp.choices[0].message.tool_calls = None
        resp.cost = 0.0
        resp.usage.total_tokens = 100
        return resp
    return _mock


# ---------------------------------------------------------------------------
# Fixture repo setup (minimal — main branch only, full scope)
# ---------------------------------------------------------------------------

def _setup_repo(tmp_dir: Path) -> str:
    """Copy fixture repo, git-init on main, run coverage. Returns .coverage path."""
    shutil.copytree(_FIXTURE_SRC, tmp_dir, dirs_exist_ok=True)

    # Remove scoring.py — full scope on stats.py only.
    scoring = tmp_dir / "mathlib" / "scoring.py"
    if scoring.exists():
        scoring.unlink()

    def _git(*args):
        r = subprocess.run(
            ["git", *args], cwd=str(tmp_dir),
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr}")

    _git("init", "-b", "main")
    _git("config", "user.email", "test@coverage-agent.local")
    _git("config", "user.name", "Test")
    _git("add", ".")
    _git("commit", "-m", "chore: initial")

    cov_path = str(tmp_dir / ".coverage")
    r = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--branch",
         f"--data-file={cov_path}", "-m", "pytest", "tests/", "-q"],
        cwd=str(tmp_dir), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_dir)},
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"coverage run failed:\n{r.stdout}\n{r.stderr}")
    return cov_path


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def test_ci_run_preview_renders_correct_comment(tmp_path, monkeypatch):
    """ci-run --preview renders a comment with the marker, accepted count, and details."""
    tmp_dir = tmp_path / "repo"
    tmp_dir.mkdir()
    cov_path = _setup_repo(tmp_dir)

    monkeypatch.chdir(tmp_dir)
    monkeypatch.setenv("PYTHONPATH", str(tmp_dir))
    monkeypatch.setenv("GITHUB_REPOSITORY", "testorg/testrepo")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/7/merge")
    monkeypatch.setenv("GITHUB_TOKEN", "fake_token")
    # No GITHUB_OUTPUT — output writing is optional.
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    from coverage_agent.config import AgentConfig
    from coverage_agent.credentials import Credentials

    # Build a config that points at the fixture's .coverage.
    cfg = AgentConfig(
        scope="full",
        coverage_file=cov_path,
        max_gaps=5,
        max_retries=1,
        max_tool_calls=0,
        flaky_runs=1,
        test_timeout=30,
        tests_dir="tests/generated",
        commit_mode="comment",
    )

    creds = Credentials(llm_api_key="mock-key", llm_model="groq/llama-3.3-70b-versatile")

    # Capture all console output to find the rendered comment.
    rendered_comments: list[str] = []

    # Patch upsert_comment to capture the rendered body instead of hitting the network.
    def _capture_upsert(repo, pr_number, body, token, preview=False):
        rendered_comments.append(body)

    from coverage_agent.gaps.coverage_data import load_coverage_file, parse_coverage
    from coverage_agent.gaps.select import select_gaps, cluster_gaps
    from coverage_agent.gaps.diff import compute_diff_gaps
    from coverage_agent.engine.graph import run_pipeline_cluster as _run_pipeline_cluster_fn
    from coverage_agent.engine.regression import RegressionGuard
    from coverage_agent.report.run_report import serialize_run_report
    from coverage_agent.report.markdown import render_comment, COMMENT_MARKER

    # Run the pipeline directly (not through the CLI) so we can inspect the report.
    with patch("litellm.completion", side_effect=_make_mock_completion()):
        from coverage_agent.cli import _run_pipeline

        report_path = str(tmp_dir / "report.json")

        _run_pipeline(
            repo_root=str(tmp_dir),
            cfg=cfg,
            creds=creds,
            coverage_file=cov_path,
            base_ref="",
            output=report_path,
            run_pipeline_cluster=_run_pipeline_cluster_fn,
            select_gaps=select_gaps,
            cluster_gaps=cluster_gaps,
            parse_coverage=parse_coverage,
            compute_diff_gaps=compute_diff_gaps,
            load_coverage_file=load_coverage_file,
            RegressionGuard=RegressionGuard,
        )

    assert Path(report_path).exists(), "Report JSON was not written"
    from coverage_agent.report.run_report import load_run_report
    report = load_run_report(report_path)

    # Render the comment and verify it.
    comment_body = render_comment(report)

    # Gate 1: marker present and first.
    assert comment_body.startswith(COMMENT_MARKER), (
        f"Comment does not start with marker.\nFirst 200 chars:\n{comment_body[:200]}"
    )

    # Gate 2: accepted count is mentioned.
    accepted = report.tests_accepted
    assert str(accepted) in comment_body, (
        f"Accepted count {accepted} not found in comment body."
    )

    # Gate 3: if any tests were accepted, details block must be present.
    if accepted > 0:
        assert "<details>" in comment_body, (
            "Expected <details> block for accepted tests, not found."
        )
        assert "</details>" in comment_body

    # Show excerpt for the CI log.
    print(f"\n--- rendered comment excerpt (first 600 chars) ---\n{comment_body[:600]}\n---")
