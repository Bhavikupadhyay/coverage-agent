"""Tests for the --repo flag on `coverage-agent run`."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from typer.testing import CliRunner

from coverage_agent.cli import app

runner = CliRunner()


def _fake_creds():
    from coverage_agent.credentials import Credentials
    creds = MagicMock(spec=Credentials)
    creds.llm_model = "gemini/gemini-2.5-flash"
    return creds


# ---------------------------------------------------------------------------
# Helpers — patch everything past credential resolution so tests stay fast
# ---------------------------------------------------------------------------

_PATCH_CREDS = "coverage_agent.cli.Credentials.for_cli_env"
_PATCH_RUN_PIPELINE = "coverage_agent.cli._run_pipeline"


def _run_pipeline_noop(**kwargs):
    pass


@pytest.fixture()
def mock_run_pipeline(monkeypatch):
    called_with = {}

    def _capture(**kwargs):
        called_with.update(kwargs)

    monkeypatch.setattr("coverage_agent.cli._run_pipeline", _capture)
    return called_with


# ---------------------------------------------------------------------------
# --repo with a GitHub URL → clones into tmpdir
# ---------------------------------------------------------------------------

def test_repo_url_triggers_clone(tmp_path, mock_run_pipeline):
    fake_clone_dir = str(tmp_path / "clone")

    with patch(_PATCH_CREDS, return_value=_fake_creds()), \
         patch("tempfile.mkdtemp", return_value=fake_clone_dir), \
         patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")) as mock_sp_run, \
         patch("shutil.rmtree") as mock_rmtree:

        result = runner.invoke(app, [
            "run",
            "--repo", "https://github.com/example/repo",
            "--model", "gemini/gemini-2.5-flash",
        ])

    # git clone should have been called with the URL
    clone_call = mock_sp_run.call_args_list[0]
    cmd = clone_call[0][0]
    assert cmd[0] == "git"
    assert cmd[1] == "clone"
    assert "https://github.com/example/repo" in cmd

    # _run_pipeline received the tmp clone dir as repo_root
    assert mock_run_pipeline.get("repo_root") == fake_clone_dir

    # cleanup ran
    mock_rmtree.assert_called_once_with(fake_clone_dir, ignore_errors=True)


def test_repo_url_clone_failure_exits(tmp_path):
    fake_clone_dir = str(tmp_path / "clone")

    with patch(_PATCH_CREDS, return_value=_fake_creds()), \
         patch("tempfile.mkdtemp", return_value=fake_clone_dir), \
         patch("subprocess.run", return_value=MagicMock(returncode=128, stderr="fatal: repository not found")), \
         patch("shutil.rmtree"):

        result = runner.invoke(app, [
            "run",
            "--repo", "https://github.com/no/such",
            "--model", "gemini/gemini-2.5-flash",
        ])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --repo with a local path → skips clone, uses resolved path
# ---------------------------------------------------------------------------

def test_repo_local_path_skips_clone(tmp_path, mock_run_pipeline):
    # Create a fake git repo directory
    fake_repo = tmp_path / "myrepo"
    fake_repo.mkdir()
    (fake_repo / ".git").mkdir()

    with patch(_PATCH_CREDS, return_value=_fake_creds()):
        result = runner.invoke(app, [
            "run",
            "--repo", str(fake_repo),
            "--model", "gemini/gemini-2.5-flash",
        ])

    assert mock_run_pipeline.get("repo_root") == str(fake_repo.resolve())


def test_repo_local_path_not_found_exits(tmp_path):
    with patch(_PATCH_CREDS, return_value=_fake_creds()):
        result = runner.invoke(app, [
            "run",
            "--repo", str(tmp_path / "doesnotexist"),
            "--model", "gemini/gemini-2.5-flash",
        ])
    assert result.exit_code != 0


def test_repo_local_path_not_git_exits(tmp_path):
    not_git = tmp_path / "notgit"
    not_git.mkdir()

    with patch(_PATCH_CREDS, return_value=_fake_creds()):
        result = runner.invoke(app, [
            "run",
            "--repo", str(not_git),
            "--model", "gemini/gemini-2.5-flash",
        ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# No --repo → uses cwd, no clone attempted
# ---------------------------------------------------------------------------

def test_no_repo_uses_cwd(mock_run_pipeline):
    with patch(_PATCH_CREDS, return_value=_fake_creds()):
        runner.invoke(app, [
            "run",
            "--model", "gemini/gemini-2.5-flash",
        ])

    from pathlib import Path
    assert mock_run_pipeline.get("repo_root") == str(Path.cwd())
