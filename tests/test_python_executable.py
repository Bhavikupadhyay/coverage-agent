"""Tests that a custom python_executable lands in subprocess argv.

Verifies the job-python invariant: when AgentConfig.python_executable is set,
executor._run_once, tools.run_candidate, and regression._run_test_command all
spawn the configured interpreter instead of sys.executable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from coverage_agent.config import AgentConfig
from coverage_agent.contracts import BranchGap, CoverageGap, DraftTest, ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CUSTOM_PYTHON = "/usr/bin/custom_python_3_99"


def _make_gap() -> CoverageGap:
    return CoverageGap(
        file_path="pkg/mod.py",
        target_symbol="fn",
        branch=BranchGap(from_line=5, to_line=8),
        surrounding_lines=[5, 6, 7, 8],
        kind="branch",
        origin="full",
        gap_id="pkg/mod.py:5->8",
    )


def _make_draft() -> DraftTest:
    return DraftTest(
        test_code="def test_fn(): assert True\n",
        mocks_used=[],
        target_branch=BranchGap(from_line=5, to_line=8),
    )


# ---------------------------------------------------------------------------
# executor._run_once
# ---------------------------------------------------------------------------

def test_executor_run_once_uses_custom_python(tmp_path):
    """_run_once must use python_executable in the coverage run argv."""
    from coverage_agent.engine.executor import _run_once

    test_file = tmp_path / "test_candidate.py"
    test_file.write_text("def test_x(): assert True\n")
    cov_file = tmp_path / ".cov"

    captured_cmds = []

    def _fake_run(cmd, capture_output=False, text=False, cwd=None, timeout=None, **kwargs):
        captured_cmds.append(list(cmd))
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    # coverage is imported locally inside _run_once (import coverage as coverage_module).
    # Patch it via sys.modules so the local import returns our mock.
    import types
    mock_cov_instance = MagicMock()
    mock_data = MagicMock()
    mock_data.arcs.return_value = [(5, 8)]
    mock_cov_instance.get_data.return_value = mock_data

    fake_coverage_mod = types.ModuleType("coverage")
    fake_coverage_mod.Coverage = MagicMock(return_value=mock_cov_instance)

    with patch("subprocess.run", side_effect=_fake_run):
        with patch("coverage_agent.engine.executor._check_targets", return_value=(1, 1)):
            with patch.dict("sys.modules", {"coverage": fake_coverage_mod}):
                _run_once(test_file, _make_gap(), cov_file, str(tmp_path), 30, CUSTOM_PYTHON)

    assert captured_cmds, "subprocess.run was never called"
    first_cmd = captured_cmds[0]
    assert first_cmd[0] == CUSTOM_PYTHON, (
        f"Expected argv[0]={CUSTOM_PYTHON!r}, got {first_cmd[0]!r}"
    )


def test_executor_run_once_defaults_to_sys_executable(tmp_path):
    """When python_executable is empty, _run_once falls back to sys.executable."""
    from coverage_agent.engine.executor import _run_once

    test_file = tmp_path / "test_candidate.py"
    test_file.write_text("def test_x(): assert True\n")
    cov_file = tmp_path / ".cov"

    captured_cmds = []

    def _fake_run(cmd, capture_output=False, text=False, cwd=None, timeout=None, **kwargs):
        captured_cmds.append(list(cmd))
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    import types
    mock_cov_instance = MagicMock()
    mock_cov_instance.get_data.return_value = MagicMock()

    fake_coverage_mod = types.ModuleType("coverage")
    fake_coverage_mod.Coverage = MagicMock(return_value=mock_cov_instance)

    with patch("subprocess.run", side_effect=_fake_run):
        with patch("coverage_agent.engine.executor._check_targets", return_value=(1, 1)):
            with patch.dict("sys.modules", {"coverage": fake_coverage_mod}):
                _run_once(test_file, _make_gap(), cov_file, str(tmp_path), 30, "")

    assert captured_cmds[0][0] == sys.executable


# ---------------------------------------------------------------------------
# ExecutionRunner._run_subprocess (integration path via AgentConfig)
# ---------------------------------------------------------------------------

def test_execution_runner_threads_python_executable(creds, tmp_path, monkeypatch):
    """ExecutionRunner must pass cfg.python_executable into _run_once."""
    monkeypatch.chdir(tmp_path)
    gap = _make_gap()
    draft = _make_draft()
    cfg = AgentConfig(
        tests_dir=str(tmp_path / "tests_gen"),
        flaky_runs=1,
        python_executable=CUSTOM_PYTHON,
    )

    captured_kwargs = {}

    def _fake_run_once(test_file, gap, cov_data_file, cwd, timeout, python_executable="", cluster=None):
        captured_kwargs["python_executable"] = python_executable
        return ExecutionResult(
            execution_success=True,
            target_branch_hit=True,
            targets_hit=1,
            targets_total=1,
        ), {}

    from coverage_agent.engine.executor import ExecutionRunner
    with patch("coverage_agent.engine.executor._run_once", side_effect=_fake_run_once):
        ExecutionRunner(creds).run(draft, gap, config=cfg)

    assert captured_kwargs.get("python_executable") == CUSTOM_PYTHON


# ---------------------------------------------------------------------------
# tools.run_candidate
# ---------------------------------------------------------------------------

def test_tools_run_candidate_uses_custom_python():
    """run_candidate must use python_executable in the pytest subprocess argv."""
    from coverage_agent.engine.tools import run_candidate

    captured_cmds = []

    def _fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    with patch("subprocess.run", side_effect=_fake_run):
        run_candidate(
            test_code="def test_x(): assert True\n",
            python_executable=CUSTOM_PYTHON,
        )

    assert captured_cmds, "subprocess.run was never called"
    assert captured_cmds[0][0] == CUSTOM_PYTHON, (
        f"Expected argv[0]={CUSTOM_PYTHON!r}, got {captured_cmds[0][0]!r}"
    )


# ---------------------------------------------------------------------------
# regression._run_test_command
# ---------------------------------------------------------------------------

def test_regression_uses_custom_python(tmp_path):
    """_run_test_command must prepend the custom python when command starts with 'pytest'."""
    from coverage_agent.engine.regression import _run_test_command

    captured_cmds = []

    def _fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        r = MagicMock()
        r.returncode = 0
        return r

    junit_xml = tmp_path / "junit.xml"

    with patch("subprocess.run", side_effect=_fake_run):
        _run_test_command("pytest -q", junit_xml, str(tmp_path), python_executable=CUSTOM_PYTHON)

    assert captured_cmds, "subprocess.run was never called"
    cmd = captured_cmds[0]
    assert cmd[0] == CUSTOM_PYTHON
    assert cmd[1] == "-m"
    assert cmd[2] == "pytest"
