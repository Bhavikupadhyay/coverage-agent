"""AgentConfig and load_config tests."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from coverage_agent.config import AgentConfig, load_config, DEFAULT_MODEL


def test_defaults():
    cfg = AgentConfig()
    assert cfg.scope == "full"
    assert cfg.max_gaps == 10
    assert cfg.model == DEFAULT_MODEL
    assert cfg.flaky_runs == 3
    assert cfg.budget_usd == 1.00


def test_load_config_from_explicit_path(tmp_path):
    yml = tmp_path / ".coverage-agent.yml"
    yml.write_text(textwrap.dedent("""\
        version: 1
        scope: diff
        max_gaps: 5
        test_command: "pytest tests/ -q"
    """))
    cfg = load_config(str(yml))
    assert cfg.scope == "diff"
    assert cfg.max_gaps == 5
    assert cfg.test_command == "pytest tests/ -q"
    # non-specified fields retain defaults
    assert cfg.flaky_runs == 3


def test_load_config_missing_path_returns_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "nonexistent.yml"))
    assert cfg.scope == "full"
    assert cfg.max_gaps == 10


def test_load_config_no_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg, AgentConfig)
    assert cfg.model == DEFAULT_MODEL


def test_load_config_cwd_discovery(tmp_path, monkeypatch):
    yml = tmp_path / ".coverage-agent.yml"
    yml.write_text("max_gaps: 3\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.max_gaps == 3


def test_load_config_parent_discovery(tmp_path, monkeypatch):
    yml = tmp_path / ".coverage-agent.yml"
    yml.write_text("tests_dir: custom/tests\n")
    subdir = tmp_path / "src" / "pkg"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)
    cfg = load_config()
    assert cfg.tests_dir == "custom/tests"


def test_unknown_keys_ignored(tmp_path):
    yml = tmp_path / ".coverage-agent.yml"
    yml.write_text("max_gaps: 7\nunknown_future_key: whatever\n")
    cfg = load_config(str(yml))
    assert cfg.max_gaps == 7
    assert not hasattr(cfg, "unknown_future_key")


def test_malformed_yaml_returns_defaults(tmp_path):
    yml = tmp_path / ".coverage-agent.yml"
    yml.write_text(": bad: yaml: [\n")
    cfg = load_config(str(yml))
    assert isinstance(cfg, AgentConfig)
    assert cfg.max_gaps == 10
