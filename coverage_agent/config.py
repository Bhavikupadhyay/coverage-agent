"""
Agent configuration — loaded from .coverage-agent.yml and overridable by flags/env.

All fields have safe defaults so the tool works out of the box with no config file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import litellm
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# litellm retry settings — applied at import time.
litellm.num_retries = 6
litellm.retry_after = 30

DEFAULT_MODEL = "gemini/gemini-2.5-flash"
_CONFIG_FILENAME = ".coverage-agent.yml"


class AgentConfig(BaseModel):
    version: int = 1
    test_command: str = "pytest -q"
    coverage_file: str = ""
    source: list[str] = Field(default_factory=list)
    scope: Literal["full", "diff"] = "full"
    paths: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(
        default_factory=lambda: [
            "**/migrations/**",
            "**/conftest.py",
            "tests/**",
            "test_*.py",
        ]
    )
    tests_dir: str = "tests/generated"
    commit_mode: Literal["comment", "commit", "pr"] = "comment"
    model: str = DEFAULT_MODEL
    max_gaps: int = 10
    max_retries: int = 3
    max_tool_calls: int = 12
    flaky_runs: int = 3
    test_timeout: int = 60
    budget_usd: float = 1.00
    dashboard_url: str = ""

    model_config = {"extra": "ignore"}


def load_config(path: Optional[str] = None) -> AgentConfig:
    """Loads .coverage-agent.yml and returns an AgentConfig.

    Search order:
    1. Explicit path argument.
    2. cwd + parent directories (stops at filesystem root).
    3. Returns defaults if no file is found.
    """
    resolved: Optional[Path] = None

    if path:
        p = Path(path)
        if p.exists():
            resolved = p
        else:
            logger.warning("Config file not found at %s — using defaults", path)
    else:
        current = Path.cwd()
        while True:
            candidate = current / _CONFIG_FILENAME
            if candidate.exists():
                resolved = candidate
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

    if resolved is None:
        logger.debug("No .coverage-agent.yml found — using defaults")
        return AgentConfig()

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        config = AgentConfig(**raw)
        logger.debug("Loaded config from %s", resolved)
        return config
    except Exception as exc:
        logger.warning("Failed to parse %s: %s — using defaults", resolved, exc)
        return AgentConfig()
