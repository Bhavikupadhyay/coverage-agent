"""Shared pytest fixtures.

Every test in this suite runs in OFFLINE_MODE by default. Tests that need to
exercise real LLM/E2B paths must mock them explicitly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OFFLINE_MODE", "true")

from coverage_agent.contracts.schemas import (  # noqa: E402
    BranchGap,
    ContextPayload,
    CoverageGap,
    DraftTest,
)
from coverage_agent.credentials import Credentials  # noqa: E402

_FIXTURES_DIR = Path(__file__).parent.parent / "coverage_agent" / "fixtures"


@pytest.fixture
def offline_creds() -> Credentials:
    return Credentials.for_offline()


@pytest.fixture
def byok_creds() -> Credentials:
    return Credentials(
        mode="byok",
        llm_api_key="test-llm-key-xxxx",
        llm_model="groq/llama-3.3-70b-versatile",
        e2b_api_key="test-e2b-key-xxxx",
    )


@pytest.fixture
def sample_gap() -> CoverageGap:
    return CoverageGap(
        file_path="requests/auth.py",
        target_symbol="handle_auth",
        branch=BranchGap(from_line=47, to_line=52),
        surrounding_lines=list(range(40, 70)),
        priority_score=0.0,
        gap_id="requests/auth.py:47->52",
    )


@pytest.fixture
def sample_context() -> ContextPayload:
    data = json.loads((_FIXTURES_DIR / "sample_context.json").read_text(encoding="utf-8"))
    return ContextPayload(**data)


@pytest.fixture
def sample_draft(sample_gap) -> DraftTest:
    code = (_FIXTURES_DIR / "sample_test.py").read_text(encoding="utf-8")
    return DraftTest(
        test_code=code,
        mocks_used=["requests.auth.extract_cookies_to_jar"],
        target_branch=sample_gap.branch,
    )
