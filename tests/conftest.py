"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coverage_agent.contracts import (
    BranchGap,
    ContextPayload,
    CoverageGap,
    DraftTest,
)
from coverage_agent.credentials import Credentials

_FIXTURES_DIR = Path(__file__).parent.parent / "coverage_agent" / "fixtures"

_TEST_CREDS = Credentials(llm_api_key="gsk_test-key-xxxx", llm_model="groq/llama-3.3-70b-versatile")


@pytest.fixture
def creds() -> Credentials:
    return _TEST_CREDS


@pytest.fixture
def sample_gap() -> CoverageGap:
    return CoverageGap(
        file_path="requests/auth.py",
        target_symbol="handle_auth",
        branch=BranchGap(from_line=47, to_line=52),
        surrounding_lines=list(range(40, 70)),
        kind="branch",
        origin="full",
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
