"""EvalAgent — deterministic gate.

After the Phase B rewrite the agent has no LLM dependency: it's a syntax +
import-plausibility check, and the sandbox is the only thing that decides
whether a test is good enough to commit. These tests pin that contract.
"""
from unittest.mock import MagicMock, patch

from coverage_agent.agents.eval_agent import EvalAgent
from coverage_agent.contracts.schemas import BranchGap, DraftTest
from coverage_agent.credentials import Credentials


def test_offline_clean_test_routes_execute(offline_creds, sample_gap, sample_context, sample_draft):
    """A clean draft in offline mode still routes EXECUTE — Eval has no LLM gate."""
    result = EvalAgent(offline_creds).run(sample_draft, sample_context, sample_gap)
    assert result.route == "EXECUTE"
    assert result.syntax_valid is True


def test_syntax_error_routes_rewrite(offline_creds, sample_gap, sample_context):
    broken = DraftTest(
        test_code="def test_bad(:\n    pass\n",
        mocks_used=[],
        target_branch=BranchGap(from_line=1, to_line=2),
    )
    result = EvalAgent(offline_creds).run(broken, sample_context, sample_gap)
    assert result.syntax_valid is False
    assert result.route == "REWRITE"


def test_modules_imported_in_primary_snippet_are_known(offline_creds, sample_gap, sample_context):
    """Target excerpt may import helpers (e.g. certifi) omitted from dependencies_code."""
    ctx = sample_context.model_copy(
        update={"primary_code": "import certifi\n\n" + sample_context.primary_code}
    )
    draft = DraftTest(
        test_code=(
            "import pytest\n"
            "import certifi\n"
            "def test_x():\n"
            "    assert certifi.where() is not None\n"
        ),
        mocks_used=[],
        target_branch=sample_gap.branch,
    )
    result = EvalAgent(offline_creds).run(draft, ctx, sample_gap)
    assert result.route == "EXECUTE"


def test_unknown_import_routes_recontextualize(offline_creds, sample_gap, sample_context):
    """Imports that aren't stdlib, target-package, or context-known → ask for more context."""
    draft = DraftTest(
        test_code=(
            "import pytest\n"
            "from totally_made_up_pkg import nonexistent_helper\n"
            "def test_x():\n"
            "    nonexistent_helper()\n"
            "    assert True\n"
        ),
        mocks_used=[],
        target_branch=BranchGap(from_line=1, to_line=2),
    )
    result = EvalAgent(offline_creds).run(draft, sample_context, sample_gap)
    assert result.route == "RECONTEXTUALIZE"
    assert "totally_made_up_pkg" in result.unknown_imports
    assert "Unknown imports" in result.critique


def test_eval_makes_zero_llm_calls_in_byok_mode(sample_gap, sample_context, sample_draft):
    """The whole point of Phase B: no LLM in the eval path. Sandbox is judge."""
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_x",
        llm_model="groq/llama-3.3-70b-versatile",
        eval_strictness="balanced",
    )
    with patch("coverage_agent.agents.eval_agent.subprocess.run") as mock_subprocess, \
         patch("litellm.completion") as mock_completion:
        # Force ruff path so we don't call into ast (still deterministic, but
        # exercises the same code path as a real install)
        mock_subprocess.return_value = MagicMock(stdout="[]", returncode=0)
        EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    mock_completion.assert_not_called()


def test_strictness_no_longer_lives_on_eval_agent():
    """Strictness moved to Credentials.should_commit / max_retry_loops. EvalAgent has none."""
    agent = EvalAgent(Credentials(mode="offline", eval_strictness="strict"))
    assert not hasattr(agent, "_min_assertion_score")
    assert not hasattr(agent, "_require_mock_complete")
    assert not hasattr(agent, "_skip_mock_check")
