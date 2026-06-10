"""EvalAgent — deterministic gate.

No LLM calls: syntax + import-plausibility only. The executor is the
ground truth for whether a test is good enough to accept.
"""
from unittest.mock import MagicMock, patch

from coverage_agent.engine.validator import EvalAgent
from coverage_agent.contracts import BranchGap, DraftTest
from coverage_agent.credentials import Credentials


def test_clean_test_routes_execute(creds, sample_gap, sample_context, sample_draft):
    result = EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    assert result.route == "EXECUTE"
    assert result.syntax_valid is True


def test_syntax_error_routes_rewrite(creds, sample_gap, sample_context):
    broken = DraftTest(
        test_code="def test_bad(:\n    pass\n",
        mocks_used=[],
        target_branch=BranchGap(from_line=1, to_line=2),
    )
    result = EvalAgent(creds).run(broken, sample_context, sample_gap)
    assert result.syntax_valid is False
    assert result.route == "REWRITE"


def test_modules_imported_in_primary_snippet_are_known(creds, sample_gap, sample_context):
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
    result = EvalAgent(creds).run(draft, ctx, sample_gap)
    assert result.route == "EXECUTE"


def test_unknown_import_routes_recontextualize(creds, sample_gap, sample_context):
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
    result = EvalAgent(creds).run(draft, sample_context, sample_gap)
    assert result.route == "RECONTEXTUALIZE"
    assert "totally_made_up_pkg" in result.unknown_imports
    assert "Unknown imports" in result.critique


def test_eval_makes_zero_llm_calls(creds, sample_gap, sample_context, sample_draft):
    """EvalAgent is fully deterministic — no LLM call, ever."""
    with patch("coverage_agent.engine.validator.subprocess.run") as mock_subprocess, \
         patch("litellm.completion") as mock_completion:
        mock_subprocess.return_value = MagicMock(stdout="[]", returncode=0)
        EvalAgent(creds).run(sample_draft, sample_context, sample_gap)
    mock_completion.assert_not_called()


def test_strictness_not_on_eval_agent():
    """Strictness lives on Credentials.should_commit / max_retry_loops. EvalAgent has none."""
    agent = EvalAgent(Credentials(eval_strictness="strict"))
    assert not hasattr(agent, "_min_assertion_score")
    assert not hasattr(agent, "_require_mock_complete")
