"""TestWriter: generates pytest code by calling the LLM."""
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

from coverage_agent.engine.writer import TestWriter, _layout_import_hint
from coverage_agent.contracts import DraftTest

_FIXTURE_TEST = (
    Path(__file__).parent.parent.parent / "coverage_agent" / "fixtures" / "sample_test.py"
).read_text(encoding="utf-8")


def _fake_completion(*args, **kwargs):
    resp = MagicMock()
    resp.choices[0].message.content = _FIXTURE_TEST
    return resp


def test_returns_valid_pytest(creds, sample_gap, sample_context):
    with patch("litellm.completion", side_effect=_fake_completion):
        draft = TestWriter(creds).run(sample_gap, sample_context)
    assert isinstance(draft, DraftTest)
    assert "def test_" in draft.test_code or "class Test" in draft.test_code
    ast.parse(draft.test_code)


def test_extracts_mocks(creds, sample_gap, sample_context):
    with patch("litellm.completion", side_effect=_fake_completion):
        draft = TestWriter(creds).run(sample_gap, sample_context)
    assert any("extract_cookies_to_jar" in m for m in draft.mocks_used)


def test_layout_import_hint_for_src_requests():
    hint = _layout_import_hint("src/requests/certs.py")
    assert "requests" in hint
    assert "src" in hint
    assert "import src" not in hint


def test_layout_import_hint_empty_for_flat_package():
    assert _layout_import_hint("requests/certs.py") == ""


def test_target_branch_propagates(creds, sample_gap, sample_context):
    with patch("litellm.completion", side_effect=_fake_completion):
        draft = TestWriter(creds).run(sample_gap, sample_context)
    assert draft.target_branch == sample_gap.branch


def test_critique_included_in_retry_prompt(creds, sample_gap, sample_context):
    captured_prompts = []

    def _capture(*args, **kwargs):
        captured_prompts.append(kwargs.get("messages", []))
        return _fake_completion()

    with patch("litellm.completion", side_effect=_capture):
        TestWriter(creds).run(sample_gap, sample_context, critique="Fix your imports.")

    user_msg = captured_prompts[0][-1]["content"]
    assert "RETRY" in user_msg
    assert "Fix your imports." in user_msg
