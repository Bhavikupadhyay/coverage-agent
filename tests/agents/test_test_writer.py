"""TestWriter should return syntactically valid pytest code in offline mode."""
import ast

from coverage_agent.agents.test_writer import TestWriter, _layout_import_hint
from coverage_agent.contracts.schemas import DraftTest


def test_offline_returns_valid_pytest(offline_creds, sample_gap, sample_context):
    draft = TestWriter(offline_creds).run(sample_gap, sample_context)
    assert isinstance(draft, DraftTest)
    assert "def test_" in draft.test_code or "class Test" in draft.test_code
    # Must parse as Python
    ast.parse(draft.test_code)


def test_extracts_mocks(offline_creds, sample_gap, sample_context):
    draft = TestWriter(offline_creds).run(sample_gap, sample_context)
    # sample_test.py uses patch("requests.auth.extract_cookies_to_jar")
    assert any("extract_cookies_to_jar" in m for m in draft.mocks_used)


def test_layout_import_hint_for_src_requests():
    hint = _layout_import_hint("src/requests/certs.py")
    assert "requests" in hint
    assert "src" in hint
    assert "import src" not in hint  # we tell the model to avoid that pattern


def test_layout_import_hint_empty_for_flat_package():
    assert _layout_import_hint("requests/certs.py") == ""


def test_target_branch_propagates(offline_creds, sample_gap, sample_context):
    draft = TestWriter(offline_creds).run(sample_gap, sample_context)
    assert draft.target_branch == sample_gap.branch
