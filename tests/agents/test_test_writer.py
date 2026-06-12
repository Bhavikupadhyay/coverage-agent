"""TestWriter: generates pytest code by calling the LLM."""
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

from coverage_agent.engine.writer import TestWriter, _layout_import_hint, _build_user_prompt
from coverage_agent.contracts import BranchGap, ContextPayload, CoverageGap, DraftTest

_FIXTURE_TEST = (
    Path(__file__).parent.parent.parent / "coverage_agent" / "fixtures" / "sample_test.py"
).read_text(encoding="utf-8")


def _fake_completion(*args, **kwargs):
    resp = MagicMock()
    resp.choices[0].message.content = _FIXTURE_TEST
    resp.choices[0].message.tool_calls = None
    resp.cost = 0.0
    resp.usage.total_tokens = 100
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


def test_react_tool_call_loop(creds, sample_gap, sample_context):
    """Writer handles one tool-call turn followed by a text response."""
    tool_call_resp = MagicMock()
    tool_call_resp.choices[0].message.content = ""
    tool_call_resp.choices[0].message.tool_calls = [MagicMock(
        id="call_1",
        function=MagicMock(name="read_source", arguments='{"path": "some/file.py"}'),
    )]
    tool_call_resp.cost = 0.0
    tool_call_resp.usage.total_tokens = 50

    text_resp = MagicMock()
    text_resp.choices[0].message.content = f"```python\n{_FIXTURE_TEST}\n```"
    text_resp.choices[0].message.tool_calls = None
    text_resp.cost = 0.0
    text_resp.usage.total_tokens = 200

    side_effects = [tool_call_resp, text_resp]

    from coverage_agent.config import AgentConfig
    cfg = AgentConfig(max_tool_calls=5)

    with patch("litellm.completion", side_effect=side_effects), \
         patch("coverage_agent.engine.tools.dispatch", return_value="# some source") as mock_dispatch:
        draft = TestWriter(creds).run(sample_gap, sample_context, config=cfg)

    mock_dispatch.assert_called_once()
    assert "def test_" in draft.test_code or "class Test" in draft.test_code


# ---------------------------------------------------------------------------
# Multi-arc cluster prompt content
# ---------------------------------------------------------------------------

def _make_gap(symbol: str, from_l: int, to_l: int) -> CoverageGap:
    return CoverageGap(
        file_path="pkg/stats.py",
        target_symbol=symbol,
        branch=BranchGap(from_line=from_l, to_line=to_l),
        surrounding_lines=list(range(from_l, to_l + 3)),
        kind="branch",
        origin="full",
        gap_id=f"pkg/stats.py:{from_l}->{to_l}",
    )


def _make_context() -> ContextPayload:
    return ContextPayload(
        primary_code="def letter_grade(score):\n    if score >= 90:\n        return 'A'\n",
        dependencies_code={},
        graph_depth_used=1,
        tokens_used=80,
    )


def test_single_gap_prompt_unchanged(sample_gap, sample_context):
    """Single-gap path produces the same singular wording as before clustering."""
    prompt = _build_user_prompt(sample_gap, sample_context, critique=None, react_mode=False, cluster=None)
    assert "Write 1-2 pytest test functions" in prompt
    assert "Uncovered branch:" in prompt


def test_multi_arc_prompt_lists_all_arcs():
    """Cluster prompt names the target symbol and lists every arc."""
    g1 = _make_gap("letter_grade", 35, 37)
    g2 = _make_gap("letter_grade", 37, 38)
    g3 = _make_gap("letter_grade", 37, 40)
    ctx = _make_context()

    prompt = _build_user_prompt(g1, ctx, critique=None, react_mode=False, cluster=[g1, g2, g3])

    assert "letter_grade" in prompt
    assert "line 35 -> line 37" in prompt
    assert "line 37 -> line 38" in prompt
    assert "line 37 -> line 40" in prompt
    # The cluster heading replaces the single-gap wording.
    assert "ALL" in prompt
    assert "Write 1-2 pytest test functions" not in prompt


def test_single_element_cluster_behaves_as_single_gap():
    """A cluster of 1 is treated identically to the no-cluster path."""
    g1 = _make_gap("compute", 10, 12)
    ctx = _make_context()
    prompt_no_cluster = _build_user_prompt(g1, ctx, critique=None, react_mode=False, cluster=None)
    prompt_single = _build_user_prompt(g1, ctx, critique=None, react_mode=False, cluster=[g1])
    assert prompt_no_cluster == prompt_single


def test_critique_included_in_retry_prompt(creds, sample_gap, sample_context):
    captured_prompts = []

    def _capture(*args, **kwargs):
        # Copy so mutations after the call don't affect the captured snapshot.
        captured_prompts.append(list(kwargs.get("messages", [])))
        return _fake_completion()

    with patch("litellm.completion", side_effect=_capture):
        TestWriter(creds).run(sample_gap, sample_context, critique="Fix your imports.")

    first_call_messages = captured_prompts[0]
    user_msgs = [m["content"] for m in first_call_messages if m.get("role") == "user"]
    assert user_msgs, "No user message found in first LLM call"
    user_msg = user_msgs[0]
    assert "RETRY" in user_msg
    assert "Fix your imports." in user_msg
