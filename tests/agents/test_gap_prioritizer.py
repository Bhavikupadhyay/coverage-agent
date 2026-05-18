"""GapPrioritizer offline + deterministic heuristic (Phase D).

The default scoring path is the deterministic heuristic, which is the only
path tested here. The LLM-ranking path is opt-in via Credentials.prioritize_with_llm
and is exercised by a separate live-mode integration test elsewhere.
"""
from unittest.mock import patch

from coverage_agent.agents.gap_prioritizer import GapPrioritizer, _score_gap
from coverage_agent.contracts.schemas import BranchGap, CoverageGap
from coverage_agent.credentials import Credentials


def _make_gap(idx: int, **overrides) -> CoverageGap:
    defaults = dict(
        file_path=f"pkg/mod{idx}.py",
        target_symbol=f"fn{idx}",
        branch=BranchGap(from_line=idx, to_line=idx + 1),
        surrounding_lines=[idx, idx + 1, idx + 2],
        priority_score=0.0,
        gap_id=f"pkg/mod{idx}.py:{idx}->{idx + 1}",
    )
    defaults.update(overrides)
    return CoverageGap(**defaults)


# ---------------------------------------------------------------------------
# Offline / empty input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(offline_creds):
    result = GapPrioritizer(offline_creds).run([])
    assert result == []


def test_offline_assigns_descending_scores(offline_creds):
    gaps = [_make_gap(i) for i in range(3)]
    result = GapPrioritizer(offline_creds).run(gaps)
    assert len(result) == 3
    scores = [g.priority_score for g in result]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_preserves_gap_identity(offline_creds):
    gaps = [_make_gap(0)]
    [scored] = GapPrioritizer(offline_creds).run(gaps)
    assert scored.gap_id == gaps[0].gap_id
    assert scored.file_path == gaps[0].file_path
    assert scored.branch == gaps[0].branch


# ---------------------------------------------------------------------------
# Deterministic heuristic — file path signals
# ---------------------------------------------------------------------------

def test_init_py_files_score_lower_than_regular_modules():
    init_gap = _make_gap(0, file_path="pkg/__init__.py", target_symbol="setup_thing")
    regular = _make_gap(1, file_path="pkg/core.py", target_symbol="process_payment",
                        surrounding_lines=list(range(10)))
    assert _score_gap(init_gap) < _score_gap(regular)


def test_conftest_py_scores_lower_than_regular_modules():
    conftest = _make_gap(0, file_path="tests/conftest.py", target_symbol="some_fixture")
    regular = _make_gap(1, file_path="pkg/business_logic.py", target_symbol="handle_request",
                       surrounding_lines=list(range(10)))
    assert _score_gap(conftest) < _score_gap(regular)


# ---------------------------------------------------------------------------
# Deterministic heuristic — symbol name signals
# ---------------------------------------------------------------------------

def test_logging_functions_score_lower_than_logic_functions():
    log_gap = _make_gap(0, target_symbol="log_event", surrounding_lines=list(range(10)))
    logic_gap = _make_gap(1, target_symbol="calculate_total", surrounding_lines=list(range(10)))
    assert _score_gap(log_gap) < _score_gap(logic_gap)


def test_dunder_methods_score_lower_than_named_methods():
    dunder = _make_gap(0, target_symbol="__init__", surrounding_lines=list(range(10)))
    named = _make_gap(1, target_symbol="serialize", surrounding_lines=list(range(10)))
    assert _score_gap(dunder) < _score_gap(named)


# ---------------------------------------------------------------------------
# Deterministic heuristic — gap size signals
# ---------------------------------------------------------------------------

def test_sweet_spot_size_scores_highest():
    """The 4-30 line sweet spot beats both tiny and huge gaps."""
    tiny = _make_gap(0, surrounding_lines=[1])
    sweet = _make_gap(1, surrounding_lines=list(range(15)))
    huge = _make_gap(2, surrounding_lines=list(range(120)))
    assert _score_gap(sweet) > _score_gap(tiny)
    assert _score_gap(sweet) > _score_gap(huge)


def test_score_bounded_to_unit_interval():
    """Even a maximally-bad gap stays >= 0.0 and a maximally-good one <= 1.0."""
    worst = _make_gap(0, file_path="pkg/__init__.py", target_symbol="log_setup_thing",
                     surrounding_lines=list(range(200)))
    best = _make_gap(1, file_path="pkg/core.py", target_symbol="process_payment",
                    surrounding_lines=list(range(15)))
    assert 0.0 <= _score_gap(worst) <= 1.0
    assert 0.0 <= _score_gap(best) <= 1.0


# ---------------------------------------------------------------------------
# LLM toggle: opt-in only
# ---------------------------------------------------------------------------

def test_default_byok_makes_zero_llm_calls():
    """Heuristic is default. Without prioritize_with_llm=True, no LLM is hit."""
    creds = Credentials(mode="byok", llm_api_key="gsk_x", llm_model="groq/llama-3.3-70b-versatile")
    assert creds.prioritize_with_llm is False  # default
    gaps = [_make_gap(i) for i in range(5)]
    with patch("litellm.completion") as mock_completion:
        result = GapPrioritizer(creds).run(gaps)
    mock_completion.assert_not_called()
    assert len(result) == 5
    assert result == sorted(result, key=lambda g: g.priority_score, reverse=True)


def test_llm_toggle_opt_in_uses_llm_path():
    """Flipping the toggle enables the original LLM scoring path."""
    creds = Credentials(
        mode="byok",
        llm_api_key="gsk_x",
        llm_model="groq/llama-3.3-70b-versatile",
        prioritize_with_llm=True,
    )
    gaps = [_make_gap(i) for i in range(3)]
    fake_response = type("R", (), {
        "choices": [type("C", (), {
            "message": type("M", (), {"content": "[0.9, 0.5, 0.1]"})()
        })()],
    })()
    with patch("litellm.completion", return_value=fake_response) as mock_completion:
        result = GapPrioritizer(creds).run(gaps)
    mock_completion.assert_called_once()
    # Scores should reflect the LLM's response, sorted desc
    assert [g.priority_score for g in result] == [0.9, 0.5, 0.1]

