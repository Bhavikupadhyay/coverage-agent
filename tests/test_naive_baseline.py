"""Phase E: naive single-shot baseline tests.

Same shape as the pipeline so run_benchmark.py compare-mode can diff them.
"""
from __future__ import annotations

from coverage_agent.baselines.naive_single_shot import (
    NaiveSingleShotRunner,
    _read_function_source,
)
from coverage_agent.credentials import Credentials


# ---------------------------------------------------------------------------
# Source extraction: needle in a haystack
# ---------------------------------------------------------------------------

def test_read_function_source_extracts_target_def(tmp_path):
    """Pulls out exactly the target function from a file with multiple defs."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "def other_func():\n"
        "    return 'other'\n"
        "\n"
        "def target_func(x):\n"
        "    if x > 0:\n"
        "        return 'positive'\n"
        "    return 'non-positive'\n"
        "\n"
        "def third_func():\n"
        "    pass\n",
        encoding="utf-8",
    )
    source = _read_function_source(str(tmp_path), "pkg/mod.py", "target_func")
    assert "def target_func(x):" in source
    assert "if x > 0:" in source
    # Doesn't bleed into other_func or third_func
    assert "def other_func" not in source
    assert "def third_func" not in source


def test_read_function_source_falls_back_to_whole_file_on_miss(tmp_path):
    """If we can't locate the function, return the whole file. Better than nothing."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    contents = "def some_other():\n    pass\n"
    (pkg / "mod.py").write_text(contents, encoding="utf-8")
    source = _read_function_source(str(tmp_path), "pkg/mod.py", "missing_func")
    assert source == contents


def test_read_function_source_handles_missing_file_gracefully(tmp_path):
    """Returns a comment rather than raising. Naive baseline never blocks the benchmark."""
    source = _read_function_source(str(tmp_path), "no/such/file.py", "fn")
    assert "Could not read" in source


# ---------------------------------------------------------------------------
# End-to-end offline behaviour: scorecard shape + commit predicate use
# ---------------------------------------------------------------------------

def test_naive_runner_offline_returns_full_scorecard(tmp_path):
    creds = Credentials.for_offline()
    runner = NaiveSingleShotRunner(creds)
    scorecard, results = runner.run(repo_url_or_path=str(tmp_path), max_gaps=2)

    assert scorecard["mode"] == "naive"
    assert scorecard["gaps_targeted"] >= 0
    assert "tests_committed" in scorecard
    assert "branch_hit_rate" in scorecard
    assert "avg_coverage_delta" in scorecard
    # Pipeline-parity shape so compare-mode can diff them
    assert "regression" in scorecard
    assert "summary" in scorecard

    # phase1 must be None for naive (no eval phase by design)
    for r in results:
        assert r.phase1_scores is None
        assert r.loops_taken == 1  # always 1 by definition


def test_naive_runner_uses_credentials_commit_predicate(tmp_path):
    """The strictness predicate on Credentials is the single source of truth."""
    creds = Credentials.for_offline()
    runner = NaiveSingleShotRunner(creds)
    _, results = runner.run(repo_url_or_path=str(tmp_path), max_gaps=2)
    for r in results:
        expected = creds.should_commit(r.phase2_scores)
        assert r.final_test_committed == expected
