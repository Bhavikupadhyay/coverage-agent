"""Acceptance harness for the coverage-agent verification suite.

Two modes:
  --mock-llm   Patches litellm.completion with canned responses. Zero API keys
               required. Used by CI on every push.
  (no flag)    Real-key mode. Same harness, same assertions, credentials from
               Credentials.for_cli_env(). Run on key day only — do not invoke
               until the .env is in place.

Flow:
  1. Copy benchmarks/fixture_repo to a temp dir.
  2. git init + initial commit on main in the temp dir.
  3. Create a 'add-scoring' branch that adds mathlib/scoring.py (the new-file case).
  4. Run coverage baseline on main (the initial test suite).
  5. Check out the diff branch.
  6. Invoke the engine for --scope diff --base main.
  7. Check out main, re-run coverage (same baseline), invoke for --scope full.
  8. Assert gap counts and accepted counts; verify fixture suite still green.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.resolve()
_FIXTURE_SRC = Path(__file__).parent / "fixture_repo"

# ---------------------------------------------------------------------------
# Known gap counts
# ---------------------------------------------------------------------------

# Full scope: all missing branch arcs in stats.py from the initial test suite.
# Verified by: cd benchmarks/fixture_repo && coverage run --branch -m pytest tests/
# then: coverage json -o /tmp/c.json && python -c "import json; d=json.load(open('/tmp/c.json')); print(d['files']['mathlib/stats.py']['missing_branches'])"
# Result: [[11,12],[13,14],[24,25],[35,37],[37,38],[37,40],[49,51],[53,54],[53,55]]
EXPECTED_FULL_GAPS = 9

# Diff scope: 3 public functions in scoring.py → 3 function gaps.
# Verified by AST: percentage (5->9), pass_fail (12->16), weighted_average (19->26).
EXPECTED_DIFF_GAPS = 3

# Minimum tests that must be accepted for the full-scope run.
# Branch gaps in stats.py are genuine and the canned tests hit all 9 arcs.
REQUIRED_ACCEPTED_FULL = 2

# ---------------------------------------------------------------------------
# Canned test code
# ---------------------------------------------------------------------------

# These tests genuinely cover the 9 missing branch arcs in stats.py.
# Verification: cd benchmarks/fixture_repo && coverage run --branch --data-file=.c
#   -m pytest /tmp/test_canned.py -q && python -c "
#   import coverage as c; cov=c.Coverage(data_file='.c'); cov.load(); data=cov.get_data()
#   arcs=set(data.arcs('<abs_path>/mathlib/stats.py') or [])
#   for a in [(11,12),(13,14),(24,25),(35,37),(37,38),(37,40),(49,51),(53,54),(53,55)]:
#       print(a, 'HIT' if a in arcs else 'MISS')"
# All 9 print HIT.
_CANNED_STATS_TESTS = '''\
"""Generated tests that cover the uncovered branches in mathlib/stats.py."""
from mathlib.stats import clamp, safe_divide, letter_grade, normalize


def test_clamp_below_lo():
    assert clamp(-1, 0, 10) == 0


def test_clamp_above_hi():
    assert clamp(15, 0, 10) == 10


def test_safe_divide_zero():
    assert safe_divide(5, 0) == 0.0


def test_letter_grade_b():
    assert letter_grade(85) == "B"


def test_letter_grade_c():
    assert letter_grade(70) == "C"


def test_normalize_uniform():
    result = normalize([3.0, 3.0, 3.0])
    assert result == [0.0, 0.0, 0.0]


def test_normalize_range():
    result = normalize([0.0, 5.0, 10.0])
    assert result == [0.0, 0.5, 1.0]
'''

# Canned test for new-file scoring gaps (function kind — tests pass pytest,
# but function-span arc check won't flip target_branch_hit in the executor;
# these are included so the writer produces valid, runnable code).
_CANNED_SCORING_TESTS = '''\
"""Generated tests for mathlib/scoring.py (new file, diff scope)."""
from mathlib.scoring import percentage, pass_fail, weighted_average


def test_percentage_normal():
    assert percentage(8, 10) == 80.0


def test_pass_fail_pass():
    assert pass_fail(70.0) == "pass"


def test_weighted_average_basic():
    result = weighted_average([80.0, 90.0], [1.0, 1.0])
    assert result == 85.0
'''


def _make_mock_completion(test_code: str):
    """Returns a litellm.completion side-effect function that emits test_code."""
    def _mock(*args, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = f"```python\n{test_code}\n```"
        resp.choices[0].message.tool_calls = None
        resp.cost = 0.0
        resp.usage.total_tokens = 100
        return resp
    return _mock


# ---------------------------------------------------------------------------
# Fixture repo setup
# ---------------------------------------------------------------------------

def _setup_fixture_repo(tmp_dir: Path) -> None:
    """Copies fixture_repo into tmp_dir and git-inits it."""
    shutil.copytree(_FIXTURE_SRC, tmp_dir, dirs_exist_ok=True)

    def _git(*args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")

    _git("init", "-b", "main")
    _git("config", "user.email", "harness@coverage-agent.local")
    _git("config", "user.name", "Acceptance Harness")
    # Remove scoring.py from the initial commit — it belongs on the diff branch.
    scoring_in_tmp = tmp_dir / "mathlib" / "scoring.py"
    if scoring_in_tmp.exists():
        scoring_in_tmp.unlink()
    _git("add", ".")
    _git("commit", "-m", "chore: initial fixture commit")


def _create_diff_branch(tmp_dir: Path) -> None:
    """Creates the 'add-scoring' branch with mathlib/scoring.py added."""
    scoring_src = _FIXTURE_SRC / "mathlib" / "scoring.py"
    scoring_dst = tmp_dir / "mathlib" / "scoring.py"

    def _git(*args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")

    _git("checkout", "-b", "add-scoring")
    shutil.copy(scoring_src, scoring_dst)
    _git("add", "mathlib/scoring.py")
    _git("commit", "-m", "feat: add scoring utilities")


def _checkout(tmp_dir: Path, ref: str) -> None:
    result = subprocess.run(
        ["git", "checkout", ref],
        cwd=str(tmp_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout {ref} failed:\n{result.stderr}")


def _run_baseline_coverage(tmp_dir: Path) -> str:
    """Runs coverage on the fixture test suite; returns path to .coverage file."""
    cov_path = str(tmp_dir / ".coverage")
    result = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            "--branch",
            f"--data-file={cov_path}",
            "-m", "pytest", "tests/", "-q",
        ],
        cwd=str(tmp_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_dir)},
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"coverage run failed (rc={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return cov_path


# ---------------------------------------------------------------------------
# Engine invocation
# ---------------------------------------------------------------------------

def _invoke_engine(
    tmp_dir: Path,
    cov_path: str,
    scope: str,
    base_ref: str,
    mock_llm: bool,
    real_creds=None,
) -> tuple[int, int]:
    """Runs the engine pipeline for all selected gaps. Returns (gaps_found, tests_accepted)."""
    sys.path.insert(0, str(_REPO_ROOT))

    from coverage_agent.config import AgentConfig
    from coverage_agent.credentials import Credentials
    from coverage_agent.gaps.coverage_data import load_coverage_file, parse_coverage
    from coverage_agent.gaps.select import select_gaps
    from coverage_agent.gaps.diff import compute_diff_gaps
    from coverage_agent.engine.graph import run_pipeline

    cfg = AgentConfig(
        scope=scope,
        max_gaps=20,
        max_retries=1,
        max_tool_calls=0,   # single-shot: no ReAct tool loop in mock mode
        flaky_runs=1,       # one run; fixture is deterministic
        test_timeout=30,
        tests_dir="tests/generated",
    )

    if mock_llm:
        creds = Credentials(llm_api_key="mock-key", llm_model="groq/llama-3.3-70b-versatile")
    else:
        creds = real_creds

    coverage_data = load_coverage_file(cov_path)

    repo_root = str(tmp_dir)

    if scope == "diff":
        all_gaps = compute_diff_gaps(
            coverage_data,
            repo_root=repo_root,
            base_ref=base_ref,
            exclude=[],
        )
    else:
        all_gaps = parse_coverage(coverage_data, repo_root=repo_root)

    gaps_found = len(all_gaps)
    candidate_gaps = select_gaps(all_gaps, max_gaps=cfg.max_gaps, exclude=[])

    if not candidate_gaps:
        return gaps_found, 0

    # Determine which canned test to use per gap.
    def _canned_for_gap(gap) -> str:
        if "scoring" in gap.file_path:
            return _CANNED_SCORING_TESTS
        return _CANNED_STATS_TESTS

    accepted_count = 0
    old_cwd = os.getcwd()
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    try:
        os.chdir(str(tmp_dir))
        os.environ["PYTHONPATH"] = str(tmp_dir)

        for gap in candidate_gaps:
            canned = _canned_for_gap(gap)
            mock_fn = _make_mock_completion(canned) if mock_llm else None

            ctx_mgr = patch("litellm.completion", side_effect=mock_fn) if mock_llm else _noop_ctx()

            with ctx_mgr:
                try:
                    gap_result, _ = run_pipeline(
                        gap=gap,
                        credentials=creds,
                        config=cfg,
                        baseline_coverage=coverage_data,
                        repo_path=repo_root,
                    )
                except Exception as exc:
                    print(f"  [warning] gap {gap.gap_id} pipeline error: {exc}", file=sys.stderr)
                    continue

            if gap_result.accepted:
                accepted_count += 1

    finally:
        os.chdir(old_cwd)
        if old_pythonpath:
            os.environ["PYTHONPATH"] = old_pythonpath
        elif "PYTHONPATH" in os.environ:
            del os.environ["PYTHONPATH"]

    return gaps_found, accepted_count


class _noop_ctx:
    """No-op context manager for real-key mode (no patching)."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


# ---------------------------------------------------------------------------
# Post-run fixture suite check
# ---------------------------------------------------------------------------

def _assert_fixture_suite_green(tmp_dir: Path) -> None:
    """Runs the fixture's own tests (plus any generated tests) and asserts all pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=str(tmp_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_dir)},
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fixture suite not green after engine run (rc={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    print(f"  fixture suite: green\n    {result.stdout.strip().splitlines()[-1]}")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-agent acceptance harness")
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Patch litellm.completion with canned responses. No API keys required.",
    )
    args = parser.parse_args()

    real_creds = None
    if not args.mock_llm:
        # Import here so missing-key errors are surfaced immediately.
        sys.path.insert(0, str(_REPO_ROOT))
        from coverage_agent.credentials import Credentials
        try:
            real_creds = Credentials.for_cli_env()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="coverage-agent-fixture-") as tmp_str:
        tmp_dir = Path(tmp_str)

        print(f"[harness] fixture dir: {tmp_dir}")
        _setup_fixture_repo(tmp_dir)
        _create_diff_branch(tmp_dir)

        # ---- Diff scope (on the add-scoring branch) ----
        print("\n[harness] scope=diff --base main")
        # Stay on add-scoring branch; baseline coverage was run on main.
        _checkout(tmp_dir, "main")
        cov_path_main = _run_baseline_coverage(tmp_dir)
        _checkout(tmp_dir, "add-scoring")

        diff_found, diff_accepted = _invoke_engine(
            tmp_dir=tmp_dir,
            cov_path=cov_path_main,
            scope="diff",
            base_ref="main",
            mock_llm=args.mock_llm,
            real_creds=real_creds,
        )
        print(f"  gaps_found={diff_found}  tests_accepted={diff_accepted}")

        if diff_found != EXPECTED_DIFF_GAPS:
            failures.append(
                f"diff scope: expected gaps_found={EXPECTED_DIFF_GAPS}, got {diff_found}"
            )

        # ---- Full scope (on main) ----
        print("\n[harness] scope=full")
        _checkout(tmp_dir, "main")
        cov_path_full = _run_baseline_coverage(tmp_dir)

        full_found, full_accepted = _invoke_engine(
            tmp_dir=tmp_dir,
            cov_path=cov_path_full,
            scope="full",
            base_ref="",
            mock_llm=args.mock_llm,
            real_creds=real_creds,
        )
        print(f"  gaps_found={full_found}  tests_accepted={full_accepted}")

        if full_found != EXPECTED_FULL_GAPS:
            failures.append(
                f"full scope: expected gaps_found={EXPECTED_FULL_GAPS}, got {full_found}"
            )
        if full_accepted < REQUIRED_ACCEPTED_FULL:
            failures.append(
                f"full scope: expected tests_accepted>={REQUIRED_ACCEPTED_FULL}, got {full_accepted}"
            )

        # ---- Fixture suite still green (on main, with generated tests present) ----
        print("\n[harness] fixture suite regression check")
        try:
            _assert_fixture_suite_green(tmp_dir)
        except AssertionError as exc:
            failures.append(str(exc))

    if failures:
        print("\n[harness] FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        sys.exit(1)

    print("\n[harness] PASSED — all assertions satisfied")


if __name__ == "__main__":
    main()
