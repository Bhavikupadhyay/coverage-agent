#!/usr/bin/env python3
"""CLI entry point for CoverageAgent benchmarks."""

import argparse
import logging
import os
import sys
from pathlib import Path


def _load_dotenv(path: str) -> None:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _ensure_env(offline: bool) -> None:
    if Path(".env").exists():
        _load_dotenv(".env")
        return
    if offline or os.environ.get("OFFLINE_MODE", "false").lower() == "true":
        return
    print("No .env found. Required API keys:")
    print("  GROQ_API_KEY        — https://console.groq.com/keys  (free, no card)")
    print("  E2B_API_KEY         — https://e2b.dev/dashboard")
    print("  BRAINTRUST_API_KEY  — optional, https://braintrust.dev/app/settings")
    keys = {}
    for key in ("GROQ_API_KEY", "E2B_API_KEY", "BRAINTRUST_API_KEY"):
        val = input(f"  Enter {key} (blank to skip): ").strip()
        keys[key] = val
        if val:
            os.environ[key] = val
    with open(".env", "w") as f:
        for k, v in keys.items():
            f.write(f"{k}={v}\n")
    print(".env written.")


def _print_scorecard(scorecard: dict, header: str | None = None) -> None:
    print()
    label = header or "Scorecard"
    print(f"─── {label} " + "─" * max(0, 50 - len(label)))
    print(f"Repo:            {scorecard['repo']}")
    if "mode" in scorecard:
        print(f"Mode:            {scorecard['mode']}")
    print(f"Gaps targeted:   {scorecard['gaps_targeted']}")
    print(f"Tests committed: {scorecard['tests_committed']}")
    print(f"Skipped:         {scorecard['skipped']}")
    print(f"Branch hit rate: {scorecard['branch_hit_rate']}")
    if "tests_passed_no_branch" in scorecard:
        print(f"Passed, no branch: {scorecard['tests_passed_no_branch']}")
    print(f"Avg coverage Δ:  {scorecard['avg_coverage_delta']}")
    print(f"Avg loops:       {scorecard['avg_loops']}")
    print(f"LLM cost:        {scorecard['llm_cost']}")
    print()


def _print_compare_table(pipeline_sc: dict, naive_sc: dict) -> None:
    print()
    print("─── Compare: pipeline vs naive " + "─" * 30)
    rows = [
        ("Tests committed", pipeline_sc["tests_committed"], naive_sc["tests_committed"]),
        ("Branch hit rate", pipeline_sc["branch_hit_rate"], naive_sc["branch_hit_rate"]),
        ("Green, no branch", pipeline_sc.get("tests_passed_no_branch", "—"), naive_sc.get("tests_passed_no_branch", "—")),
        ("Avg coverage Δ ", pipeline_sc["avg_coverage_delta"], naive_sc["avg_coverage_delta"]),
        ("Avg loops      ", pipeline_sc["avg_loops"], naive_sc["avg_loops"]),
        ("LLM cost       ", pipeline_sc["llm_cost"], naive_sc["llm_cost"]),
    ]
    print(f"{'Metric':<18} {'pipeline':>15} {'naive':>15}")
    print("─" * 50)
    for label, p, n in rows:
        print(f"{label:<18} {str(p):>15} {str(n):>15}")
    # The number we care about most. If pipeline doesn't beat naive on commits,
    # the multi-agent story is unjustified.
    delta = int(pipeline_sc["tests_committed"]) - int(naive_sc["tests_committed"])
    verdict = (
        "pipeline WINS" if delta > 0
        else "TIE" if delta == 0
        else "pipeline LOSES — multi-agent overhead unjustified on this repo"
    )
    print("─" * 50)
    print(f"Commit delta:    pipeline {delta:+d} vs naive   →  {verdict}")
    print()


def _gap_status(r) -> tuple[str, str]:
    """Returns (icon, label) for one GapResult."""
    if r.final_test_committed:
        return "✓", "COMMITTED"
    if r.phase2_scores and r.phase2_scores.execution_success and not r.phase2_scores.target_branch_hit:
        return "△", "EXECUTED — missed target branch"
    if r.phase2_scores and not r.phase2_scores.execution_success:
        return "✗", "FAILED — test crashed in sandbox"
    if r.skipped:
        return "✗", "SKIPPED — eval loop exhausted"
    return "?", "UNKNOWN"


def _print_gap_report(results: list) -> None:
    if not results:
        print("(no gaps to report)")
        return
    print("─── Per-gap report " + "─" * 50)
    for i, r in enumerate(results, 1):
        gap = r.gap
        icon, label = _gap_status(r)
        print()
        print(f"[{i}] {icon} {label}")
        print(f"    File:   {gap.file_path}")
        print(f"    Symbol: {gap.target_symbol}")
        print(f"    Branch: line {gap.branch.from_line} → {gap.branch.to_line}")
        print(f"    Loops:  {r.loops_taken}")
        if r.phase1_scores:
            p1 = r.phase1_scores
            print(f"    Eval:   syntax={p1.syntax_valid} assertion={p1.assertion_score}/5 route={p1.route}")
        if r.phase2_scores:
            p2 = r.phase2_scores
            print(f"    Sandbox: success={p2.execution_success} branch_hit={p2.target_branch_hit} Δ={p2.coverage_delta:+.2f}%")
        if r.skip_reason:
            print(f"    Why:    {r.skip_reason}")
        if r.recommendation:
            print(f"    Next:   {r.recommendation}")
        if r.test_code:
            preview = "\n".join("        " + ln for ln in r.test_code.strip().splitlines()[:8])
            print("    Last draft (first 8 lines):")
            print(preview)
            if r.test_code.count("\n") > 8:
                print("        ...")
    print()


def _result_to_dict(r) -> dict:
    """Serializes a GapResult into a JSON-friendly dict for benchmark output."""
    return {
        "gap_id": r.gap.gap_id,
        "file_path": r.gap.file_path,
        "target_symbol": r.gap.target_symbol,
        "branch_from": r.gap.branch.from_line,
        "branch_to": r.gap.branch.to_line,
        "priority_score": r.gap.priority_score,
        "status": _gap_status(r)[1],
        "committed": r.final_test_committed,
        "skipped": r.skipped,
        "loops_taken": r.loops_taken,
        "skip_reason": r.skip_reason,
        "recommendation": r.recommendation or "",
        "phase1_eval": r.phase1_scores.model_dump() if r.phase1_scores else None,
        "phase2_execution": r.phase2_scores.model_dump() if r.phase2_scores else None,
        "test_code": r.test_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CoverageAgent benchmark on a repo.")
    parser.add_argument("--repo", required=True, help="Local path or git URL of the repo to analyze")
    parser.add_argument("--max-gaps", type=int, default=10, help="Maximum number of gaps to target (default: 10)")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (no real LLM/E2B calls; uses fixtures)")
    parser.add_argument(
        "--strictness",
        choices=["strict", "balanced", "loose"],
        default="balanced",
        help="Retry budget only: loose=1 loop, balanced/strict=3. Commits always require branch proof.",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "naive", "compare"],
        default="pipeline",
        help=(
            "Which runner to use. `pipeline` is the multi-agent loop. "
            "`naive` is a one-shot LLM prompt with no retries — the apples-to-apples baseline. "
            "`compare` runs both and prints a side-by-side table (uses 2x the budget)."
        ),
    )
    parser.add_argument(
        "--sandbox",
        choices=["local", "e2b"],
        default="local",
        help=(
            "Execution backend. `local` runs in a subprocess+venv on this machine "
            "(free, no external APIs, correct for benchmarks on trusted repos). "
            "`e2b` spins up a cloud VM (requires E2B_API_KEY). Default: local."
        ),
    )
    parser.add_argument(
        "--ignore-file",
        dest="ignore_file",
        default=None,
        help=(
            "Path to a gitignore-style file listing paths/globs to exclude from gap analysis. "
            "Each line is a pattern (supports fnmatch globs, trailing / for directories). "
            "Lines starting with # are comments. Repos can also ship a .coverageagentignore "
            "in their root for automatic discovery."
        ),
    )
    parser.add_argument("--output", help="Write the scorecard JSON to this path after the run")
    args = parser.parse_args()

    if args.offline:
        os.environ["OFFLINE_MODE"] = "true"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )

    _ensure_env(args.offline)

    offline = os.environ.get("OFFLINE_MODE", "false").lower() == "true"

    from coverage_agent.baselines.naive_single_shot import NaiveSingleShotRunner
    from coverage_agent.credentials import Credentials
    from coverage_agent.orchestrator import Orchestrator

    os.environ.setdefault("EVAL_STRICTNESS", args.strictness)
    creds = Credentials.for_offline() if offline else Credentials.for_cli_env(sandbox_mode=args.sandbox)

    braintrust_logger = None
    if not offline and creds.braintrust_api_key:
        from coverage_agent.evals.braintrust_logger import BraintrustLogger
        braintrust_logger = BraintrustLogger(
            project_name="coverage-agent",
            api_key=creds.braintrust_api_key,
            model=creds.llm_model,
        )

    def _run_pipeline():
        return Orchestrator(credentials=creds).run(
            repo_url_or_path=args.repo,
            max_gaps=args.max_gaps,
            braintrust_logger=braintrust_logger,
            ignore_file=args.ignore_file,
        )

    def _run_naive():
        return NaiveSingleShotRunner(credentials=creds).run(
            repo_url_or_path=args.repo,
            max_gaps=args.max_gaps,
        )

    payload: dict = {"mode": args.mode}

    if args.mode == "pipeline":
        scorecard, results = _run_pipeline()
        _print_scorecard(scorecard, header="Scorecard (pipeline)")
        _print_gap_report(results)
        payload["pipeline"] = {
            "scorecard": scorecard,
            "results": [_result_to_dict(r) for r in results],
        }
    elif args.mode == "naive":
        scorecard, results = _run_naive()
        _print_scorecard(scorecard, header="Scorecard (naive single-shot)")
        _print_gap_report(results)
        payload["naive"] = {
            "scorecard": scorecard,
            "results": [_result_to_dict(r) for r in results],
        }
    else:  # compare
        pipeline_sc, pipeline_results = _run_pipeline()
        naive_sc, naive_results = _run_naive()
        _print_scorecard(pipeline_sc, header="Scorecard (pipeline)")
        _print_gap_report(pipeline_results)
        _print_scorecard(naive_sc, header="Scorecard (naive single-shot)")
        _print_gap_report(naive_results)
        _print_compare_table(pipeline_sc, naive_sc)
        payload["pipeline"] = {
            "scorecard": pipeline_sc,
            "results": [_result_to_dict(r) for r in pipeline_results],
        }
        payload["naive"] = {
            "scorecard": naive_sc,
            "results": [_result_to_dict(r) for r in naive_results],
        }

    if args.output:
        import json
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Full report written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
