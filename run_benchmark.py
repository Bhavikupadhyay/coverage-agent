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


def _ensure_env(dry_run: bool) -> None:
    if Path(".env").exists():
        _load_dotenv(".env")
        return
    if dry_run or os.environ.get("DRY_RUN", "false").lower() == "true":
        return
    print("No .env found. Required API keys:")
    print("  GEMINI_API_KEY   — https://aistudio.google.com/app/apikey")
    print("  E2B_API_KEY      — https://e2b.dev/dashboard")
    print("  BRAINTRUST_API_KEY — https://braintrust.dev/app/settings")
    keys = {}
    for key in ("GEMINI_API_KEY", "E2B_API_KEY", "BRAINTRUST_API_KEY"):
        val = input(f"  Enter {key}: ").strip()
        keys[key] = val
        os.environ[key] = val
    with open(".env", "w") as f:
        for k, v in keys.items():
            f.write(f"{k}={v}\n")
    print(".env written.")


_llm_cost_accumulator: dict[str, float] = {"total": 0.0}


def _setup_cost_tracking() -> None:
    try:
        import litellm

        # Retry up to 6 times on rate limit / transient errors with exponential backoff
        litellm.num_retries = 6

        def _cost_callback(kwargs, response, start_time, end_time):
            try:
                cost = litellm.completion_cost(completion_response=response)
                _llm_cost_accumulator["total"] += cost
            except Exception:
                pass

        litellm.success_callback = [_cost_callback]
    except ImportError:
        pass


def _print_scorecard(scorecard: dict) -> None:
    print()
    print(f"Repo:            {scorecard['repo']}")
    print(f"Gaps targeted:   {scorecard['gaps_targeted']}")
    print(f"Tests committed: {scorecard['tests_committed']}")
    print(f"Skipped:         {scorecard['skipped']}")
    print(f"Branch hit rate: {scorecard['branch_hit_rate']}")
    print(f"Avg coverage Δ:  {scorecard['avg_coverage_delta']}")
    print(f"Avg loops:       {scorecard['avg_loops']}")
    print(f"LLM cost:        {scorecard['llm_cost']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CoverageAgent benchmark on a repo.")
    parser.add_argument("--repo", required=True, help="Local path or git URL of the repo to analyze")
    parser.add_argument("--max-gaps", type=int, default=10, help="Maximum number of gaps to target (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run mode (no real API calls)")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )

    # Load .env before evaluating dry_run so DRY_RUN in .env is respected,
    # but --dry-run flag already set above takes precedence via setdefault.
    _ensure_env(args.dry_run)

    # Recompute after .env is loaded
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    _setup_cost_tracking()

    braintrust_logger = None
    if not dry_run:
        from coverage_agent.evals.braintrust_logger import BraintrustLogger
        braintrust_logger = BraintrustLogger(project_name="coverage-agent")

    from coverage_agent.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    scorecard, results = orchestrator.run(
        repo_url_or_path=args.repo,
        max_gaps=args.max_gaps,
        braintrust_logger=braintrust_logger,
    )

    # Inject real LLM cost if tracked
    real_cost = _llm_cost_accumulator["total"]
    if real_cost > 0:
        dry_run_suffix = " (DRY_RUN)" if dry_run else ""
        scorecard["llm_cost"] = f"${real_cost:.4f}{dry_run_suffix}"

    _print_scorecard(scorecard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
