"""
CLI entrypoint — coverage-agent run / models / report / ci-run.

All LLM calls and subprocess execution happen inside the engine; the CLI
is responsible only for wiring config, credentials, gap loading, and output.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from coverage_agent.config import load_config, DEFAULT_MODEL
from coverage_agent.contracts import RunReport
from coverage_agent.credentials import Credentials, list_models

app = typer.Typer(
    name="coverage-agent",
    help="Generate pytest tests that cover uncovered branches.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


@app.command()
def run(
    scope: str = typer.Option("full", help="Gap scope: full or diff"),
    base: str = typer.Option(
        "", help="Diff scope only: git ref to diff against (e.g. origin/main, HEAD~3, a SHA). "
        "Defaults to the merge-base with origin/main or origin/master."
    ),
    max_gaps: int = typer.Option(0, help="Max gaps to target (0 = use config)"),
    coverage_file: str = typer.Option("", help="Path to .coverage, coverage.json, or coverage.xml"),
    model: str = typer.Option("", help="LiteLLM model ID (overrides config/env)"),
    config_path: str = typer.Option("", "--config", help="Path to .coverage-agent.yml"),
    output: str = typer.Option("", help="Write RunReport JSON to this path"),
    verbose: bool = typer.Option(False, "-v", help="Verbose logging"),
) -> None:
    """Run the coverage agent on the current checkout.

    Runs in the current working directory's git checkout. Check out the
    branch or commit you want to target before running; use --scope diff
    with --base to target only the changes since a ref.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(config_path or None)

    # CLI flags override config values when explicitly provided.
    if scope:
        cfg = cfg.model_copy(update={"scope": scope})
    if max_gaps > 0:
        cfg = cfg.model_copy(update={"max_gaps": max_gaps})
    if coverage_file:
        cfg = cfg.model_copy(update={"coverage_file": coverage_file})
    if model:
        cfg = cfg.model_copy(update={"model": model})

    # Override env model so credentials.for_cli_env() picks it up.
    if cfg.model and cfg.model != DEFAULT_MODEL:
        os.environ.setdefault("COVERAGE_AGENT_MODEL", cfg.model)

    try:
        creds = Credentials.for_cli_env()
    except RuntimeError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    # Lazy import to keep startup fast.
    from coverage_agent.gaps.coverage_data import load_coverage_file, parse_coverage
    from coverage_agent.gaps.select import select_gaps
    from coverage_agent.gaps.diff import compute_diff_gaps
    from coverage_agent.engine.graph import run_pipeline
    from coverage_agent.engine.regression import RegressionGuard
    from coverage_agent.report.run_report import serialize_run_report

    _run_pipeline(
        repo_root=str(Path.cwd()),
        cfg=cfg,
        creds=creds,
        coverage_file=coverage_file,
        base_ref=base,
        output=output,
        run_pipeline=run_pipeline,
        select_gaps=select_gaps,
        parse_coverage=parse_coverage,
        compute_diff_gaps=compute_diff_gaps,
        load_coverage_file=load_coverage_file,
        RegressionGuard=RegressionGuard,
    )


def _run_pipeline(
    repo_root: str,
    cfg,
    creds,
    coverage_file: str,
    base_ref: str,
    output: str,
    run_pipeline,
    select_gaps,
    parse_coverage,
    compute_diff_gaps,
    load_coverage_file,
    RegressionGuard,
) -> None:
    import subprocess as _sp

    # ---- Auto-run coverage baseline if no coverage file provided ----
    cov_path = cfg.coverage_file or coverage_file
    if not cov_path:
        for candidate in (".coverage", "coverage.json", "coverage.xml"):
            if Path(repo_root, candidate).exists():
                cov_path = candidate
                break
    if not cov_path:
        console.print("No coverage file found — running [bold]coverage run --branch -m pytest[/bold] …")
        result = _sp.run(
            ["coverage", "run", "--branch", "-m", "pytest", "-q"],
            cwd=repo_root,
        )
        if result.returncode not in (0, 1):
            err_console.print("[red]coverage run failed — cannot continue.[/red]")
            raise typer.Exit(1)
        for candidate in (".coverage", "coverage.json", "coverage.xml"):
            if Path(repo_root, candidate).exists():
                cov_path = candidate
                break

    if not cov_path:
        err_console.print(
            "[red]No coverage file found.[/red] Run your tests with "
            "`coverage run --branch -m pytest` first, or pass --coverage-file."
        )
        raise typer.Exit(1)

    try:
        abs_cov = str(Path(repo_root) / cov_path) if not Path(cov_path).is_absolute() else cov_path
        coverage_data = load_coverage_file(abs_cov)
    except Exception as exc:
        err_console.print(f"[red]Failed to load coverage file:[/red] {exc}")
        raise typer.Exit(1)

    # ---- Select gaps ----
    if cfg.scope == "diff":
        all_gaps = compute_diff_gaps(
            coverage_data, repo_root=repo_root, base_ref=base_ref, exclude=cfg.exclude
        )
    else:
        all_gaps = parse_coverage(coverage_data, repo_root=repo_root, ignore_patterns=cfg.exclude)

    # Pull up to 2× the target so skipped gaps can be replaced from the tail.
    target_count = cfg.max_gaps
    candidate_gaps = select_gaps(all_gaps, max_gaps=target_count * 2, exclude=cfg.exclude)

    console.print(f"[bold]coverage-agent[/bold] scope={cfg.scope} gaps_found={len(all_gaps)} selected={len(candidate_gaps)}")

    if not candidate_gaps:
        console.print("[green]No actionable gaps found — nothing to do.[/green]")
        report = RunReport(
            scope=cfg.scope,
            model=creds.llm_model,
            gaps_found=0,
            gaps_accepted=0,
            tests_accepted=0,
        )
        _write_output(report, output)
        return

    # ---- Run engine per gap (with skip substitution) ----
    gap_results = []
    agent_traces = []
    accepted_count = 0
    attempted = 0

    for gap in candidate_gaps:
        if accepted_count >= target_count or attempted >= target_count * 2:
            break
        attempted += 1
        console.print(f"  [{attempted}] {gap.gap_id} ({gap.kind})")
        try:
            gap_result, final_state = run_pipeline(
                gap=gap,
                credentials=creds,
                config=cfg,
                baseline_coverage=coverage_data,
                repo_path=repo_root,
            )
        except Exception as exc:
            err_console.print(f"  [yellow]Warning:[/yellow] gap {gap.gap_id} failed: {exc}")
            continue
        gap_results.append(gap_result)

        if gap_result.accepted and gap_result.test_code:
            accepted_count += 1
            tests_dir = Path(repo_root) / cfg.tests_dir
            tests_dir.mkdir(parents=True, exist_ok=True)
            from coverage_agent.engine.regression import _filename_for
            fname = tests_dir / _filename_for(gap_result)
            fname.write_text(gap_result.test_code, encoding="utf-8")
            console.print(f"    [green]✓ accepted[/green] → {fname.relative_to(repo_root)}")
        else:
            console.print(f"    [dim]skipped[/dim]")

    accepted = [r for r in gap_results if r.accepted]

    # ---- RegressionGuard ----
    regression = None
    if accepted:
        try:
            regression = RegressionGuard(creds).run(
                config=cfg,
                committed_results=gap_results,
                baseline_passed=0,
                baseline_failed=0,
                repo_root=repo_root,
            )
            status = "[red]REGRESSION[/red]" if regression.regression_detected else "[green]clean[/green]"
            console.print(f"[bold]Regression guard:[/bold] {status} — {regression.summary}")
        except Exception as exc:
            err_console.print(f"[yellow]RegressionGuard failed:[/yellow] {exc}")

    report = RunReport(
        scope=cfg.scope,
        model=creds.llm_model,
        gaps_found=len(all_gaps),
        gaps_accepted=len(accepted),
        tests_accepted=len(accepted),
        gap_results=gap_results,
        regression=regression,
    )
    _write_output(report, output)
    console.print(
        f"\n[bold]Done.[/bold] gaps_found={len(all_gaps)} accepted={len(accepted)}"
    )


@app.command()
def models() -> None:
    """List all models in the registry."""
    registry = list_models()
    table = Table(title="coverage-agent model registry")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Tool calling")
    table.add_column("Free tier")
    table.add_column("USD / M tokens (in/out)")
    for entry in registry:
        pricing = entry.get("pricing_per_mtok", [])
        price_str = f"{pricing[0]}/{pricing[1]}" if len(pricing) == 2 else "—"
        table.add_row(
            entry.get("id", ""),
            entry.get("provider", ""),
            "✓" if entry.get("tool_calling") else "✗",
            "✓" if entry.get("free_tier") else "✗",
            price_str,
        )
    console.print(table)


@app.command()
def report(
    input_path: str = typer.Argument(..., help="Path to a RunReport JSON file"),
) -> None:
    """Pretty-print a RunReport JSON file."""
    try:
        raw = Path(input_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        rr = RunReport(**data)
    except Exception as exc:
        err_console.print(f"[red]Failed to load report:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[bold]RunReport[/bold] scope={rr.scope} model={rr.model}")
    console.print(f"  gaps_found={rr.gaps_found}  accepted={rr.tests_accepted}")
    console.print(f"  cost_usd={rr.total_cost_usd:.4f}")
    if rr.regression:
        status = "REGRESSION" if rr.regression.regression_detected else "clean"
        console.print(f"  regression={status}")
    for gr in rr.gap_results:
        icon = "✓" if gr.accepted else "✗"
        console.print(f"  {icon} {gr.gap.gap_id}")


def _write_output(report: RunReport, path: str) -> None:
    from coverage_agent.report.run_report import serialize_run_report
    json_str = serialize_run_report(report, path or None)
    if not path:
        # Print a compact summary JSON to stdout for piping.
        summary = {
            "gaps_found": report.gaps_found,
            "tests_accepted": report.tests_accepted,
            "scope": report.scope,
        }
        print(json.dumps(summary))


@app.command("ci-run")
def ci_run(
    coverage_file: str = typer.Option("", help="Path to coverage file (.coverage / coverage.json / coverage.xml)"),
    scope: str = typer.Option("diff", help="Gap scope: full or diff"),
    model: str = typer.Option("", help="LiteLLM model ID (overrides config/env)"),
    config_path: str = typer.Option("", "--config", help="Path to .coverage-agent.yml"),
    commit_mode: str = typer.Option(
        "comment",
        help="Delivery mode: comment (default), commit, or pr",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help=(
            "Run the full pipeline but print delivery actions instead of executing them. "
            "Required for CI runs without GITHUB_TOKEN. "
            "NOTE: the Action's CI check uses --preview, so the delivery layer is "
            "exercised with printed output only."
        ),
    ),
    verbose: bool = typer.Option(False, "-v", help="Verbose logging"),
) -> None:
    """CI entrypoint — reads GITHUB_* env vars, runs the pipeline, delivers results.

    Required environment variables:
      GITHUB_REPOSITORY   owner/repo (e.g. acme/myapp)
      GITHUB_BASE_REF     base branch for the PR (e.g. main)
      GITHUB_REF          full ref (e.g. refs/pull/42/merge) — used to extract PR number
      GITHUB_TOKEN        personal access token or ${{ github.token }}

    Optional:
      GITHUB_EVENT_PATH   path to the event JSON (alternative PR number source)
      GITHUB_OUTPUT       path to the GitHub Actions output file

    Git checkout requirement:
      The workflow must fetch enough history for the merge-base to resolve.
      Add `fetch-depth: 0` (or a sufficiently deep fetch) to the checkout step.
      If the merge-base fails, coverage-agent will exit with a copy-pasteable fix:

        - uses: actions/checkout@v4
          with:
            fetch-depth: 0
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Read GITHUB_* env ----
    gh_repo = os.environ.get("GITHUB_REPOSITORY", "")
    gh_base_ref = os.environ.get("GITHUB_BASE_REF", "")
    gh_ref = os.environ.get("GITHUB_REF", "")
    gh_output_file = os.environ.get("GITHUB_OUTPUT", "")
    gh_token = os.environ.get("GITHUB_TOKEN", "")

    # ---- Resolve PR number ----
    pr_number = _extract_pr_number(gh_ref)
    if pr_number is None:
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        if event_path and Path(event_path).exists():
            try:
                event = json.loads(Path(event_path).read_text(encoding="utf-8"))
                pr_number = event.get("pull_request", {}).get("number")
            except Exception:
                pass

    # ---- Self-trigger guard ----
    # Exit 0 immediately if this commit was authored by coverage-agent itself,
    # or if the diff touches only the tests_dir (avoids infinite loops in commit mode).
    if _is_self_triggered(gh_base_ref):
        console.print("[dim]coverage-agent: self-trigger detected — skipping.[/dim]")
        raise typer.Exit(0)

    # ---- Load config ----
    cfg = load_config(config_path or None)
    if scope:
        cfg = cfg.model_copy(update={"scope": scope})
    if coverage_file:
        cfg = cfg.model_copy(update={"coverage_file": coverage_file})
    if model:
        cfg = cfg.model_copy(update={"model": model})
    if commit_mode:
        cfg = cfg.model_copy(update={"commit_mode": commit_mode})

    # ---- Job-python invariant ----
    # The Action installs coverage-agent into its own isolated uv env.
    # pytest/coverage subprocesses must run in the user's job env.
    # shutil.which("python") returns the job's python from PATH.
    job_python = shutil.which("python") or shutil.which("python3") or sys.executable
    if job_python != sys.executable:
        cfg = cfg.model_copy(update={"python_executable": job_python})

    # ---- Resolve credentials ----
    if cfg.model and cfg.model != DEFAULT_MODEL:
        os.environ.setdefault("COVERAGE_AGENT_MODEL", cfg.model)

    try:
        creds = Credentials.for_cli_env()
    except RuntimeError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    # ---- Compute diff base ----
    base_ref = ""
    if cfg.scope == "diff":
        if gh_base_ref:
            base_ref = f"origin/{gh_base_ref}"
        # Validate that the merge-base can be resolved.
        if base_ref:
            import subprocess as _sp
            check = _sp.run(
                ["git", "merge-base", base_ref, "HEAD"],
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                err_console.print(
                    f"[red]Cannot resolve merge-base for {base_ref}.[/red]\n"
                    "Add `fetch-depth: 0` to your checkout step:\n\n"
                    "    - uses: actions/checkout@v4\n"
                    "      with:\n"
                    "        fetch-depth: 0\n"
                )
                raise typer.Exit(1)

    # ---- Run pipeline ----
    from coverage_agent.gaps.coverage_data import load_coverage_file, parse_coverage
    from coverage_agent.gaps.select import select_gaps
    from coverage_agent.gaps.diff import compute_diff_gaps
    from coverage_agent.engine.graph import run_pipeline
    from coverage_agent.engine.regression import RegressionGuard
    from coverage_agent.report.run_report import serialize_run_report
    from coverage_agent.report.markdown import render_comment
    from coverage_agent.report.github import upsert_comment, push_commit, open_pr

    repo_root = str(Path.cwd())
    report_path = str(Path(repo_root) / "coverage-agent-report.json")

    # Reuse the shared pipeline helper.
    _run_pipeline(
        repo_root=repo_root,
        cfg=cfg,
        creds=creds,
        coverage_file=coverage_file,
        base_ref=base_ref,
        output=report_path,
        run_pipeline=run_pipeline,
        select_gaps=select_gaps,
        parse_coverage=parse_coverage,
        compute_diff_gaps=compute_diff_gaps,
        load_coverage_file=load_coverage_file,
        RegressionGuard=RegressionGuard,
    )

    # ---- Load the written report ----
    try:
        from coverage_agent.report.run_report import load_run_report
        report = load_run_report(report_path)
    except Exception as exc:
        err_console.print(f"[red]Failed to load pipeline report:[/red] {exc}")
        raise typer.Exit(1)

    # ---- Render comment ----
    comment_body = render_comment(report)

    # ---- Deliver ----
    delivery_mode = cfg.commit_mode

    if delivery_mode == "comment" or delivery_mode not in ("commit", "pr"):
        if not gh_repo or pr_number is None:
            if preview:
                console.print("[dim]preview: no GITHUB_REPOSITORY/PR number — comment delivery skipped[/dim]")
            else:
                err_console.print("[yellow]Warning:[/yellow] GITHUB_REPOSITORY or PR number not found — skipping comment.")
        else:
            upsert_comment(
                repo=gh_repo,
                pr_number=int(pr_number),
                body=comment_body,
                token=gh_token,
                preview=preview,
            )

    elif delivery_mode == "commit":
        push_commit(
            repo_root=repo_root,
            tests_dir=cfg.tests_dir,
            token=gh_token,
            preview=preview,
        )

    elif delivery_mode == "pr":
        head_branch = gh_ref.removeprefix("refs/heads/") if gh_ref.startswith("refs/heads/") else gh_base_ref
        open_pr(
            repo=gh_repo,
            pr_number=int(pr_number) if pr_number is not None else 0,
            head_branch=head_branch,
            tests_dir=cfg.tests_dir,
            repo_root=repo_root,
            token=gh_token,
            preview=preview,
        )

    # ---- Write GITHUB_OUTPUT ----
    _write_github_output(
        gh_output_file=gh_output_file,
        tests_added=report.tests_accepted,
        gaps_found=report.gaps_found,
        report_path=report_path,
    )

    console.print(
        f"\n[bold]ci-run done.[/bold] gaps_found={report.gaps_found} "
        f"tests_accepted={report.tests_accepted} report={report_path}"
    )


# ---------------------------------------------------------------------------
# ci-run helpers
# ---------------------------------------------------------------------------

def _extract_pr_number(gh_ref: str) -> int | None:
    """Extract the PR number from GITHUB_REF (refs/pull/<n>/merge)."""
    import re
    m = re.match(r"refs/pull/(\d+)/", gh_ref)
    if m:
        return int(m.group(1))
    return None


def _is_self_triggered(gh_base_ref: str) -> bool:
    """Return True if this run was triggered by a coverage-agent commit or tests-only diff."""
    import subprocess as _sp

    # Check if the HEAD commit message starts with coverage-agent:.
    result = _sp.run(
        ["git", "log", "-1", "--format=%s"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        subject = result.stdout.strip()
        if subject.startswith("coverage-agent:"):
            return True

    # If diffing against a base, check if the diff touches only tests/generated/.
    if gh_base_ref:
        diff_result = _sp.run(
            ["git", "diff", "--name-only", f"origin/{gh_base_ref}...HEAD"],
            capture_output=True,
            text=True,
        )
        if diff_result.returncode == 0:
            changed = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]
            if changed and all(f.startswith("tests/generated/") for f in changed):
                return True

    return False


def _write_github_output(
    gh_output_file: str,
    tests_added: int,
    gaps_found: int,
    report_path: str,
) -> None:
    """Write key=value pairs to $GITHUB_OUTPUT."""
    lines = [
        f"tests-added={tests_added}",
        f"gaps-found={gaps_found}",
        f"report-path={report_path}",
    ]
    if gh_output_file:
        try:
            with open(gh_output_file, "a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to write GITHUB_OUTPUT: %s", exc)
    else:
        for line in lines:
            logger.debug("GITHUB_OUTPUT: %s", line)
