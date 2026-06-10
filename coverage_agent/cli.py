"""
CLI entrypoint — coverage-agent run / models / report.

All LLM calls and subprocess execution happen inside the engine; the CLI
is responsible only for wiring config, credentials, gap loading, and output.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

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
    max_gaps: int = typer.Option(0, help="Max gaps to target (0 = use config)"),
    coverage_file: str = typer.Option("", help="Path to .coverage, coverage.json, or coverage.xml"),
    model: str = typer.Option("", help="LiteLLM model ID (overrides config/env)"),
    config_path: str = typer.Option("", "--config", help="Path to .coverage-agent.yml"),
    output: str = typer.Option("", help="Write RunReport JSON to this path"),
    repo: str = typer.Option("", help="GitHub URL or local path to target repo (clones if URL)"),
    verbose: bool = typer.Option(False, "-v", help="Verbose logging"),
) -> None:
    """Run the coverage agent on the current repo."""
    import os
    import shutil
    import tempfile

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

    # ---- Resolve repo_root ----
    _tmp_clone: Optional[str] = None
    if repo:
        if repo.startswith("http://") or repo.startswith("https://") or repo.startswith("git@"):
            _tmp_clone = tempfile.mkdtemp(prefix="coverage-agent-")
            console.print(f"Cloning {repo} …")
            import subprocess as _sp
            result = _sp.run(
                ["git", "clone", "--depth=1", repo, _tmp_clone],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                err_console.print(f"[red]git clone failed:[/red]\n{result.stderr.strip()}")
                shutil.rmtree(_tmp_clone, ignore_errors=True)
                raise typer.Exit(1)
            repo_root = _tmp_clone
        else:
            p = Path(repo).resolve()
            if not p.exists():
                err_console.print(f"[red]Path not found:[/red] {repo}")
                raise typer.Exit(1)
            if not (p / ".git").exists():
                err_console.print(f"[red]Not a git repo:[/red] {p}")
                raise typer.Exit(1)
            repo_root = str(p)
    else:
        repo_root = str(Path.cwd())

    try:
        _run_pipeline(
            repo_root=repo_root,
            cfg=cfg,
            creds=creds,
            coverage_file=coverage_file,
            output=output,
            run_pipeline=run_pipeline,
            select_gaps=select_gaps,
            parse_coverage=parse_coverage,
            compute_diff_gaps=compute_diff_gaps,
            load_coverage_file=load_coverage_file,
            RegressionGuard=RegressionGuard,
        )
    finally:
        if _tmp_clone:
            shutil.rmtree(_tmp_clone, ignore_errors=True)


def _run_pipeline(
    repo_root: str,
    cfg,
    creds,
    coverage_file: str,
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
        all_gaps = compute_diff_gaps(coverage_data, repo_root=repo_root, exclude=cfg.exclude)
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

    from coverage_agent.evals.braintrust_logger import log_gap_result
    import datetime
    run_id = datetime.datetime.utcnow().strftime("run-%Y%m%dT%H%M%S")

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
        log_gap_result(gap_result, final_state.get("context"), run_id=run_id)

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
