"""
CoverageAgent web application.
FastAPI backend + Gradio frontend, deployable to HuggingFace Spaces.
"""
import io
import logging
import os
import queue
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from coverage_agent.rate_limiter import RateLimiter
from coverage_agent.recommendations import generate as generate_recommendations

# ---------------------------------------------------------------------------
# LangGraph diagram — generated once at startup with DRY_RUN=true
# so no real API calls are made. Cleared after generation.
# ---------------------------------------------------------------------------
os.environ.setdefault("DRY_RUN", "true")
try:
    from coverage_agent.pipeline import build_pipeline
    from coverage_agent.sandbox.e2b_runner import E2BSandbox as _E2BSandbox
    MERMAID_DIAGRAM = build_pipeline(_E2BSandbox(".")).get_graph().draw_mermaid()
except Exception:
    MERMAID_DIAGRAM = "graph TD\n  A[Error generating diagram]"
finally:
    # Only clear if we set it; respect an externally-set DRY_RUN value
    if os.environ.get("DRY_RUN") == "true" and not os.environ.get("_DRY_RUN_EXTERNAL"):
        os.environ.pop("DRY_RUN", None)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory run store
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    status: str = "queued"          # queued | running | completed | failed
    logs: list[str] = field(default_factory=list)
    scorecard: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    error: str = ""


_runs: dict[str, RunRecord] = {}
_rate_limiter = RateLimiter()

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
fast_app = FastAPI(title="CoverageAgent API")


@fast_app.get("/health")
def health():
    return {"status": "ok"}


@fast_app.get("/api/graph")
def get_graph():
    return {"mermaid": MERMAID_DIAGRAM}


@fast_app.post("/api/run")
async def start_run(body: dict):
    mode = body.get("mode", "demo").lower()
    repo_url = body.get("repo_url", "").strip()
    max_gaps = int(body.get("max_gaps", 2))
    gemini_api_key = body.get("gemini_api_key", "")
    e2b_api_key = body.get("e2b_api_key", "")
    braintrust_api_key = body.get("braintrust_api_key", "")
    model = body.get("model", "gemini/gemini-2.5-flash")

    if not repo_url:
        return JSONResponse(status_code=400, content={"error": "repo_url is required"})

    if mode == "demo":
        if not repo_url.startswith(("https://github.com/", "http://github.com/")):
            return JSONResponse(
                status_code=400,
                content={"error": "Demo mode only supports public GitHub URLs."},
            )
        quota_error = _rate_limiter.check_demo_quota()
        if quota_error:
            return JSONResponse(status_code=429, content={"error": quota_error})
        max_gaps = min(max_gaps, 3)

    run_id = str(uuid.uuid4())
    record = RunRecord(run_id=run_id)
    _runs[run_id] = record

    threading.Thread(
        target=_execute_run,
        args=(record, mode, repo_url, max_gaps, model, gemini_api_key, e2b_api_key, braintrust_api_key),
        daemon=True,
    ).start()

    return {"run_id": run_id, "status": "queued"}


@fast_app.get("/api/run/{run_id}")
def get_run(run_id: str):
    record = _runs.get(run_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "run not found"})
    return {
        "run_id": record.run_id,
        "status": record.status,
        "logs": record.logs,
        "scorecard": record.scorecard,
        "recommendations": record.recommendations,
        "results": [_gap_result_to_dict(r) for r in record.results],
        "error": record.error,
    }


# ---------------------------------------------------------------------------
# Run execution (blocking, runs in a thread)
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord):
        self._q.put(self.format(record))


def _execute_run(
    record: RunRecord,
    mode: str,
    repo_url: str,
    max_gaps: int,
    model: str,
    gemini_api_key: str,
    e2b_api_key: str,
    braintrust_api_key: str,
):
    sem = _rate_limiter.demo_sem if mode == "demo" else _rate_limiter.byok_sem
    log_queue: queue.Queue = queue.Queue()
    handler = _QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # Save and inject API keys
    _prev = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY", ""),
        "BRAINTRUST_API_KEY": os.environ.get("BRAINTRUST_API_KEY", ""),
        "COVERAGE_AGENT_MODEL": os.environ.get("COVERAGE_AGENT_MODEL", ""),
    }

    def _flush_logs():
        while not log_queue.empty():
            try:
                record.logs.append(log_queue.get_nowait())
            except queue.Empty:
                break

    import asyncio

    async def _run():
        async with sem:
            nonlocal record
            record.status = "running"
            if mode == "demo":
                _rate_limiter.increment_demo()

            try:
                if gemini_api_key:
                    os.environ["GEMINI_API_KEY"] = gemini_api_key
                if e2b_api_key:
                    os.environ["E2B_API_KEY"] = e2b_api_key
                if braintrust_api_key:
                    os.environ["BRAINTRUST_API_KEY"] = braintrust_api_key
                os.environ["COVERAGE_AGENT_MODEL"] = model

                braintrust_logger = None
                if os.environ.get("BRAINTRUST_API_KEY"):
                    from coverage_agent.evals.braintrust_logger import BraintrustLogger
                    braintrust_logger = BraintrustLogger(project_name="coverage-agent")

                from coverage_agent.orchestrator import Orchestrator
                scorecard, results = Orchestrator().run(
                    repo_url_or_path=repo_url,
                    max_gaps=max_gaps,
                    braintrust_logger=braintrust_logger,
                )
                _flush_logs()
                record.scorecard = scorecard
                record.results = results
                record.recommendations = generate_recommendations(scorecard, results)
                record.status = "completed"

            except Exception as exc:
                _flush_logs()
                record.error = str(exc)
                record.status = "failed"
                logger.exception("Run %s failed", record.run_id)

            finally:
                for k, v in _prev.items():
                    if v:
                        os.environ[k] = v
                    else:
                        os.environ.pop(k, None)
                root_logger.removeHandler(handler)
                _flush_logs()

    asyncio.run(_run())


def _gap_result_to_dict(r) -> dict:
    return {
        "gap_id": r.gap.gap_id,
        "file_path": r.gap.file_path,
        "target_symbol": r.gap.target_symbol,
        "branch": f"{r.gap.branch.from_line}->{r.gap.branch.to_line}",
        "skipped": r.skipped,
        "loops_taken": r.loops_taken,
        "branch_hit": r.phase2_scores.target_branch_hit if r.phase2_scores else False,
        "coverage_delta": r.phase2_scores.coverage_delta if r.phase2_scores else 0.0,
        "test_committed": r.final_test_committed,
        "test_code": r.test_code or "",
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
]


def _build_gradio():
    import time

    with gr.Blocks(title="CoverageAgent") as demo:
        gr.Markdown("# CoverageAgent\nAutonomously improve Python test coverage using AI.")

        with gr.Tab("Run"):
            with gr.Row():
                mode_radio = gr.Radio(
                    ["Demo", "BYOK"],
                    value="Demo",
                    label="API Key Mode",
                    info="Demo uses server keys (limited). BYOK uses your own keys.",
                )

            with gr.Group(visible=False) as byok_group:
                gemini_key_in = gr.Textbox(label="GEMINI_API_KEY", type="password")
                e2b_key_in = gr.Textbox(label="E2B_API_KEY", type="password")
                braintrust_key_in = gr.Textbox(
                    label="BRAINTRUST_API_KEY (optional)", type="password"
                )
                model_in = gr.Dropdown(_MODELS, value=_MODELS[0], label="Model")

            repo_url_in = gr.Textbox(
                label="GitHub Repo URL",
                placeholder="https://github.com/psf/requests",
            )
            max_gaps_in = gr.Slider(1, 3, value=2, step=1, label="Max Gaps")
            run_btn = gr.Button("Run", variant="primary")

            log_out = gr.Textbox(
                label="Live Logs", lines=15, max_lines=15, interactive=False
            )
            scorecard_out = gr.Dataframe(label="Scorecard", interactive=False)
            recs_out = gr.Markdown(label="Recommendations")

            with gr.Accordion("Generated Tests", open=False) as tests_accordion:
                tests_out = gr.Dataframe(
                    headers=["gap_id", "file_path", "branch_hit"],
                    label="Results",
                    interactive=False,
                )
                download_btn = gr.DownloadButton(label="Download all tests (.zip)", visible=False)

        with gr.Tab("Pipeline Architecture"):
            gr.HTML(f"""
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <div class="mermaid">{MERMAID_DIAGRAM}</div>
            <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
            """)

        # --- Callbacks ---

        def _toggle_byok(mode):
            is_byok = mode == "BYOK"
            return (
                gr.update(visible=is_byok),
                gr.update(maximum=15 if is_byok else 3),
            )

        mode_radio.change(
            _toggle_byok,
            inputs=[mode_radio],
            outputs=[byok_group, max_gaps_in],
        )

        def _run(mode, gemini_key, e2b_key, braintrust_key, model, repo_url, max_gaps):
            import requests as _req

            payload = {
                "repo_url": repo_url,
                "max_gaps": int(max_gaps),
                "mode": mode.lower(),
                "gemini_api_key": gemini_key,
                "e2b_api_key": e2b_key,
                "braintrust_api_key": braintrust_key,
                "model": model if mode == "BYOK" else _MODELS[0],
            }
            resp = _req.post("http://localhost:7860/api/run", json=payload, timeout=10)
            if resp.status_code != 200:
                error = resp.json().get("error", resp.text)
                yield error, None, "", None, gr.update(visible=False), gr.update(visible=False)
                return

            run_id = resp.json()["run_id"]
            logs_seen = 0

            while True:
                time.sleep(2)
                status_resp = _req.get(f"http://localhost:7860/api/run/{run_id}", timeout=10)
                data = status_resp.json()
                new_logs = data.get("logs", [])[logs_seen:]
                logs_seen += len(new_logs)
                log_text = "\n".join(data["logs"])

                if data["status"] in ("completed", "failed"):
                    if data["status"] == "failed":
                        yield log_text + f"\n\nERROR: {data['error']}", None, "", None, gr.update(visible=False), gr.update(visible=False)
                        return

                    sc = data.get("scorecard", {})
                    scorecard_rows = [[k, v] for k, v in sc.items()]
                    recs_text = "\n\n".join(f"- {r}" for r in data.get("recommendations", []))
                    results = data.get("results", [])
                    table_rows = [
                        [r["gap_id"], r["file_path"], r["branch_hit"]]
                        for r in results
                    ]

                    # Build zip in memory
                    zip_buf = io.BytesIO()
                    has_tests = False
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for r in results:
                            if r.get("test_code"):
                                safe = r["gap_id"].replace("/", "_").replace(":", "_").replace("->", "_").replace(".", "_")
                                zf.writestr(f"tests/test_auto_{safe}.py", r["test_code"])
                                has_tests = True
                    zip_buf.seek(0)

                    yield (
                        log_text,
                        scorecard_rows,
                        recs_text,
                        table_rows,
                        gr.update(visible=has_tests, value=zip_buf.getvalue() if has_tests else None),
                        gr.update(open=has_tests),
                    )
                    return
                else:
                    yield log_text, None, "", None, gr.update(visible=False), gr.update(visible=False)

        run_btn.click(
            _run,
            inputs=[mode_radio, gemini_key_in, e2b_key_in, braintrust_key_in, model_in, repo_url_in, max_gaps_in],
            outputs=[log_out, scorecard_out, recs_out, tests_out, download_btn, tests_accordion],
        )

    return demo


gradio_app = _build_gradio()
app = gr.mount_gradio_app(fast_app, gradio_app, path="/")
