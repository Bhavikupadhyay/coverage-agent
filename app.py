"""
CoverageAgent web application.
FastAPI backend + Gradio frontend, deployable to HuggingFace Spaces.
"""
import asyncio
import io
import json
import logging
import os
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from coverage_agent.db import init_db, list_runs, upsert_run
from coverage_agent.rate_limiter import RateLimiter
from coverage_agent.recommendations import generate as generate_recommendations

# ---------------------------------------------------------------------------
# LangGraph diagram — generated once at startup with DRY_RUN=true
# ---------------------------------------------------------------------------
os.environ.setdefault("DRY_RUN", "true")
try:
    from coverage_agent.pipeline import build_pipeline
    from coverage_agent.sandbox.e2b_runner import E2BSandbox as _E2BSandbox
    MERMAID_DIAGRAM = build_pipeline(_E2BSandbox(".")).get_graph().draw_mermaid()
except Exception:
    MERMAID_DIAGRAM = "graph TD\n  A[Error generating diagram]"
finally:
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
# Typed event system
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class AgentEvent:
    type: str           # log | agent_start | agent_end | gap_start | gap_end | done | error
    ts: str
    agent: str = ""
    gap_id: str = ""
    loop: int = 0
    data: dict = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps(asdict(self))}\n\n"


class EventBus:
    """Thread-safe ordered list of AgentEvents. Consumed by SSE endpoint."""

    def __init__(self):
        self._events: list[AgentEvent] = []
        self._lock = threading.Lock()
        self._done = False

    def emit(self, evt: AgentEvent) -> None:
        with self._lock:
            self._events.append(evt)

    def mark_done(self) -> None:
        self._done = True

    def events_since(self, idx: int) -> list[AgentEvent]:
        with self._lock:
            return list(self._events[idx:])

    def is_done(self) -> bool:
        return self._done


# ---------------------------------------------------------------------------
# Run store
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
    # Live tracking — ephemeral, not persisted
    event_bus: EventBus = field(default_factory=EventBus)
    current_agent: str = ""
    current_gap_id: str = ""
    current_gap_idx: int = 0
    current_gap_total: int = 0
    current_loop: int = 0
    completed_agents: list[str] = field(default_factory=list)
    gap_original_codes: dict[str, str] = field(default_factory=dict)


_runs: dict[str, RunRecord] = {}
_rate_limiter = RateLimiter()

# SQLite persistence
_db_conn = init_db()
for _saved in list_runs(_db_conn):
    _r = RunRecord(
        run_id=_saved["run_id"],
        status=_saved["status"],
        logs=_saved["logs"],
        scorecard=_saved["scorecard"],
        recommendations=_saved["recommendations"],
        results=_saved["results"],   # list[dict] from DB — handled in _to_result_dict
        error=_saved["error"],
    )
    _runs[_r.run_id] = _r

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
        "results": [_to_result_dict(r, record) for r in record.results],
        "error": record.error,
        "current_agent": record.current_agent,
        "current_gap_id": record.current_gap_id,
        "current_gap_idx": record.current_gap_idx,
        "current_gap_total": record.current_gap_total,
        "current_loop": record.current_loop,
        "completed_agents": list(record.completed_agents),
    }


@fast_app.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    """Server-Sent Events stream for a run. Clients receive typed AgentEvent objects."""
    record = _runs.get(run_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    async def _generator():
        idx = 0
        while True:
            events = record.event_bus.events_since(idx)
            for evt in events:
                yield evt.to_sse()
                idx += 1
            if record.event_bus.is_done():
                yield AgentEvent(type="done", ts=_now()).to_sse()
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Run execution (blocking, runs in a daemon thread)
# ---------------------------------------------------------------------------

class _EventBusLogHandler(logging.Handler):
    """Intercepts log records and emits them as AgentEvent(type='log') to an EventBus."""

    def __init__(self, bus: EventBus):
        super().__init__()
        self._bus = bus

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self._bus.emit(AgentEvent(type="log", ts=_now(), data={"msg": msg}))


def _make_event_callback(record: RunRecord) -> Callable:
    """Returns a callback that updates RunRecord tracking state and emits to EventBus."""

    def _cb(event_type: str, agent: str, loop: int, gap_id: str, data: dict):
        evt = AgentEvent(type=event_type, ts=_now(), agent=agent, loop=loop, gap_id=gap_id, data=data)
        record.event_bus.emit(evt)

        if event_type == "agent_start":
            record.current_agent = agent
        elif event_type == "agent_end":
            record.current_agent = ""
            if agent not in record.completed_agents:
                record.completed_agents.append(agent)
        elif event_type == "gap_start":
            record.current_gap_id = gap_id
            record.current_gap_idx = data.get("gap_idx", 0)
            record.current_gap_total = data.get("total_gaps", 0)
            record.current_loop = loop
            record.completed_agents = []
        elif event_type == "gap_end":
            record.current_agent = ""
            original_code = data.get("original_code", "")
            if original_code:
                record.gap_original_codes[gap_id] = original_code

    return _cb


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
    handler = _EventBusLogHandler(record.event_bus)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    _prev = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY", ""),
        "BRAINTRUST_API_KEY": os.environ.get("BRAINTRUST_API_KEY", ""),
        "COVERAGE_AGENT_MODEL": os.environ.get("COVERAGE_AGENT_MODEL", ""),
    }

    def _flush_logs():
        for evt in record.event_bus.events_since(0):
            if evt.type == "log":
                msg = evt.data.get("msg", "")
                if msg and msg not in record.logs:
                    record.logs.append(msg)

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
                    event_callback=_make_event_callback(record),
                )
                _flush_logs()
                record.scorecard = scorecard
                record.results = results
                record.recommendations = generate_recommendations(scorecard, results)
                record.status = "completed"
                upsert_run(
                    _db_conn,
                    record.run_id,
                    record.status,
                    record.logs,
                    record.scorecard,
                    record.recommendations,
                    [_to_result_dict(r, record) for r in record.results],
                    record.error,
                )

            except Exception as exc:
                _flush_logs()
                record.error = str(exc)
                record.status = "failed"
                logger.exception("Run %s failed", record.run_id)
                upsert_run(
                    _db_conn,
                    record.run_id,
                    record.status,
                    record.logs,
                    record.scorecard,
                    record.recommendations,
                    [],
                    record.error,
                )

            finally:
                for k, v in _prev.items():
                    if v:
                        os.environ[k] = v
                    else:
                        os.environ.pop(k, None)
                root_logger.removeHandler(handler)
                _flush_logs()
                record.event_bus.mark_done()

    asyncio.run(_run())


def _to_result_dict(r: Any, record: RunRecord) -> dict:
    """Serializes a GapResult (Pydantic) or already-serialized dict (from DB) to a response dict."""
    if isinstance(r, dict):
        return r
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
        "original_code": record.gap_original_codes.get(r.gap.gap_id, ""),
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_PIPELINE_NODES = [
    ("context_architect", "Context Architect"),
    ("test_writer", "Test Writer"),
    ("eval_agent", "Eval Agent"),
    ("execution_runner", "Execution Runner"),
]

_TRACE_CSS = """
<style>
  .ca-trace { font-family: sans-serif; padding: 12px 0; }
  .ca-nodes { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; }
  .ca-node {
    display: inline-block; padding: 7px 13px; border-radius: 6px;
    border: 2px solid #ddd; font-size: 13px; transition: all 0.3s;
  }
  .ca-node.pending  { background:#f5f5f5; color:#aaa; border-color:#e0e0e0; }
  .ca-node.active   { background:#fff3cd; color:#856404; border-color:#ffc107;
                      font-weight:bold; box-shadow:0 0 0 3px rgba(255,193,7,.3); }
  .ca-node.done     { background:#d4edda; color:#155724; border-color:#28a745; }
  .ca-node.skipped  { background:#e9ecef; color:#6c757d; border-color:#adb5bd; text-decoration:line-through; }
  .ca-arrow { color:#bbb; font-size:16px; }
  .ca-gap-info { margin-top:8px; font-size:12px; color:#666; font-family:monospace; }
  .ca-gap-label { font-weight:bold; color:#444; }
</style>
"""


def _build_trace_html(data: dict) -> str:
    current = data.get("current_agent", "")
    completed = set(data.get("completed_agents", []))
    gap_id = data.get("current_gap_id", "")
    gap_idx = data.get("current_gap_idx", 0)
    gap_total = data.get("current_gap_total", 0)
    loop = data.get("current_loop", 0)
    status = data.get("status", "queued")

    parts = []
    for i, (node_id, label) in enumerate(_PIPELINE_NODES):
        if node_id == "execution_runner" and "skip" in completed:
            cls = "skipped"
        elif node_id == current:
            cls = "active"
        elif node_id in completed:
            cls = "done"
        else:
            cls = "pending"
        parts.append(f'<span class="ca-node {cls}">{label}</span>')
        if i < len(_PIPELINE_NODES) - 1:
            parts.append('<span class="ca-arrow">→</span>')

    gap_html = ""
    if gap_id and status == "running":
        gap_html = (
            f'<div class="ca-gap-info">'
            f'<span class="ca-gap-label">Gap {gap_idx}/{gap_total}</span>'
            f' — <code>{gap_id}</code>'
            f'{"  (loop " + str(loop) + ")" if loop > 0 else ""}'
            f'</div>'
        )
    elif status == "completed":
        gap_html = '<div class="ca-gap-info" style="color:#155724;">✓ Run complete</div>'
    elif status == "failed":
        gap_html = '<div class="ca-gap-info" style="color:#721c24;">✗ Run failed</div>'

    nodes_html = "\n".join(parts)
    return f'{_TRACE_CSS}<div class="ca-trace"><div class="ca-nodes">{nodes_html}</div>{gap_html}</div>'


def _build_monaco_html(results: list[dict]) -> str:
    """Renders test results with Monaco editors. Lazy-initialised when accordion opens."""
    if not results:
        return "<p style='color:#888; font-family:sans-serif; padding:12px'>No tests generated.</p>"

    items_html = []
    # Map gap index → JS snippet that initialises that gap's editor(s)
    per_gap_inits: list[str] = []

    for i, r in enumerate(results):
        gap_id = r.get("gap_id", f"gap_{i}")
        branch_hit_icon = "✅" if r.get("branch_hit") else "❌"
        test_code = r.get("test_code") or ""
        original_code = r.get("original_code") or ""
        loops = r.get("loops_taken", 0)
        delta = r.get("coverage_delta", 0.0)
        skipped = r.get("skipped", False)
        target_sym = r.get("target_symbol", "")

        escaped_test = json.dumps(test_code)
        escaped_orig = json.dumps(original_code)
        has_orig = bool(original_code.strip())

        summary_text = (
            f"{branch_hit_icon} &nbsp;"
            f"<code style='font-size:12px'>{gap_id}</code>"
            f"&nbsp; <span style='color:#888;font-size:11px'>"
            f"{'SKIPPED' if skipped else f'loops={loops} | &Delta;cov={delta:.2f}%'}"
            f"{'  |  ' + target_sym if target_sym else ''}"
            f"</span>"
        )

        orig_div = (
            f'<div style="flex:1;min-height:280px;" id="orig-editor-{i}"></div>'
            if has_orig else ""
        )
        test_flex = "flex:1.5;" if has_orig else "flex:1;"
        items_html.append(f"""
<details id="gap-details-{i}" style="margin-bottom:8px;border:1px solid #ddd;border-radius:6px;overflow:hidden;">
  <summary style="padding:10px 14px;cursor:pointer;background:#f8f9fa;font-family:monospace;
                  list-style:none;display:flex;align-items:center;gap:6px;user-select:none;">
    {summary_text}
  </summary>
  <div style="display:flex;gap:6px;padding:10px;background:#fff;">
    {orig_div}
    <div style="{test_flex}min-height:280px;" id="test-editor-{i}"></div>
  </div>
</details>""")

        # Build per-gap init snippet
        orig_init = (
            f"initEditor('orig-editor-{i}', {escaped_orig});"
            if has_orig else ""
        )
        per_gap_inits.append(f"'{i}': function(){{ {orig_init} initEditor('test-editor-{i}', {escaped_test}); }}")

    inits_map = ",\n    ".join(per_gap_inits)

    return f"""
<div id="ca-results">
{"".join(items_html)}
</div>
<script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs/loader.js"></script>
<script>
(function() {{
  var done = {{}};
  var inits = {{
    {inits_map}
  }};

  function initEditor(id, code) {{
    if (done[id]) return;
    done[id] = true;
    var el = document.getElementById(id);
    if (!el) return;
    require.config({{ paths: {{ vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs' }} }});
    require(['vs/editor/editor.main'], function() {{
      monaco.editor.create(el, {{
        value: code,
        language: 'python',
        readOnly: true,
        minimap: {{ enabled: false }},
        scrollBeyondLastLine: false,
        fontSize: 12,
        lineNumbers: 'on',
        automaticLayout: true,
      }});
    }});
  }}

  document.querySelectorAll('[id^="gap-details-"]').forEach(function(details) {{
    details.addEventListener('toggle', function() {{
      if (!details.open) return;
      var idx = details.id.replace('gap-details-', '');
      if (inits[idx]) inits[idx]();
    }});
  }});
}})();
</script>
"""


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

            # Live pipeline trace panel
            trace_html = gr.HTML(
                value=_build_trace_html({"status": "queued"}),
                label="Pipeline",
            )

            log_out = gr.Textbox(
                label="Live Logs", lines=12, max_lines=12, interactive=False
            )
            scorecard_out = gr.Dataframe(label="Scorecard", interactive=False)
            recs_out = gr.Markdown(label="Recommendations")

            with gr.Accordion("Generated Tests", open=False) as tests_accordion:
                test_viewer = gr.HTML(
                    value="<p style='color:#888;padding:12px;font-family:sans-serif'>Run the pipeline to see generated tests here.</p>",
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
                yield (
                    _build_trace_html({"status": "failed"}),
                    error, None, "", None,
                    gr.update(visible=False), gr.update(visible=False),
                )
                return

            run_id = resp.json()["run_id"]
            logs_seen = 0

            while True:
                time.sleep(2)
                status_resp = _req.get(f"http://localhost:7860/api/run/{run_id}", timeout=10)
                data = status_resp.json()

                log_text = "\n".join(data.get("logs", []))
                new_log_count = len(data.get("logs", []))
                logs_seen = new_log_count

                trace_update = _build_trace_html(data)

                if data["status"] in ("completed", "failed"):
                    if data["status"] == "failed":
                        yield (
                            trace_update,
                            log_text + f"\n\nERROR: {data['error']}",
                            None, "", None,
                            gr.update(visible=False), gr.update(visible=False),
                        )
                        return

                    sc = data.get("scorecard", {})
                    scorecard_rows = [[k, v] for k, v in sc.items()]
                    recs_text = "\n\n".join(f"- {r}" for r in data.get("recommendations", []))
                    results = data.get("results", [])

                    monaco_html = _build_monaco_html(results)

                    # Build zip in memory
                    zip_buf = io.BytesIO()
                    has_tests = False
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for r in results:
                            if r.get("test_code"):
                                safe = (
                                    r["gap_id"]
                                    .replace("/", "_").replace(":", "_")
                                    .replace("->", "_").replace(".", "_")
                                )
                                zf.writestr(f"tests/test_auto_{safe}.py", r["test_code"])
                                has_tests = True
                    zip_buf.seek(0)

                    yield (
                        trace_update,
                        log_text,
                        scorecard_rows,
                        recs_text,
                        monaco_html,
                        gr.update(visible=has_tests, value=zip_buf.getvalue() if has_tests else None),
                        gr.update(open=has_tests),
                    )
                    return
                else:
                    yield (
                        trace_update,
                        log_text, None, "", None,
                        gr.update(visible=False), gr.update(visible=False),
                    )

        run_btn.click(
            _run,
            inputs=[mode_radio, gemini_key_in, e2b_key_in, braintrust_key_in, model_in, repo_url_in, max_gaps_in],
            outputs=[trace_html, log_out, scorecard_out, recs_out, test_viewer, download_btn, tests_accordion],
        )

    return demo


gradio_app = _build_gradio()
app = gr.mount_gradio_app(fast_app, gradio_app, path="/")
