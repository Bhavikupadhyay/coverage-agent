"""
CoverageAgent — FastAPI app: REST API + static UI.

Run: uvicorn app:app --reload
"""
import io
import logging
import os
import pathlib
import threading
import uuid
import zipfile

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from coverage_agent.config import DEFAULT_MODEL, is_offline_mode
from coverage_agent.credentials import Credentials
from coverage_agent.preflight import run_preflight
from coverage_agent.run_engine import (
    AgentEvent,
    RunRecord,
    _now,
    _rate_limiter,
    _runs,
    _to_result_dict,
    execute_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

# Models surfaced in the UI dropdown. BYOK users can supply any LiteLLM model string.
_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "openai/gpt-4o-mini",
]


# ---------------------------------------------------------------------------
# REST router
# ---------------------------------------------------------------------------

api_router = APIRouter()


@api_router.get("/health")
def health():
    demo_available = bool(os.environ.get("DEMO_GROQ_API_KEY") and os.environ.get("DEMO_E2B_API_KEY"))
    return {
        "status": "ok",
        "demo_available": demo_available,
        "default_model": DEFAULT_MODEL,
    }


@api_router.post("/api/preflight")
async def preflight(body: dict):
    """Cheap pre-run validation: GitHub reachability, LLM auth, E2B auth.

    Saves users from waiting through a 2-minute pipeline only to fail at the
    sandbox setup or first LLM call. The UI calls this when the user clicks
    "Generate tests" — if anything is red, the run never starts.
    """
    mode = (body.get("mode") or "demo").lower()
    repo_url = (body.get("repo_url") or "").strip()

    if is_offline_mode():
        creds = Credentials.for_offline()
    elif mode == "demo":
        try:
            creds = Credentials.for_demo()
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
    elif mode == "byok":
        try:
            creds = Credentials.for_byok(body)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    elif mode == "offline":
        creds = Credentials.for_offline()
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown mode '{mode}'. Use demo | byok | offline."},
        )

    report = run_preflight(repo_url=repo_url, mode=creds.mode, credentials=creds)
    return report.to_dict()


@api_router.post("/api/run")
async def start_run(body: dict):
    mode = body.get("mode", "demo").lower()
    repo_url = body.get("repo_url", "").strip()
    max_gaps = int(body.get("max_gaps", 2))

    if not repo_url:
        return JSONResponse(status_code=400, content={"error": "repo_url is required"})

    # OFFLINE_MODE=true in the server env forces offline regardless of UI mode,
    # so contributors can exercise the full UI without provisioning real keys.
    if is_offline_mode():
        credentials = Credentials.for_offline()
        max_gaps = min(max_gaps, 3)
    elif mode == "demo":
        if not repo_url.startswith(("https://github.com/", "http://github.com/")):
            return JSONResponse(
                status_code=400,
                content={"error": "Demo mode only supports public GitHub URLs."},
            )
        quota_error = _rate_limiter.check_demo_quota()
        if quota_error:
            return JSONResponse(status_code=429, content={"error": quota_error})
        max_gaps = min(max_gaps, 3)
        try:
            credentials = Credentials.for_demo(eval_strictness=body.get("eval_strictness", "balanced"))
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
    elif mode == "byok":
        try:
            credentials = Credentials.for_byok(body)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    elif mode == "offline":
        credentials = Credentials.for_offline()
        max_gaps = min(max_gaps, 3)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown mode '{mode}'. Use demo | byok | offline."},
        )

    run_id = str(uuid.uuid4())
    record = RunRecord(run_id=run_id)
    _runs[run_id] = record

    threading.Thread(
        target=execute_run,
        args=(record, credentials, repo_url, max_gaps),
        daemon=True,
    ).start()

    return {"run_id": run_id, "status": "queued", "mode": credentials.mode}


@api_router.get("/api/run/{run_id}")
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


@api_router.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    """Server-Sent Events stream for a run. Clients receive typed AgentEvent objects."""
    import asyncio

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


@api_router.get("/api/run/{run_id}/zip")
def download_zip(run_id: str):
    """Download all committed tests for a run as a ZIP archive."""
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in record.results:
            result = _to_result_dict(r, record)
            if result.get("test_committed") and result.get("test_code"):
                symbol = result.get("target_symbol", "unknown").replace(".", "_")
                zf.writestr(f"test_{symbol}.py", result["test_code"])
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="coverage_tests_{run_id[:8]}.zip"'},
    )


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

_PUBLIC = pathlib.Path(__file__).parent / "public"

fast_app = FastAPI(title="CoverageAgent API")
fast_app.include_router(api_router)


@fast_app.get("/")
async def index():
    return FileResponse(_PUBLIC / "index.html")


@fast_app.get("/public/{file_path:path}")
async def static(file_path: str):
    full = _PUBLIC / file_path
    if not full.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(full), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


app = fast_app
