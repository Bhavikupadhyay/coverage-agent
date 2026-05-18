"""
Shared run logic: event system, run state, execution thread.
Used by app.py (FastAPI) and run_benchmark.py (CLI).
"""
import asyncio
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from coverage_agent.credentials import Credentials
from coverage_agent.db import init_db, list_runs, upsert_run
from coverage_agent.error_mapper import friendly_error
from coverage_agent.rate_limiter import RateLimiter
from coverage_agent.recommendations import generate as generate_recommendations


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Typed event system
# ---------------------------------------------------------------------------

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
    """Thread-safe ordered list of AgentEvents. Consumed by the SSE endpoint."""

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
# Run state
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

_db_conn = init_db()
for _saved in list_runs(_db_conn):
    _r = RunRecord(
        run_id=_saved["run_id"],
        status=_saved["status"],
        logs=_saved["logs"],
        scorecard=_saved["scorecard"],
        recommendations=_saved["recommendations"],
        results=_saved["results"],
        error=_saved["error"],
    )
    _runs[_r.run_id] = _r


# ---------------------------------------------------------------------------
# Run execution (blocking — always called from a daemon thread)
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


def execute_run(
    record: RunRecord,
    credentials: Credentials,
    repo_url: str,
    max_gaps: int,
) -> None:
    """Blocking run executor. Always called from a daemon thread.

    Credentials are passed in by the caller (app.py for web, run_benchmark.py
    for CLI). This function never touches os.environ — agents read their
    credentials from the Credentials object exclusively.
    """
    sem = (
        _rate_limiter.demo_sem if credentials.mode == "demo"
        else _rate_limiter.byok_sem
    )
    handler = _EventBusLogHandler(record.event_bus)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    def _flush_logs():
        for evt in record.event_bus.events_since(0):
            if evt.type == "log":
                msg = evt.data.get("msg", "")
                if msg and msg not in record.logs:
                    record.logs.append(msg)

    async def _run():
        async with sem:
            record.status = "running"
            if credentials.mode == "demo":
                _rate_limiter.increment_demo()

            try:
                braintrust_logger = None
                if credentials.braintrust_api_key and not credentials.is_offline:
                    from coverage_agent.evals.braintrust_logger import BraintrustLogger
                    braintrust_logger = BraintrustLogger(
                        project_name="coverage-agent",
                        api_key=credentials.braintrust_api_key,
                        model=credentials.llm_model,
                    )

                from coverage_agent.orchestrator import Orchestrator
                scorecard, results = Orchestrator(credentials=credentials).run(
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
                record.error = friendly_error(exc)
                record.status = "failed"
                logging.getLogger(__name__).exception("Run %s failed", record.run_id)
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
                root_logger.removeHandler(handler)
                _flush_logs()
                record.event_bus.mark_done()

    asyncio.run(_run())


def _gap_status_label(r: Any) -> str:
    if r.final_test_committed:
        return "committed"
    if r.phase2_scores and r.phase2_scores.execution_success and not r.phase2_scores.target_branch_hit:
        return "executed_missed_branch"
    if r.phase2_scores and not r.phase2_scores.execution_success:
        return "sandbox_failed"
    if r.skipped:
        return "retry_budget_exhausted"
    return "unknown"


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
        "status": _gap_status_label(r),
        "skip_reason": r.skip_reason,
        "recommendation": r.recommendation or "",
        "assertion_score": r.phase1_scores.assertion_score if r.phase1_scores else None,
        "critique": r.phase1_scores.critique if r.phase1_scores else "",
        "stderr_trace": (r.phase2_scores.stderr_trace[:500] if r.phase2_scores else ""),
        "test_code": r.test_code or "",
        "original_code": record.gap_original_codes.get(r.gap.gap_id, ""),
    }
