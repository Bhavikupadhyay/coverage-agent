import logging
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from coverage_agent.context.jedi_graph import JediContextGraph
from coverage_agent.engine.validator import EvalAgent
from coverage_agent.engine.executor import ExecutionRunner
from coverage_agent.engine.writer import TestWriter
from coverage_agent.contracts import (
    ContextPayload,
    CoverageGap,
    DraftTest,
    EvalResult,
    ExecutionResult,
    GapResult,
)
from coverage_agent.credentials import Credentials

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    # Inputs — set before each gap
    repo_path: str
    target_gap: CoverageGap
    baseline_coverage: dict

    # Agent outputs — populated as pipeline runs
    context: Optional[ContextPayload]
    draft_test: Optional[DraftTest]
    eval_result: Optional[EvalResult]
    exec_result: Optional[ExecutionResult]

    # Loop management
    loop_count: int
    context_depth_requested: int
    last_critique: Optional[str]
    skipped: bool
    gap_difficulty: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_context_architect_node(credentials: Credentials):
    def _context_architect_node(state: PipelineState) -> dict:
        from coverage_agent.context.jedi_graph import JediContextGraph
        from coverage_agent.context.branch_conditions import extract_branch_condition_from_source
        depth = state["context_depth_requested"] if state["loop_count"] > 0 else 1
        gap = state["target_gap"]
        repo_root = state["repo_path"] or "."

        jedi = JediContextGraph(repo_root=repo_root)
        context = jedi.build_context(gap, depth_override=depth)

        # Attach branch condition hint if not already present
        if context.branch_condition_hint is None:
            try:
                from pathlib import Path
                src = (Path(repo_root) / gap.file_path).read_text(encoding="utf-8")
                hint = extract_branch_condition_from_source(src, gap.branch.from_line)
                context = context.model_copy(update={"branch_condition_hint": hint})
            except Exception:
                pass

        logger.info(
            "context_architect: gap=%s depth=%d tokens=%d",
            gap.gap_id,
            context.graph_depth_used,
            context.tokens_used,
        )
        return {"context": context}
    return _context_architect_node


def _make_gap_filter_node():
    def _gap_filter_node(state: PipelineState) -> dict:
        # IO-difficulty heuristic: demotes (marks hard) but never skips.
        # Phase 0 placeholder — full heuristic lands in gaps/select.py in Phase 1.
        context = state.get("context")
        difficulty = "easy"
        if context and context.tokens_used > 8000:
            difficulty = "hard"
        logger.info(
            "gap_filter: gap=%s difficulty=%s",
            state["target_gap"].gap_id,
            difficulty,
        )
        return {"gap_difficulty": difficulty}
    return _gap_filter_node


def _route_after_gap_filter(state: PipelineState) -> str:
    return "SKIP" if state.get("gap_difficulty") == "hard" else "TEST_WRITER"


def _make_test_writer_node(credentials: Credentials):
    def _test_writer_node(state: PipelineState) -> dict:
        draft = TestWriter(credentials).run(
            state["target_gap"],
            state["context"],
            critique=state["last_critique"],
        )
        logger.info(
            "test_writer: gap=%s mocks=%s",
            state["target_gap"].gap_id,
            draft.mocks_used,
        )
        return {"draft_test": draft}
    return _test_writer_node


def _make_eval_agent_node(credentials: Credentials):
    def _eval_agent_node(state: PipelineState) -> dict:
        result = EvalAgent(credentials).run(
            state["draft_test"],
            state["context"],
            state["target_gap"],
        )
        logger.info(
            "eval_agent: gap=%s syntax=%s route=%s loop=%d",
            state["target_gap"].gap_id,
            result.syntax_valid,
            result.route,
            state["loop_count"],
        )
        updates: dict = {
            "eval_result": result,
            "last_critique": result.critique,
        }
        if result.route in ("REWRITE", "RECONTEXTUALIZE"):
            updates["loop_count"] = state["loop_count"] + 1
        if result.route == "RECONTEXTUALIZE":
            updates["context_depth_requested"] = state["context_depth_requested"] + 1
        return updates
    return _eval_agent_node


def _make_execution_runner_node(sandbox: Any, credentials: Credentials):
    def _execution_runner_node(state: PipelineState) -> dict:
        exec_result = ExecutionRunner(credentials).run(
            state["draft_test"],
            state["target_gap"],
            sandbox,
            baseline_coverage=state.get("baseline_coverage", {}),
        )
        logger.info(
            "execution_runner: gap=%s success=%s branch_hit=%s delta=%.2f",
            state["target_gap"].gap_id,
            exec_result.execution_success,
            exec_result.target_branch_hit,
            exec_result.coverage_delta,
        )
        updates: dict = {"exec_result": exec_result}

        if exec_result.is_system_error:
            return updates

        if not credentials.should_commit(exec_result):
            if not exec_result.execution_success:
                stderr = (exec_result.stderr_trace or "").strip()
                updates["last_critique"] = (
                    "The previous attempt CRASHED at runtime. "
                    "Here is the captured stderr:\n"
                    f"```\n{stderr[:1800]}\n```\n"
                    "Inspect the trace and fix the failing assertion, mock setup, "
                    "import path, or test fixture. The test must run to completion "
                    "with `pytest` exit code 0."
                )
            else:
                gap = state["target_gap"]
                hint = state["context"].branch_condition_hint if state.get("context") else None
                hint_line = (
                    f"\n\nThe condition controlling this branch is: `{hint}`\n"
                    "Write test inputs that force this condition to evaluate to the untaken path."
                    if hint else ""
                )
                updates["last_critique"] = (
                    "The previous attempt RAN SUCCESSFULLY but did not exercise "
                    f"the target branch (line {gap.branch.from_line} -> {gap.branch.to_line})."
                    f"{hint_line}"
                )
            updates["loop_count"] = state["loop_count"] + 1

        return updates

    return _execution_runner_node


def _accept_node(state: PipelineState) -> dict:
    """Terminal node for tests that pass the acceptance gate."""
    exec_result = state["exec_result"]
    logger.info(
        "accept: gap=%s — exec_success=%s branch_hit=%s",
        state["target_gap"].gap_id,
        exec_result.execution_success if exec_result else None,
        exec_result.target_branch_hit if exec_result else None,
    )
    return {}


def _skip_node(state: PipelineState) -> dict:
    logger.info(
        "skip: gap=%s reached loop limit (%d loops)",
        state["target_gap"].gap_id,
        state["loop_count"],
    )
    return {"skipped": True}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _make_route_after_eval(credentials: Credentials):
    max_loops = credentials.max_retry_loops()

    def _route(state: PipelineState) -> str:
        if state["loop_count"] >= max_loops:
            return "SKIP"
        route = state["eval_result"].route
        if route == "RECONTEXTUALIZE":
            return "CONTEXT_ARCHITECT"
        elif route == "REWRITE":
            return "TEST_WRITER"
        return "EXECUTION_RUNNER"

    return _route


def _make_route_after_execution(credentials: Credentials):
    max_loops = credentials.max_retry_loops()

    def _route(state: PipelineState) -> str:
        exec_result = state.get("exec_result")
        if exec_result and exec_result.is_system_error:
            return "SKIP"
        if credentials.should_commit(exec_result):
            return "ACCEPT"
        if state["loop_count"] >= max_loops:
            return "SKIP"
        return "TEST_WRITER"

    return _route


# ---------------------------------------------------------------------------
# Node tracing wrapper
# ---------------------------------------------------------------------------

def _traced(fn: Callable, name: str, cb: Optional[Callable]) -> Callable:
    def _inner(state: PipelineState) -> dict:
        gap_id = state["target_gap"].gap_id if state.get("target_gap") else ""
        loop = state.get("loop_count", 0)
        if cb:
            cb("agent_start", name, loop, gap_id, {})
        result = fn(state)
        if cb:
            cb("agent_end", name, loop, gap_id, {})
        return result
    return _inner


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_pipeline(
    sandbox: Any,
    credentials: Credentials,
    event_callback: Optional[Callable] = None,
):
    """Builds and compiles the LangGraph graph for a single gap."""
    graph = StateGraph(PipelineState)
    cb = event_callback

    graph.add_node("context_architect", _traced(_make_context_architect_node(credentials), "context_architect", cb))
    graph.add_node("gap_filter", _traced(_make_gap_filter_node(), "gap_filter", cb))
    graph.add_node("test_writer", _traced(_make_test_writer_node(credentials), "test_writer", cb))
    graph.add_node("eval_agent", _traced(_make_eval_agent_node(credentials), "eval_agent", cb))
    graph.add_node("execution_runner", _traced(_make_execution_runner_node(sandbox, credentials), "execution_runner", cb))
    graph.add_node("accept", _traced(_accept_node, "accept", cb))
    graph.add_node("skip", _traced(_skip_node, "skip", cb))

    graph.set_entry_point("context_architect")
    graph.add_edge("context_architect", "gap_filter")
    graph.add_conditional_edges(
        "gap_filter",
        _route_after_gap_filter,
        {"TEST_WRITER": "test_writer", "SKIP": "skip"},
    )
    graph.add_edge("test_writer", "eval_agent")

    graph.add_conditional_edges(
        "eval_agent",
        _make_route_after_eval(credentials),
        {
            "CONTEXT_ARCHITECT": "context_architect",
            "TEST_WRITER": "test_writer",
            "EXECUTION_RUNNER": "execution_runner",
            "SKIP": "skip",
        },
    )

    graph.add_conditional_edges(
        "execution_runner",
        _make_route_after_execution(credentials),
        {
            "TEST_WRITER": "test_writer",
            "ACCEPT": "accept",
            "SKIP": "skip",
        },
    )

    graph.add_edge("accept", END)
    graph.add_edge("skip", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    gap: CoverageGap,
    sandbox: Any,
    credentials: Credentials,
    baseline_coverage: dict,
    repo_path: str = "",
    event_callback: Optional[Callable] = None,
) -> tuple[GapResult, PipelineState]:
    """Runs one gap through the full LangGraph pipeline. Returns a GapResult."""
    compiled = build_pipeline(sandbox, credentials, event_callback=event_callback)

    initial_state: PipelineState = {
        "repo_path": repo_path,
        "target_gap": gap,
        "baseline_coverage": baseline_coverage,
        "context": None,
        "draft_test": None,
        "eval_result": None,
        "exec_result": None,
        "loop_count": 0,
        "context_depth_requested": 1,
        "last_critique": None,
        "skipped": False,
        "gap_difficulty": "easy",
    }

    final_state: PipelineState = compiled.invoke(initial_state)

    gap_result = GapResult(
        gap=gap,
        skipped=final_state["skipped"],
        loops_taken=final_state["loop_count"],
        phase1_scores=final_state.get("eval_result"),
        phase2_scores=final_state.get("exec_result"),
        final_test_committed=False,
        gap_difficulty=final_state.get("gap_difficulty", "easy"),
    )

    return gap_result, final_state
