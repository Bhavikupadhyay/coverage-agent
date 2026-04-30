import logging
import os
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from coverage_agent.agents.context_architect import ContextArchitect
from coverage_agent.agents.eval_agent import EvalAgent
from coverage_agent.agents.execution_runner import ExecutionRunner
from coverage_agent.agents.test_writer import TestWriter
from coverage_agent.contracts.schemas import (
    ContextPayload,
    CoverageGap,
    DraftTest,
    EvalResult,
    ExecutionResult,
    GapResult,
)
from coverage_agent.sandbox.e2b_runner import E2BSandbox

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    # Inputs — set by Orchestrator before each gap
    repo_path: str
    target_gap: CoverageGap
    baseline_coverage: dict           # coverage.json dict from run_coverage_baseline()

    # Agent outputs — populated as pipeline runs
    context: Optional[ContextPayload]
    draft_test: Optional[DraftTest]
    eval_result: Optional[EvalResult]
    exec_result: Optional[ExecutionResult]

    # Loop management
    loop_count: int                   # increments on every REWRITE or RECONTEXTUALIZE
    context_depth_requested: int      # starts at 1, incremented on RECONTEXTUALIZE
    last_critique: Optional[str]      # Eval Agent critique forwarded to Test Writer on retry
    skipped: bool                     # True if loop_count hit limit without EXECUTE


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_context_architect_node(sandbox: E2BSandbox):
    def _context_architect_node(state: PipelineState) -> dict:
        depth_override = state["context_depth_requested"] if state["loop_count"] > 0 else None
        context = ContextArchitect().run(
            state["target_gap"],
            depth_override=depth_override,
            repo_root=state["repo_path"],
            sandbox=sandbox,
        )
        logger.info(
            "context_architect: gap=%s depth=%d tokens=%d",
            state["target_gap"].gap_id,
            context.graph_depth_used,
            context.tokens_used,
        )
        return {"context": context}
    return _context_architect_node


def _test_writer_node(state: PipelineState) -> dict:
    draft = TestWriter().run(
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


def _eval_agent_node(state: PipelineState) -> dict:
    result = EvalAgent().run(
        state["draft_test"],
        state["context"],
        state["target_gap"],
    )
    logger.info(
        "eval_agent: gap=%s syntax=%s assert=%d route=%s loop=%d",
        state["target_gap"].gap_id,
        result.syntax_valid,
        result.assertion_score,
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


def _make_execution_runner_node(sandbox: E2BSandbox):
    def _execution_runner_node(state: PipelineState) -> dict:
        exec_result = ExecutionRunner().run(
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
        return {"exec_result": exec_result}

    return _execution_runner_node


def _skip_node(state: PipelineState) -> dict:
    logger.info(
        "skip: gap=%s reached loop limit (%d loops) — skipping",
        state["target_gap"].gap_id,
        state["loop_count"],
    )
    return {"skipped": True}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_eval(state: PipelineState) -> str:
    if state["loop_count"] >= 3:
        return "SKIP"

    route = state["eval_result"].route
    if route == "RECONTEXTUALIZE":
        return "CONTEXT_ARCHITECT"
    elif route == "REWRITE":
        return "TEST_WRITER"
    else:
        return "EXECUTION_RUNNER"


# ---------------------------------------------------------------------------
# Node tracing wrapper
# ---------------------------------------------------------------------------

def _traced(
    fn: Callable,
    name: str,
    cb: Optional[Callable],
) -> Callable:
    """Wraps a node function to emit agent_start / agent_end events via cb."""
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
    sandbox: E2BSandbox,
    event_callback: Optional[Callable] = None,
):
    """
    Builds and compiles the LangGraph pipeline for a single gap.

    The sandbox is captured via closure so it can be reused across all gaps
    in a benchmark run without being serialized into PipelineState.
    If event_callback is provided it is called as:
        cb(event_type, agent_name, loop_count, gap_id, data)
    """
    graph = StateGraph(PipelineState)
    cb = event_callback

    graph.add_node("context_architect", _traced(_make_context_architect_node(sandbox), "context_architect", cb))
    graph.add_node("test_writer", _traced(_test_writer_node, "test_writer", cb))
    graph.add_node("eval_agent", _traced(_eval_agent_node, "eval_agent", cb))
    graph.add_node("execution_runner", _traced(_make_execution_runner_node(sandbox), "execution_runner", cb))
    graph.add_node("skip", _traced(_skip_node, "skip", cb))

    graph.set_entry_point("context_architect")
    graph.add_edge("context_architect", "test_writer")
    graph.add_edge("test_writer", "eval_agent")

    graph.add_conditional_edges(
        "eval_agent",
        _route_after_eval,
        {
            "CONTEXT_ARCHITECT": "context_architect",
            "TEST_WRITER": "test_writer",
            "EXECUTION_RUNNER": "execution_runner",
            "SKIP": "skip",
        },
    )

    graph.add_edge("execution_runner", END)
    graph.add_edge("skip", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point used by Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    gap: CoverageGap,
    sandbox: E2BSandbox,
    baseline_coverage: dict,
    repo_path: str = "",
    braintrust_logger=None,
    event_callback: Optional[Callable] = None,
) -> tuple[GapResult, PipelineState]:
    """
    Runs one gap through the full LangGraph pipeline.

    Builds a fresh compiled graph per call (cheap — no I/O). The sandbox is
    reused across all gaps via closure. Returns a GapResult. If a
    BraintrustLogger is provided, the result is logged before returning.
    """
    compiled = build_pipeline(sandbox, event_callback=event_callback)

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
    }

    final_state: PipelineState = compiled.invoke(initial_state)

    gap_result = GapResult(
        gap=gap,
        skipped=final_state["skipped"],
        loops_taken=final_state["loop_count"],
        phase1_scores=final_state.get("eval_result"),
        phase2_scores=final_state.get("exec_result"),
        final_test_committed=False,  # Orchestrator sets this after writing the test file
    )

    if braintrust_logger is not None:
        braintrust_logger.log(gap_result, final_state)

    return gap_result, final_state
