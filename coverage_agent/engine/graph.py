import logging
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from coverage_agent.config import AgentConfig
from coverage_agent.engine.validator import EvalAgent
from coverage_agent.engine.writer import TestWriter
from coverage_agent.contracts import (
    ContextPayload,
    CoverageGap,
    DraftTest,
    ValidationResult,
    ExecutionResult,
    GapResult,
)
from coverage_agent.credentials import Credentials

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    # Inputs — set before each gap
    repo_path: str
    target_gap: CoverageGap
    # Cluster of sibling gaps sharing the same (file_path, target_symbol).
    # None or a single-element list → single-gap behaviour (unchanged).
    cluster: Optional[list]
    baseline_coverage: dict
    config: AgentConfig

    # Agent outputs — populated as pipeline runs
    context: Optional[ContextPayload]
    draft_test: Optional[DraftTest]
    eval_result: Optional[ValidationResult]
    exec_result: Optional[ExecutionResult]
    # Per-arc hit map from the verifying execution (multi-gap clusters only)
    cluster_arc_hits: dict

    # ReAct trace steps (list of dicts — serialized later into AgentTrace)
    agent_trace: list

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
        from coverage_agent.context.jedi_graph import build_context
        from coverage_agent.context.branch_conditions import extract_branch_condition_from_source
        from pathlib import Path

        depth = state["context_depth_requested"] if state["loop_count"] > 0 else 1
        gap = state["target_gap"]
        repo_root = state["repo_path"] or "."

        context = build_context(
            file_path=gap.file_path,
            target_symbol=gap.target_symbol,
            depth=depth,
            repo_root=repo_root,
            from_line=gap.branch.from_line,
        )

        if context.branch_condition_hint is None:
            try:
                src = (Path(repo_root) / gap.file_path).read_text(encoding="utf-8")
                hint = extract_branch_condition_from_source(src, gap.branch.from_line)
                context = context.model_copy(update={"branch_condition_hint": hint})
            except Exception:
                pass

        logger.info(
            "context_architect: gap=%s depth=%d tokens=%d",
            gap.gap_id, context.graph_depth_used, context.tokens_used,
        )
        return {"context": context}
    return _context_architect_node


def _make_gap_filter_node():
    def _gap_filter_node(state: PipelineState) -> dict:
        from coverage_agent.gaps.select import io_difficulty_flag
        difficulty = io_difficulty_flag(state["target_gap"], state.get("context"))
        logger.info("gap_filter: gap=%s difficulty=%s", state["target_gap"].gap_id, difficulty)
        return {"gap_difficulty": difficulty}
    return _gap_filter_node


def _route_after_gap_filter(state: PipelineState) -> str:
    # Hard gaps are not skipped at the graph level — select_gaps already capped the list.
    # This node only marks difficulty for the writer prompt; routing always continues.
    return "TEST_WRITER"


def _make_test_writer_node(credentials: Credentials):
    def _test_writer_node(state: PipelineState) -> dict:
        cfg = state.get("config") or AgentConfig()
        trace = state.get("agent_trace", [])
        draft = TestWriter(credentials).run(
            state["target_gap"],
            state["context"],
            critique=state["last_critique"],
            config=cfg,
            trace=trace,
            cluster=state.get("cluster"),
        )
        logger.info("test_writer: gap=%s mocks=%s", state["target_gap"].gap_id, draft.mocks_used)
        return {"draft_test": draft, "agent_trace": trace}
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
            state["target_gap"].gap_id, result.syntax_valid, result.route, state["loop_count"],
        )
        updates: dict = {"eval_result": result, "last_critique": result.critique}
        if result.route in ("REWRITE", "RECONTEXTUALIZE"):
            updates["loop_count"] = state["loop_count"] + 1
        if result.route == "RECONTEXTUALIZE":
            updates["context_depth_requested"] = state["context_depth_requested"] + 1
        return updates
    return _eval_agent_node


def _make_execution_runner_node(credentials: Credentials):
    def _execution_runner_node(state: PipelineState) -> dict:
        from coverage_agent.engine.executor import ExecutionRunner
        cfg = state.get("config") or AgentConfig()
        cluster: list | None = state.get("cluster")
        runner = ExecutionRunner(credentials)
        exec_result = runner.run(
            state["draft_test"],
            state["target_gap"],
            config=cfg,
            baseline_coverage=state.get("baseline_coverage", {}),
            cluster=cluster,
        )
        logger.info(
            "execution_runner: gap=%s success=%s branch_hit=%s targets=%d/%d",
            state["target_gap"].gap_id,
            exec_result.execution_success,
            exec_result.target_branch_hit,
            exec_result.targets_hit,
            exec_result.targets_total,
        )
        updates: dict = {"exec_result": exec_result, "cluster_arc_hits": runner.last_arc_hits}

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
                # Build critique naming missed arcs, including sibling gaps.
                missed_arcs = _missed_arcs_from_cluster(
                    cluster=cluster,
                    primary_gap=gap,
                    exec_result=exec_result,
                    arc_hits=runner.last_arc_hits,
                )
                hint = state["context"].branch_condition_hint if state.get("context") else None
                hint_line = (
                    f"\n\nThe condition controlling the primary branch is: `{hint}`\n"
                    "Write test inputs that force this condition to evaluate to the untaken path."
                    if hint else ""
                )
                if missed_arcs:
                    arc_list = "\n".join(
                        f"  - line {g.branch.from_line} -> line {g.branch.to_line}"
                        for g in missed_arcs
                    )
                    updates["last_critique"] = (
                        "The previous attempt RAN SUCCESSFULLY but did not exercise "
                        f"the following uncovered arcs of `{gap.target_symbol}`:\n"
                        f"{arc_list}"
                        f"{hint_line}"
                    )
                else:
                    updates["last_critique"] = (
                        "The previous attempt RAN SUCCESSFULLY but did not exercise "
                        f"the target branch (line {gap.branch.from_line} -> {gap.branch.to_line})."
                        f"{hint_line}"
                    )
            updates["loop_count"] = state["loop_count"] + 1

        return updates
    return _execution_runner_node


def _missed_arcs_from_cluster(
    cluster: list | None,
    primary_gap,
    exec_result,
    arc_hits: dict | None = None,
) -> list:
    """Returns list of gaps in cluster whose arc was not hit by exec_result."""
    from coverage_agent.engine.executor import _cluster_results_from_exec
    if not cluster or len(cluster) <= 1:
        return []
    per_gap = _cluster_results_from_exec(cluster, exec_result, arc_hits)
    return [g for g, r in zip(cluster, per_gap) if not r.target_branch_hit]


def _accept_node(state: PipelineState) -> dict:
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

def _traced(fn: Callable, name: str, cb) -> Callable:
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
    credentials: Credentials,
    event_callback=None,
):
    """Builds and compiles the LangGraph graph for a single gap."""
    graph = StateGraph(PipelineState)
    cb = event_callback

    graph.add_node("context_architect", _traced(_make_context_architect_node(credentials), "context_architect", cb))
    graph.add_node("gap_filter", _traced(_make_gap_filter_node(), "gap_filter", cb))
    graph.add_node("test_writer", _traced(_make_test_writer_node(credentials), "test_writer", cb))
    graph.add_node("eval_agent", _traced(_make_eval_agent_node(credentials), "eval_agent", cb))
    graph.add_node("execution_runner", _traced(_make_execution_runner_node(credentials), "execution_runner", cb))
    graph.add_node("accept", _traced(_accept_node, "accept", cb))
    graph.add_node("skip", _traced(_skip_node, "skip", cb))

    graph.set_entry_point("context_architect")
    graph.add_edge("context_architect", "gap_filter")
    graph.add_edge("gap_filter", "test_writer")
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
    credentials: Credentials,
    config: Optional[AgentConfig] = None,
    baseline_coverage: Optional[dict] = None,
    repo_path: str = "",
    event_callback=None,
    cluster: Optional[list] = None,
) -> tuple[GapResult, PipelineState]:
    """Runs one gap (or a cluster sharing file+symbol) through the full pipeline.

    When cluster is None or length 1, behaviour is identical to the pre-clustering
    code path.  When cluster has >1 gaps, the writer receives all sibling arcs and
    the executor verifies each gap independently; the primary gap (first in cluster)
    drives context building and determines the returned GapResult.

    Returns a GapResult for the primary gap and the final PipelineState.  Callers
    that need per-gap results for every sibling should use run_pipeline_cluster.
    """
    cfg = config or AgentConfig()
    compiled = build_pipeline(credentials, event_callback=event_callback)

    initial_state: PipelineState = {
        "repo_path": repo_path,
        "target_gap": gap,
        "cluster": cluster,
        "baseline_coverage": baseline_coverage or {},
        "config": cfg,
        "context": None,
        "draft_test": None,
        "eval_result": None,
        "exec_result": None,
        "agent_trace": [],
        "loop_count": 0,
        "context_depth_requested": 1,
        "last_critique": None,
        "skipped": False,
        "gap_difficulty": "easy",
    }

    final_state: PipelineState = compiled.invoke(initial_state)

    # The primary exec_result drives acceptance: >=1 arc in the cluster was hit.
    exec_result = final_state.get("exec_result")
    accepted = credentials.should_commit(exec_result)
    gap_result = GapResult(
        gap=gap,
        skipped=final_state["skipped"],
        loops_taken=final_state["loop_count"],
        validation=final_state.get("eval_result"),
        execution=exec_result,
        accepted=accepted,
        test_code=final_state["draft_test"].test_code if accepted and final_state.get("draft_test") else None,
        gap_difficulty=final_state.get("gap_difficulty", "easy"),
    )

    return gap_result, final_state


def run_pipeline_cluster(
    cluster: list[CoverageGap],
    credentials: Credentials,
    config: Optional[AgentConfig] = None,
    baseline_coverage: Optional[dict] = None,
    repo_path: str = "",
    event_callback=None,
) -> tuple[list[GapResult], PipelineState]:
    """Runs a cluster of sibling gaps through one pipeline invocation.

    Returns one GapResult per gap in cluster plus the final PipelineState.
    The primary gap is cluster[0]; it drives context building exactly as today.

    Acceptance rule: the test is kept if >=1 gap in the cluster was hit AND the
    flake gate passes.  Each gap gets its own GapResult with its individual hit
    status.  Gaps whose arc was not hit by the accepted test are recorded with
    accepted=False and skip_reason naming the miss.
    """
    from coverage_agent.engine.executor import _cluster_results_from_exec

    primary = cluster[0]
    primary_result, final_state = run_pipeline(
        gap=primary,
        credentials=credentials,
        config=config,
        baseline_coverage=baseline_coverage,
        repo_path=repo_path,
        event_callback=event_callback,
        cluster=cluster,
    )

    exec_result = final_state.get("exec_result")
    test_code = primary_result.test_code
    skipped = final_state["skipped"]

    if len(cluster) == 1:
        return [primary_result], final_state

    # Build per-gap results using coverage data from the single execution.
    per_exec = _cluster_results_from_exec(
        cluster, exec_result, final_state.get("cluster_arc_hits")
    )

    gap_results: list[GapResult] = []
    for gap, gexec in zip(cluster, per_exec):
        gap_accepted = (
            not skipped
            and gexec is not None
            and credentials.should_commit(gexec)
        )
        skip_reason = ""
        if not gap_accepted and not skipped and exec_result and exec_result.execution_success:
            skip_reason = (
                f"arc {gap.branch.from_line}->{gap.branch.to_line} not hit by accepted test"
            )
        elif skipped:
            skip_reason = primary_result.skip_reason or "cluster skipped"
        gap_results.append(GapResult(
            gap=gap,
            skipped=skipped or (not gap_accepted and skip_reason != ""),
            loops_taken=final_state["loop_count"],
            validation=final_state.get("eval_result"),
            execution=gexec,
            accepted=gap_accepted,
            test_code=test_code if gap_accepted else None,
            gap_difficulty=final_state.get("gap_difficulty", "easy"),
            skip_reason=skip_reason,
        ))

    return gap_results, final_state
