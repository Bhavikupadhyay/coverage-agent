from typing import Literal, Optional
from pydantic import BaseModel, Field


class BranchGap(BaseModel):
    """Raw uncovered branch from coverage.py --branch output."""
    from_line: int = Field(..., description="Line where branch originates")
    to_line: int = Field(..., description="Line the branch jumps to when taken")


class CoverageGap(BaseModel):
    """A gap ready for the engine pipeline."""
    file_path: str = Field(..., description="Repo-relative path to the file")
    target_symbol: str = Field(..., description="Function or method name containing the gap")
    branch: BranchGap = Field(..., description="The specific uncovered branch arc")
    surrounding_lines: list[int] = Field(..., description="Line numbers of the enclosing function for context")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority (1.0 = highest)")
    gap_id: str = Field(..., description="Unique ID: {file_path}:{from_line}->{to_line}")


class ContextPayload(BaseModel):
    primary_code: str = Field(..., description="Full source of the target function/method")
    dependencies_code: dict[str, str] = Field(..., description="Map of dependency name to source or signature")
    graph_depth_used: int = Field(..., description="Jedi traversal depth used")
    tokens_used: int = Field(..., description="Total token count of this payload")
    fallback_used: bool = Field(default=False, description="True if Jedi failed and fell back to local scope")
    branch_condition_hint: Optional[str] = Field(
        default=None,
        description=(
            "Source text of the condition controlling the target branch. Extracted "
            "by AST walk over the file, keyed off the branch's from_line. Used by "
            "the writer to choose inputs that trigger the uncovered branch. "
            "None if extraction failed (pipeline still runs)."
        ),
    )


class DraftTest(BaseModel):
    test_code: str = Field(..., description="Complete executable pytest source code including imports")
    mocks_used: list[str] = Field(..., description="List of symbols mocked with unittest.mock.patch")
    target_branch: BranchGap = Field(..., description="The branch this test is designed to cover")


class EvalResult(BaseModel):
    syntax_valid: bool = Field(..., description="Passed ast.parse() check")
    unknown_imports: list[str] = Field(default_factory=list, description="Imports not found in context or stdlib")
    mock_complete: bool = Field(..., description="All external IO/network deps are mocked")
    assertion_score: int = Field(..., ge=1, le=5, description="Assertion quality placeholder (3 = neutral)")
    critique: str = Field(..., description="Actionable feedback for the writer")
    route: str = Field(..., pattern="^(EXECUTE|REWRITE|RECONTEXTUALIZE)$")


class ExecutionResult(BaseModel):
    execution_success: bool = Field(..., description="pytest exited 0")
    target_branch_hit: bool = Field(..., description="The specific target branch was newly covered")
    coverage_delta: float = Field(..., description="Percentage point increase in overall branch coverage")
    stderr_trace: str = Field(default="", description="Full stderr if execution failed")
    flaky: bool = Field(default=False, description="True if test passed sometimes and failed others")
    is_system_error: bool = Field(
        default=False,
        description="True when stderr indicates an environment-level failure that retry cannot fix",
    )


class GapResult(BaseModel):
    """Aggregated result for one gap."""
    gap: CoverageGap
    skipped: bool
    loops_taken: int
    phase1_scores: Optional[EvalResult]
    phase2_scores: Optional[ExecutionResult]
    final_test_committed: bool
    test_code: Optional[str] = None
    skip_reason: str = ""
    gap_difficulty: Literal["easy", "hard"] = Field(default="easy")


class RegressionResult(BaseModel):
    """Outcome of the RegressionGuard re-running the full suite with accepted tests in place."""
    baseline_passed: int = Field(..., description="Passing tests before any new tests were added")
    baseline_failed: int = Field(..., description="Failing tests before any new tests were added")
    post_passed: int = Field(..., description="Passing tests after all accepted tests were added")
    post_failed: int = Field(..., description="Failing tests after all accepted tests were added")
    new_failures: int = Field(..., description="Previously-passing tests that now fail")
    regression_detected: bool = Field(..., description="True if any previously-passing test now fails")
    summary: str = Field(default="", description="Human-readable one-liner for the report")
    skipped: bool = Field(default=False, description="True if no accepted tests existed")


class RunSummary(BaseModel):
    """Narrative for a completed run."""
    pr_description: str = Field(..., description="Short, markdown, ~120 words. Drop into a PR body.")
    full_summary: str = Field(..., description="Longer narrative covering what was accepted, skipped, and why.")
