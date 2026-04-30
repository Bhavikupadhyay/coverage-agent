from typing import Optional
from pydantic import BaseModel, Field


class BranchGap(BaseModel):
    """Raw uncovered branch from coverage.py --branch output."""
    from_line: int = Field(..., description="Line where branch originates")
    to_line: int = Field(..., description="Line the branch jumps to when taken")


class CoverageGap(BaseModel):
    """A prioritized, enriched gap ready for the pipeline."""
    file_path: str = Field(..., description="Repo-relative path to the file")
    target_symbol: str = Field(..., description="Function or method name containing the gap")
    branch: BranchGap = Field(..., description="The specific uncovered branch")
    surrounding_lines: list[int] = Field(..., description="Line numbers of the enclosing function for context")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="LLM-assigned priority (1.0 = highest)")
    gap_id: str = Field(..., description="Unique ID: {file_path}:{from_line}->{to_line}")


class ContextPayload(BaseModel):
    primary_code: str = Field(..., description="Full source of the target function/method")
    dependencies_code: dict[str, str] = Field(..., description="Map of dependency name to source or signature")
    graph_depth_used: int = Field(..., description="Jedi traversal depth used")
    tokens_used: int = Field(..., description="Total token count of this payload")
    fallback_used: bool = Field(default=False, description="True if Jedi failed and fell back to local scope")


class DraftTest(BaseModel):
    test_code: str = Field(..., description="Complete executable pytest source code including imports")
    mocks_used: list[str] = Field(..., description="List of symbols mocked with unittest.mock.patch")
    target_branch: BranchGap = Field(..., description="The branch this test is designed to cover")


class EvalResult(BaseModel):
    syntax_valid: bool = Field(..., description="Passed ast.parse() check")
    unknown_imports: list[str] = Field(default_factory=list, description="Imports not found in context or stdlib")
    mock_complete: bool = Field(..., description="All external IO/network deps are mocked")
    assertion_score: int = Field(..., ge=1, le=5, description="Assertion quality: 1=trivial, 5=rigorous")
    critique: str = Field(..., description="Actionable feedback for Test Writer or Context Architect")
    route: str = Field(..., pattern="^(EXECUTE|REWRITE|RECONTEXTUALIZE)$")


class ExecutionResult(BaseModel):
    execution_success: bool = Field(..., description="pytest exited 0")
    target_branch_hit: bool = Field(..., description="The specific target branch was newly covered")
    coverage_delta: float = Field(..., description="Percentage point increase in overall branch coverage")
    stderr_trace: str = Field(default="", description="Full stderr if execution failed")
    flaky: bool = Field(default=False, description="True if test passed sometimes and failed others")


class GapResult(BaseModel):
    """Aggregated result for one gap, logged to Braintrust and scorecard."""
    gap: CoverageGap
    skipped: bool
    loops_taken: int
    phase1_scores: Optional[EvalResult]
    phase2_scores: Optional[ExecutionResult]
    final_test_committed: bool  # True = test verified by E2B and available for download
    test_code: Optional[str] = None  # Passing test source; populated when final_test_committed=True
