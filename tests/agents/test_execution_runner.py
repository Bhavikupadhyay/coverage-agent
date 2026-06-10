"""ExecutionRunner: re-runs a successful test twice for flakiness detection."""
from coverage_agent.engine.executor import ExecutionRunner
from coverage_agent.contracts import ExecutionResult


class _FakeSandbox:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def run_test(self, test_code, **kwargs):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return ExecutionResult(
            execution_success=True,
            target_branch_hit=True,
            coverage_delta=0.0,
            stderr_trace="",
            flaky=False,
        )


def _make_result(success: bool, branch_hit: bool = True, delta: float = 0.4) -> ExecutionResult:
    return ExecutionResult(
        execution_success=success,
        target_branch_hit=branch_hit if success else False,
        coverage_delta=delta if success else 0.0,
        stderr_trace="" if success else "boom",
        flaky=False,
    )


def test_failure_returns_immediately(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(success=False)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox)
    assert sandbox.calls == 1
    assert result.execution_success is False


def test_three_passes_returns_non_flaky(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(True), _make_result(True), _make_result(True)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox)
    assert sandbox.calls == 3
    assert result.execution_success is True
    assert result.flaky is False
    assert result.target_branch_hit is True


def test_inconsistent_runs_marked_flaky(creds, sample_gap, sample_draft):
    sandbox = _FakeSandbox([_make_result(True), _make_result(True), _make_result(False)])
    result = ExecutionRunner(creds).run(sample_draft, sample_gap, sandbox)
    assert sandbox.calls == 3
    assert result.flaky is True
    assert result.execution_success is False
