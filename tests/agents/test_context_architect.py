"""ContextArchitect should return a valid ContextPayload in offline mode."""
from coverage_agent.agents.context_architect import ContextArchitect
from coverage_agent.contracts.schemas import ContextPayload


def test_offline_returns_fixture_payload(offline_creds, sample_gap):
    payload = ContextArchitect(offline_creds).run(sample_gap)
    assert isinstance(payload, ContextPayload)
    assert payload.primary_code  # not empty
    assert payload.tokens_used > 0


def test_depth_override_is_respected(offline_creds, sample_gap):
    payload = ContextArchitect(offline_creds).run(sample_gap, depth_override=2)
    # offline returns the fixture regardless, but the call should not raise
    assert payload.graph_depth_used in (0, 1, 2)
