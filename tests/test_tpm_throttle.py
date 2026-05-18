"""TPM throttle keeps usage under the configured cap."""
import time

from coverage_agent.tpm_throttle import TPMThrottle, estimate_tokens


def test_acquire_is_instant_under_capacity():
    t = TPMThrottle(model="test/model", tpm_limit=10000)
    start = time.monotonic()
    for _ in range(5):
        t.acquire(estimated_tokens=500)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"acquire should be fast under capacity, took {elapsed:.2f}s"


def test_acquire_blocks_near_capacity(monkeypatch):
    """When acquiring would overflow, sleep is called then the call eventually succeeds.

    We advance the mocked monotonic clock past the 60s window inside the sleep
    stub so the throttle's internal book-keeping ages the recorded event out.
    """
    sleeps: list[float] = []
    fake_now = [0.0]

    def fake_monotonic():
        return fake_now[0]

    def fake_sleep(s):
        sleeps.append(s)
        fake_now[0] += s

    monkeypatch.setattr("coverage_agent.tpm_throttle.time.sleep", fake_sleep)
    monkeypatch.setattr("coverage_agent.tpm_throttle.time.monotonic", fake_monotonic)

    t = TPMThrottle(model="test/model", tpm_limit=1000)
    t.acquire(estimated_tokens=800)
    t.acquire(estimated_tokens=200)
    assert sleeps, "Expected throttle to sleep when over capacity"
    assert sleeps[0] >= 1.0


def test_estimate_tokens_handles_empty():
    assert estimate_tokens(None) >= 256
    assert estimate_tokens([]) >= 256


def test_estimate_tokens_grows_with_content():
    short = estimate_tokens([{"content": "hello"}])
    long_text = "x" * 8000
    big = estimate_tokens([{"content": long_text}])
    assert big > short
