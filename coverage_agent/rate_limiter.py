import asyncio
import time


class RateLimiter:
    DEMO_DAILY_LIMIT = 10

    def __init__(self):
        self._demo_sem = asyncio.Semaphore(1)
        self._byok_sem = asyncio.Semaphore(1)
        self._demo_count = 0
        self._demo_reset_time = time.time() + 86400

    def _reset_if_needed(self):
        if time.time() > self._demo_reset_time:
            self._demo_count = 0
            self._demo_reset_time = time.time() + 86400

    def check_demo_quota(self) -> str | None:
        self._reset_if_needed()
        if self._demo_count >= self.DEMO_DAILY_LIMIT:
            return f"Demo quota reached ({self.DEMO_DAILY_LIMIT} runs/day). Use BYOK mode."
        return None

    def increment_demo(self):
        self._demo_count += 1

    @property
    def demo_sem(self):
        return self._demo_sem

    @property
    def byok_sem(self):
        return self._byok_sem
