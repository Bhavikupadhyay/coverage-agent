"""
Per-run LLM cost tracking.

LiteLLM exposes a `completion_cost(...)` helper for any model in its catalog.
The global `litellm.success_callback` fires after every successful completion,
so we install a per-instance callback that accumulates into a thread-safe
counter. One CostTracker per run keeps web runs and CLI runs isolated.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class CostTracker:
    """Accumulates LLM cost across all completions during a single run.

    Usage:
        tracker = CostTracker()
        tracker.install()
        try:
            # ... run pipeline ...
        finally:
            tracker.uninstall()
        print(tracker.total_usd)
    """

    _global_lock = threading.Lock()
    _active: list["CostTracker"] = []

    def __init__(self) -> None:
        self._total_usd: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()
        self._installed = False

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._count

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self._total_usd += cost_usd
            self._count += 1

    def install(self) -> None:
        """Register this tracker as the active callback target.

        Multiple trackers can be active concurrently (e.g. parallel runs);
        each litellm completion fans out to all of them.
        """
        if self._installed:
            return
        with CostTracker._global_lock:
            CostTracker._active.append(self)
            _ensure_litellm_callback()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        with CostTracker._global_lock:
            try:
                CostTracker._active.remove(self)
            except ValueError:
                pass
        self._installed = False


_callback_installed = False


def _ensure_litellm_callback() -> None:
    """Wires the litellm success_callback once per process."""
    global _callback_installed
    if _callback_installed:
        return

    try:
        import litellm
    except ImportError:
        return

    def _on_success(kwargs, response, start_time, end_time):
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        if not cost:
            return
        for tracker in list(CostTracker._active):
            tracker.add(cost)

    existing = list(getattr(litellm, "success_callback", None) or [])
    if _on_success not in existing:
        existing.append(_on_success)
    litellm.success_callback = existing
    _callback_installed = True
