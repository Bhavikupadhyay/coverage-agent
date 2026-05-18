"""
Token-per-minute throttling for LiteLLM completions.

Free tiers (especially Groq) cap total tokens per minute per model. Our pipeline
fires 3-5 LLM calls per gap with prompts in the 3-5k range, so a 5-gap run
easily blows past a 12000 TPM ceiling. LiteLLM's own retry handles transient
429s, but it does so with a fixed delay and no awareness of the rolling window.

This module installs a `pre_call_check` callback on litellm that sleeps just
enough to keep the per-model token usage below the configured TPM cap.

Defaults match Groq's free tier as of 2026-05. Override via env:
    COVERAGE_AGENT_TPM_LIMIT=12000
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)


_DEFAULT_TPM = {
    "groq/llama-3.3-70b-versatile": 12000,
    "groq/llama-3.1-8b-instant": 6000,
    "groq/llama-3.1-70b-versatile": 12000,
}
_FALLBACK_TPM = 6000
_WINDOW_SECONDS = 60.0
_HEADROOM_RATIO = 0.85  # stay under 85% of the cap to leave room for retries


def _tpm_for(model: str) -> int:
    override = os.environ.get("COVERAGE_AGENT_TPM_LIMIT", "").strip()
    if override.isdigit():
        return int(override)
    return _DEFAULT_TPM.get(model, _FALLBACK_TPM)


class TPMThrottle:
    """Rolling-window token throttle per model.

    Records token usage timestamped at request time. Before issuing a new
    request, waits until the rolling 60-second window has enough headroom
    for the estimated tokens of the upcoming call.
    """

    def __init__(self, model: str, tpm_limit: int | None = None) -> None:
        self.model = model
        self.tpm_limit = tpm_limit if tpm_limit is not None else _tpm_for(model)
        self._events: Deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _used_in_window(self, now: float) -> int:
        while self._events and now - self._events[0][0] > _WINDOW_SECONDS:
            self._events.popleft()
        return sum(tokens for _, tokens in self._events)

    def acquire(self, estimated_tokens: int) -> None:
        """Blocks until estimated_tokens can fit under the rolling window."""
        capacity = int(self.tpm_limit * _HEADROOM_RATIO)
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._used_in_window(now)
                if used + estimated_tokens <= capacity:
                    self._events.append((now, estimated_tokens))
                    return
                # Need to wait until the oldest event ages out
                oldest_ts, _ = self._events[0]
                sleep_for = max(1.0, _WINDOW_SECONDS - (now - oldest_ts) + 0.5)
            logger.info(
                "[TPM] %s: %d/%d used + %d est -> sleeping %.1fs",
                self.model, used, capacity, estimated_tokens, sleep_for,
            )
            time.sleep(sleep_for)

    def record_actual(self, actual_tokens: int) -> None:
        """Updates the most recent event with the actual token count.

        Called from the litellm success callback once we know the true usage.
        """
        with self._lock:
            if not self._events:
                return
            ts, _ = self._events[-1]
            self._events[-1] = (ts, actual_tokens)


_throttles: dict[str, TPMThrottle] = {}
_throttles_lock = threading.Lock()


def get_throttle(model: str) -> TPMThrottle:
    """Returns the process-wide throttle for the given model."""
    with _throttles_lock:
        t = _throttles.get(model)
        if t is None:
            t = TPMThrottle(model)
            _throttles[model] = t
        return t


def estimate_tokens(messages: list[dict] | None) -> int:
    """Cheap token estimate: 1 token ≈ 4 chars (English rule of thumb)."""
    if not messages:
        return 256
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
    return max(256, total_chars // 4 + 256)  # +256 for the completion itself


def _parse_tpd_retry_after(message: str) -> float | None:
    """Extracts seconds to wait from a Groq TPD error message.

    Groq embeds 'Please try again in Xm Ys.' or 'in Xs.' in the error body.
    Returns seconds as a float, or None if the pattern isn't found.
    """
    import re
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", message or "")
    if m:
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return None


def install_litellm_hook() -> None:
    """Wraps litellm.completion with the TPM throttle.

    Before every completion call, blocks until the rolling 60s window for
    the target model has enough headroom for the estimated token count.
    After every successful call, the recorded estimate is corrected to the
    real `usage.total_tokens` value.

    Also handles Groq's per-day (TPD) rolling limit: when a 429 contains
    'tokens per day', extracts the retry-after duration and sleeps in-process
    rather than crashing, so a long benchmark run resumes automatically.

    Idempotent. Safe to call multiple times.
    """
    try:
        import litellm
    except ImportError:
        return

    if getattr(install_litellm_hook, "_installed", False):
        return

    original_completion = litellm.completion

    def throttled_completion(*args, **kwargs):
        model = kwargs.get("model", "")
        if model:
            messages = kwargs.get("messages")
            estimated = estimate_tokens(messages)
            get_throttle(model).acquire(estimated)

        while True:
            try:
                response = original_completion(*args, **kwargs)
                break
            except Exception as exc:
                msg = str(exc)
                if "tokens per day" in msg or "TPD" in msg:
                    wait = _parse_tpd_retry_after(msg)
                    if wait is not None:
                        sleep_for = wait + 5.0  # small buffer
                        logger.warning(
                            "[TPD] daily token limit hit — sleeping %.0fs before retry", sleep_for
                        )
                        time.sleep(sleep_for)
                        continue
                raise

        if model:
            try:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    total = int(getattr(usage, "total_tokens", 0) or 0)
                    if total:
                        get_throttle(model).record_actual(total)
            except Exception:
                pass
        return response

    litellm.completion = throttled_completion
    install_litellm_hook._installed = True  # type: ignore[attr-defined]
    logger.info("TPM throttle installed on litellm.completion")
