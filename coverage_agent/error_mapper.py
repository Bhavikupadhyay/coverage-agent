"""
Translate raw exception messages from E2B / LiteLLM / pytest into one-line,
user-friendly explanations. Used by run_engine when populating RunRecord.error.

Order matters: more specific patterns first, generic fallbacks last.
"""
from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    # --- LLM auth ---
    (
        re.compile(r"AuthenticationError|invalid[\s_-]?api[\s_-]?key|401", re.IGNORECASE),
        "LLM API key is invalid or expired. Double-check your key in the BYOK form or .env.",
    ),
    (
        re.compile(r"rate[\s_-]?limit|429|RateLimitError", re.IGNORECASE),
        "LLM rate limit hit. Free tiers are bursty — wait a minute and retry, or supply your own key in BYOK mode.",
    ),
    (
        re.compile(r"context[\s_-]?length|maximum context length|tokens.*exceed", re.IGNORECASE),
        "Context window exceeded. Try a smaller repo or a model with a larger context window.",
    ),
    (
        re.compile(r"timed?\s?out|timeout|deadline exceeded", re.IGNORECASE),
        "Operation timed out. The repo's test suite or dependency install may be too slow for the free sandbox tier.",
    ),

    # --- E2B sandbox ---
    (
        re.compile(r"E2B|sandbox.*not\s?found|sandbox.*closed", re.IGNORECASE),
        "Sandbox error. The E2B VM may have been terminated or the key is invalid.",
    ),
    (
        re.compile(r"git clone failed|fatal:.*repository", re.IGNORECASE),
        "Could not clone the repository. Confirm the URL is public and reachable.",
    ),
    (
        re.compile(r"pip.*install.*fail|ERROR: Could not.*build|Could not find a version", re.IGNORECASE),
        "Repository dependencies failed to install in the sandbox. The project may require system packages or a Python version mismatch.",
    ),
    (
        re.compile(r"No module named", re.IGNORECASE),
        "A Python module the repo expects is missing in the sandbox. Check that the project's dev dependencies install cleanly.",
    ),

    # --- pytest / coverage ---
    (
        re.compile(r"no tests ran|collected 0 items", re.IGNORECASE),
        "No pytest tests were found in the repository. CoverageAgent needs an existing test suite to measure baseline coverage.",
    ),
    (
        re.compile(r"pytest.*not found|command not found.*pytest", re.IGNORECASE),
        "pytest is not installed in the repository's environment.",
    ),
    (
        re.compile(r"coverage.*command not found", re.IGNORECASE),
        "coverage.py is not installed in the sandbox.",
    ),
    (
        re.compile(r"Not a Python repository", re.IGNORECASE),
        "Not a Python project — no .py files found in the repository.",
    ),

    # --- Network ---
    (
        re.compile(r"ConnectionError|getaddrinfo failed|Network is unreachable", re.IGNORECASE),
        "Network error reaching the LLM or sandbox provider. Check your internet connection or try again.",
    ),
]


def friendly_error(raw_error: str | Exception) -> str:
    """Returns a user-facing message for an underlying error.

    Always returns something non-empty. Falls back to the truncated raw error
    if no rule matches.
    """
    text = str(raw_error or "").strip()
    if not text:
        return "Unknown error."

    for pattern, message in _RULES:
        if pattern.search(text):
            return message

    if len(text) > 240:
        return text[:240].rsplit(" ", 1)[0] + "…"
    return text
