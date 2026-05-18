import os

import litellm

# Retry on 429/503. LiteLLM reads Retry-After from the response and waits the
# server-specified delay between attempts.
litellm.num_retries = 6
litellm.retry_after = 30


DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"


def is_offline_mode() -> bool:
    """Returns True if the run should skip all real LLM and E2B calls.

    Offline mode is used for local development, CI, and screenshots.
    Agents return deterministic fixture data; the sandbox runs against a local mock.

    This is a process-wide toggle. Per-run callers should prefer
    `Credentials.is_offline` to read mode explicitly.
    """
    return os.environ.get("OFFLINE_MODE", "false").lower() == "true"
