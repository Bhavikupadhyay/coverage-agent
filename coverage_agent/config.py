import litellm

# Retry on 429/503. LiteLLM reads Retry-After from the response and waits the
# server-specified delay between attempts.
litellm.num_retries = 6
litellm.retry_after = 30

DEFAULT_MODEL = "gemini/gemini-2.5-flash"
