import os


def is_dry_run() -> bool:
    return os.environ.get("DRY_RUN", "false").lower() == "true"


def get_model() -> str:
    return os.environ.get("COVERAGE_AGENT_MODEL", "gemini/gemini-2.5-flash")
