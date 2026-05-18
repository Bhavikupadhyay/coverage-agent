from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coverage_agent.credentials import Credentials


def get_sandbox(creds: "Credentials"):
    """Factory: returns LocalSandbox or E2BSandbox based on creds.sandbox_mode."""
    if creds.sandbox_mode == "local":
        from coverage_agent.sandbox.local_runner import LocalSandbox
        return LocalSandbox(offline=creds.is_offline)
    from coverage_agent.sandbox.e2b_runner import E2BSandbox
    return E2BSandbox("", e2b_api_key=creds.e2b_api_key, offline=creds.is_offline)
