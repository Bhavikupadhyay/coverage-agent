"""
RunReport serialization.

serialize_run_report writes the RunReport to a file (if path given) and
always returns the JSON string. The CLI uses this for both --output and
the dogfood gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from coverage_agent.contracts import RunReport


def serialize_run_report(report: RunReport, path: Optional[str] = None) -> str:
    """JSON-serialize a RunReport. Writes to path if provided; always returns the string."""
    data = report.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, default=str)
    if path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json_str, encoding="utf-8")
    return json_str


def load_run_report(path: str) -> RunReport:
    """Deserializes a RunReport from a JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    return RunReport(**json.loads(raw))
