import json
import sqlite3
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent.parent / "coverage_agent_runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'queued',
    logs         TEXT NOT NULL DEFAULT '[]',
    scorecard    TEXT NOT NULL DEFAULT '{}',
    recommendations TEXT NOT NULL DEFAULT '[]',
    results      TEXT NOT NULL DEFAULT '[]',
    error        TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def init_db(db_path: str | Path = _DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def upsert_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    logs: list[str],
    scorecard: dict,
    recommendations: list[str],
    results: list[dict],
    error: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs"
        " (run_id, status, logs, scorecard, recommendations, results, error)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            status,
            json.dumps(logs),
            json.dumps(scorecard),
            json.dumps(recommendations),
            json.dumps(results),
            error,
        ),
    )
    conn.commit()


def load_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute(
        "SELECT run_id, status, logs, scorecard, recommendations, results, error"
        " FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "status": row[1],
        "logs": json.loads(row[2]),
        "scorecard": json.loads(row[3]),
        "recommendations": json.loads(row[4]),
        "results": json.loads(row[5]),
        "error": row[6],
    }


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT run_id, status, logs, scorecard, recommendations, results, error"
        " FROM runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "run_id": r[0],
            "status": r[1],
            "logs": json.loads(r[2]),
            "scorecard": json.loads(r[3]),
            "recommendations": json.loads(r[4]),
            "results": json.loads(r[5]),
            "error": r[6],
        }
        for r in rows
    ]
