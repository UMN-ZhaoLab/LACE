"""SQLite index for runs, steps, and artifacts.

Metadata lives in SQLite for fast querying.
Large objects stay on the filesystem.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.config import LACEConfig

DB_PATH = Path(LACEConfig.ARTIFACT_DIR) / "runs.db"

DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    parent_run_id TEXT,
    cpu_name TEXT,
    spec_hash TEXT,
    status TEXT,
    created_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT,
    step_index INTEGER,
    step_name TEXT,
    status TEXT,
    input_refs TEXT,
    output_refs TEXT,
    latency_ms INTEGER,
    error TEXT,
    PRIMARY KEY (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    kind TEXT,
    path TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    created_by_step TEXT
);
"""


def ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(DDL)
        conn.commit()


def insert_run(
    run_id: str,
    parent_run_id: str | None,
    cpu_name: str,
    spec_hash: str,
    status: str,
    created_at: str,
) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, parent_run_id, cpu_name, spec_hash, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, parent_run_id, cpu_name, spec_hash, status, created_at),
        )
        conn.commit()


def update_run_status(run_id: str, status: str, completed_at: str | None = None) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        if completed_at:
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, completed_at, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, run_id),
            )
        conn.commit()


def insert_step(
    run_id: str,
    step_index: int,
    step_name: str,
    status: str,
    error: str = "",
) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO steps (run_id, step_index, step_name, status, error) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, step_index, step_name, status, error),
        )
        conn.commit()


def insert_artifact(
    artifact_id: str,
    run_id: str,
    kind: str,
    path: str,
    size_bytes: int,
    created_by_step: str,
) -> None:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO artifacts (artifact_id, run_id, kind, path, size_bytes, created_by_step) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (artifact_id, run_id, kind, path, size_bytes, created_by_step),
        )
        conn.commit()


def get_run_status(run_id: str) -> dict[str, Any] | None:
    """Return {status, completed_at} for a run, or None if not found."""
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status, completed_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def query_failed_runs(limit: int = 100) -> list[dict[str, Any]]:
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs WHERE status != 'success' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
