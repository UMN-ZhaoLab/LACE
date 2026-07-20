"""Memory storage for LLM context."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import LACEConfig


DEFAULT_DB_PATH = "memory/memory.db"
DEFAULT_MEMORY_MAX_ITEMS = LACEConfig.MEMORY_MAX_ITEMS
DEFAULT_MEMORY_MAX_CHARS = LACEConfig.MEMORY_MAX_CHARS
DEFAULT_MEMORY_TTL_HOURS = LACEConfig.MEMORY_TTL_HOURS

TABLES = {
    "spec2op_memory": """
        CREATE TABLE IF NOT EXISTS spec2op_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            meta TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """,
    "cpu_analyzer_memory": """
        CREATE TABLE IF NOT EXISTS cpu_analyzer_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            meta TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """,
    "candidate_selector_memory": """
        CREATE TABLE IF NOT EXISTS candidate_selector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            meta TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """,
}


def build_cpu_id(cpu_dir: str) -> str:
    """Build a CPU identifier from the directory path."""
    if not cpu_dir:
        return "unknown"
    return Path(cpu_dir).name or "unknown"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def ensure_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Ensure the database, tables, and indexes exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for ddl in TABLES.values():
            conn.execute(ddl)
        # Migrate legacy tables that lack content_hash.
        for table in TABLES:
            if _table_exists(conn, table) and not _column_exists(conn, table, "content_hash"):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
        # Idempotently create helper indexes for dedup and TTL pruning.
        for table in TABLES:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_dedup "
                f"ON {table} (cpu_id, content_hash)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_ttl "
                f"ON {table} (cpu_id, created_at)"
            )
        conn.commit()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def write_memory(
    table: str,
    cpu_id: str,
    content: str,
    meta: dict[str, Any] | None = None,
    db_path: str = DEFAULT_DB_PATH,
    skip_empty: bool = True,
    skip_duplicate: bool = True,
    auto_prune: bool = True,
) -> None:
    """Write a memory record to the database.

    Args:
        auto_prune: If True, prune expired records for the same *cpu_id*
            after a successful write.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown memory table: {table}")
    normalized = (content or "").strip()
    if skip_empty and not normalized:
        return
    ensure_db(db_path)
    payload = json.dumps(meta or {}, ensure_ascii=True)
    created_at = datetime.now(timezone.utc).isoformat()
    h = _content_hash(normalized)
    with sqlite3.connect(db_path) as conn:
        if skip_duplicate:
            cursor = conn.execute(
                f"SELECT 1 FROM {table} WHERE cpu_id = ? AND content_hash = ? LIMIT 1",
                (cpu_id, h),
            )
            if cursor.fetchone():
                return
        conn.execute(
            f"INSERT INTO {table} (cpu_id, content, content_hash, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cpu_id, normalized, h, payload, created_at),
        )
        conn.commit()
        if auto_prune:
            prune_expired(table, cpu_id, db_path=db_path)


def _parse_created_at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def prune_expired(
    table: str,
    cpu_id: str | None = None,
    ttl_hours: int | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Delete expired records from *table* and return the number of rows removed.

    If *cpu_id* is None, prune for **all** CPUs.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown memory table: {table}")
    path = Path(db_path)
    if not path.exists():
        return 0
    resolved_ttl = ttl_hours if ttl_hours is not None else LACEConfig.MEMORY_TTL_HOURS
    if resolved_ttl is None or resolved_ttl <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=resolved_ttl)).isoformat()
    with sqlite3.connect(db_path) as conn:
        if cpu_id is not None:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE cpu_id = ? AND created_at < ?",
                (cpu_id, cutoff),
            )
        else:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE created_at < ?",
                (cutoff,),
            )
        conn.commit()
        return cursor.rowcount


def read_memory_with_ttl(
    table: str,
    cpu_id: str,
    limit: int | None = None,
    ttl_hours: int | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Read memory records from the database with TTL filtering."""
    if table not in TABLES:
        raise ValueError(f"Unknown memory table: {table}")
    path = Path(db_path)
    if not path.exists():
        return []

    resolved_limit = limit if limit is not None else LACEConfig.MEMORY_MAX_ITEMS
    resolved_ttl = ttl_hours if ttl_hours is not None else LACEConfig.MEMORY_TTL_HOURS

    cutoff_iso: str | None = None
    if resolved_ttl is not None and resolved_ttl > 0:
        cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(hours=resolved_ttl)
        ).isoformat()

    with sqlite3.connect(db_path) as conn:
        if cutoff_iso:
            cursor = conn.execute(
                f"SELECT content, meta, created_at FROM {table} "
                "WHERE cpu_id = ? AND created_at >= ? "
                "ORDER BY id DESC LIMIT ?",
                (cpu_id, cutoff_iso, resolved_limit),
            )
        else:
            cursor = conn.execute(
                f"SELECT content, meta, created_at FROM {table} WHERE cpu_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (cpu_id, resolved_limit),
            )
        rows = cursor.fetchall()

    results: list[dict[str, Any]] = []
    for content, meta_text, created_at in rows:
        try:
            meta = json.loads(meta_text)
        except json.JSONDecodeError:
            meta = {}
        results.append({"content": content, "meta": meta, "created_at": created_at})

    return list(reversed(results))


def read_memory(
    table: str,
    cpu_id: str,
    limit: int | None = None,
    db_path: str = DEFAULT_DB_PATH,
    ttl_hours: int | None = None,
) -> list[dict[str, Any]]:
    """Read memory records from the database."""
    return read_memory_with_ttl(
        table=table,
        cpu_id=cpu_id,
        limit=limit,
        ttl_hours=ttl_hours,
        db_path=db_path,
    )


def _truncate_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def build_memory_block(
    records: list[dict[str, Any]],
    label: str,
    max_items: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Build a bounded memory block for LLM context."""
    if not records:
        return ""
    resolved_max_items = max_items if max_items is not None else LACEConfig.MEMORY_MAX_ITEMS
    resolved_max_chars = max_chars if max_chars is not None else LACEConfig.MEMORY_MAX_CHARS
    bounded = records[-resolved_max_items:]
    header = (
        f"{label} memory (most recent last). "
        "Use as hints; validate before applying."
    )
    lines = [header]
    for item in bounded:
        content = item.get("content", "")
        created_at = item.get("created_at", "unknown")
        lines.append(f"- {created_at}: {content}")
    while len("\n".join(lines)) > resolved_max_chars and len(lines) > 1:
        lines.pop(1)
    block = "\n".join(lines)
    if len(block) > resolved_max_chars:
        block = _truncate_line(block, resolved_max_chars)
    return block


def format_memory(records: list[dict[str, Any]], label: str) -> str:
    """Format memory records as a string for LLM context."""
    return build_memory_block(records, label)


class VectorIndexStub:
    """In-memory vector index stub for similarity search.

    This is a placeholder implementation that performs brute-force cosine
    similarity.  It can be swapped for ``sqlite-vec``, ``faiss``, or a
    Neo4j-native vector index once those dependencies are available.
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(
        self,
        item_id: str,
        embedding: list[float],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index."""
        self._items.append(
            {"id": item_id, "embedding": embedding, "meta": meta or {}}
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar items."""
        scored = []
        for item in self._items:
            sim = self._cosine_similarity(item["embedding"], query_embedding)
            scored.append({**item, "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        """Remove all items from the index."""
        self._items.clear()
