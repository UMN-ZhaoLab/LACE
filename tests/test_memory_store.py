import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.memory_store import (
    TABLES,
    VectorIndexStub,
    ensure_db,
    prune_expired,
    read_memory_with_ttl,
    write_memory,
)


class TestMemoryStore(unittest.TestCase):
    def test_read_memory_with_ttl_filters_old_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/memory.db"
            ensure_db(db_path)
            now = datetime.now(timezone.utc)
            old = now - timedelta(hours=25)

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO spec2op_memory (cpu_id, content, content_hash, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                    ("cpu1", "old", "hash_old", "{}", old.isoformat()),
                )
                conn.execute(
                    "INSERT INTO spec2op_memory (cpu_id, content, content_hash, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                    ("cpu1", "new", "hash_new", "{}", now.isoformat()),
                )
                conn.commit()

            records = read_memory_with_ttl(
                table="spec2op_memory",
                cpu_id="cpu1",
                limit=10,
                ttl_hours=24,
                db_path=db_path,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["content"], "new")

    def test_write_memory_skip_duplicate_across_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/memory.db"
            write_memory(
                "spec2op_memory", "cpu1", "first", db_path=db_path, auto_prune=False
            )
            write_memory(
                "spec2op_memory", "cpu1", "second", db_path=db_path, auto_prune=False
            )
            write_memory(
                "spec2op_memory", "cpu1", "first", db_path=db_path, auto_prune=False
            )
            records = read_memory_with_ttl(
                "spec2op_memory", "cpu1", limit=10, db_path=db_path
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["content"], "first")
            self.assertEqual(records[1]["content"], "second")

    def test_prune_expired_by_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/memory.db"
            ensure_db(db_path)
            now = datetime.now(timezone.utc)
            old = now - timedelta(hours=25)

            with sqlite3.connect(db_path) as conn:
                for content, created_at in [("old_cpu1", old.isoformat()), ("new_cpu1", now.isoformat()), ("old_cpu2", old.isoformat())]:
                    conn.execute(
                        "INSERT INTO spec2op_memory (cpu_id, content, content_hash, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                        (content.split("_")[1], content, f"hash_{content}", "{}", created_at),
                    )
                conn.commit()

            removed = prune_expired("spec2op_memory", "cpu1", ttl_hours=24, db_path=db_path)
            self.assertEqual(removed, 1)

            records_cpu1 = read_memory_with_ttl(
                "spec2op_memory", "cpu1", limit=10, ttl_hours=24, db_path=db_path
            )
            self.assertEqual(len(records_cpu1), 1)

            records_cpu2 = read_memory_with_ttl(
                "spec2op_memory", "cpu2", limit=10, ttl_hours=24, db_path=db_path
            )
            self.assertEqual(len(records_cpu2), 0)  # still expired, not pruned because cpu_id filter

    def test_prune_expired_all_cpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/memory.db"
            ensure_db(db_path)
            old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()

            with sqlite3.connect(db_path) as conn:
                for cpu_id in ["cpu1", "cpu2"]:
                    conn.execute(
                        "INSERT INTO spec2op_memory (cpu_id, content, content_hash, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                        (cpu_id, "old", f"hash_{cpu_id}", "{}", old),
                    )
                conn.commit()

            removed = prune_expired("spec2op_memory", ttl_hours=24, db_path=db_path)
            self.assertEqual(removed, 2)

    def test_vector_index_stub_search(self) -> None:
        index = VectorIndexStub()
        index.add("a", [1.0, 0.0, 0.0], {"filename": "a.sv"})
        index.add("b", [0.0, 1.0, 0.0], {"filename": "b.sv"})
        index.add("c", [0.5, 0.5, 0.0], {"filename": "c.sv"})
        results = index.search([0.0, 1.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "b")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=5)

    def test_vector_index_stub_clear(self) -> None:
        index = VectorIndexStub()
        index.add("a", [1.0, 0.0])
        index.clear()
        self.assertEqual(index.search([1.0, 0.0]), [])


if __name__ == "__main__":
    unittest.main()
