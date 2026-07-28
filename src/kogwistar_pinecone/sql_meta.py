from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class SQLiteProjectionMeta:
    """Test-only SQL truth/outbox boundary; not production Kogwistar schema."""

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS entity_events(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,collection TEXT NOT NULL,entity_id TEXT NOT NULL,op TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS projection_jobs(job_id INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,status TEXT NOT NULL DEFAULT 'PENDING',attempts INTEGER NOT NULL DEFAULT 0);
        """)
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def append_event(self, *, event_id: str, collection: str, entity_id: str, op: str, payload: Mapping[str, Any]) -> None:
        self.conn.execute("INSERT INTO entity_events(event_id,collection,entity_id,op,payload) VALUES (?,?,?,?,?)", (event_id, collection, entity_id, op, json.dumps(payload)))
        self.conn.execute("INSERT INTO projection_jobs(event_id) VALUES (?)", (event_id,))

    def claim(self) -> tuple[int, str, str] | None:
        with self.transaction():
            row = self.conn.execute("SELECT job_id,event_id FROM projection_jobs WHERE status='PENDING' ORDER BY job_id LIMIT 1").fetchone()
            if row is None:
                return None
            self.conn.execute("UPDATE projection_jobs SET status='INFLIGHT', attempts=attempts+1 WHERE job_id=?", (row[0],))
            event = self.conn.execute("SELECT collection,payload FROM entity_events WHERE event_id=?", (row[1],)).fetchone()
        return int(row[0]), str(event[0]), str(event[1])

    def mark_done(self, job_id: int) -> None:
        with self.transaction():
            self.conn.execute("UPDATE projection_jobs SET status='DONE' WHERE job_id=?", (job_id,))

    def requeue(self, job_id: int) -> None:
        with self.transaction():
            self.conn.execute("UPDATE projection_jobs SET status='PENDING' WHERE job_id=?", (job_id,))

    def status(self, event_id: str) -> str | None:
        row = self.conn.execute("SELECT status FROM projection_jobs WHERE event_id=?", (event_id,)).fetchone()
        return None if row is None else str(row[0])

    def event_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM entity_events").fetchone()[0])


class SQLiteProjectionHarness:
    def __init__(self, meta: SQLiteProjectionMeta, backend: Any):
        self.meta, self.backend = meta, backend

    def enqueue_upsert(self, *, event_id: str, collection: str, entity_id: str, document: str, metadata: Mapping[str, Any], vector: Sequence[float]) -> None:
        with self.meta.transaction():
            self.meta.append_event(event_id=event_id, collection=collection, entity_id=entity_id, op="UPSERT", payload={"id": entity_id, "document": document, "metadata": dict(metadata), "vector": list(vector)})

    def project_once(self, *, crash_after_write: bool = False) -> bool:
        claimed = self.meta.claim()
        if claimed is None:
            return False
        job_id, collection, payload_json = claimed
        payload = json.loads(payload_json)
        self.backend.call(collection, "upsert", ids=[payload["id"]], documents=[payload["document"]], metadatas=[payload["metadata"]], embeddings=[payload["vector"]])
        if crash_after_write:
            self.meta.requeue(job_id)
            raise RuntimeError("simulated crash after vector write")
        self.meta.mark_done(job_id)
        return True
