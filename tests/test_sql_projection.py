import pytest

from kogwistar_pinecone import PineconeBackend
from kogwistar_pinecone.sql_meta import SQLiteProjectionHarness, SQLiteProjectionMeta
from test_contract import FakeIndex


def test_sql_commit_rollback_retry_and_replay(tmp_path):
    meta = SQLiteProjectionMeta(tmp_path / "meta.sqlite")
    backend = PineconeBackend(FakeIndex(), dimension=3, prefix="sql")
    harness = SQLiteProjectionHarness(meta, backend)
    with pytest.raises(RuntimeError):
        with meta.transaction():
            meta.append_event(event_id="rolled-back", collection="node", entity_id="r", op="UPSERT", payload={})
            raise RuntimeError("rollback")
    assert meta.event_count() == 0

    harness.enqueue_upsert(event_id="e1", collection="node", entity_id="n1", document="truth", metadata={"source": "sql"}, vector=[1, 0, 0])
    assert meta.status("e1") == "PENDING"
    claimed = meta.claim()
    assert claimed is not None
    meta.requeue(claimed[0])
    assert meta.status("e1") == "PENDING"
    with pytest.raises(RuntimeError):
        harness.project_once(crash_after_write=True)
    assert meta.status("e1") == "PENDING"
    assert harness.project_once() is True
    assert meta.status("e1") == "DONE"
    assert backend.node_get(ids=["n1"])["ids"] == ["n1"]
    backend.node_upsert(ids=["n1"], documents=["truth"], metadatas=[{"source": "sql"}], embeddings=[[1, 0, 0]])
    assert backend.node_get(ids=["n1"])["ids"] == ["n1"]
