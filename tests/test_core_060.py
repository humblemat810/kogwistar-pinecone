from test_contract import FakeIndex
from kogwistar_pinecone import PineconeBackend
from kogwistar.engine_core.embedding_profile import EmbeddingStorageState
from kogwistar.engine_core.storage_backend import TwoStageProjectionCapability
import json


def test_core_060_pending_lifecycle_and_embedding_inspector():
    backend = PineconeBackend(FakeIndex(), dimension=3)
    backend.node_add(ids=["pending", "ready"], documents=["p", "r"], metadatas=[{"kind": "p"}, {"kind": "r"}], embeddings=[None, [1, 0, 0]])
    backend.node_update(ids=["pending"], metadatas=[{"lifecycle_status": "tombstoned"}])
    stored = backend.node_get(ids=["pending"], include=["embeddings", "metadatas"])
    assert stored["embeddings"] == [None]
    assert stored["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=10)["ids"] == [["ready"]]
    assert backend.embedding_storage_scope().startswith("pinecone:")
    storage = backend.inspect_embedding_storage()
    assert isinstance(storage, EmbeddingStorageState)
    assert storage.backend_kind == "pinecone"
    assert isinstance(backend.two_stage_projection_capability, TwoStageProjectionCapability)
    assert backend.two_stage_projection_capability.atomic_promotion == "eventual_reconcile"


def test_two_stage_same_store_promotion_is_revision_gated():
    class Indexing:
        def canonical_revision_payload(self, **_):
            return json.dumps({"source_fingerprint": "v1"})
        def enqueue_index_job(self, **_):
            return None
    class Embed:
        @staticmethod
        def iterative_defensive_emb(_text):
            return [0.2, 0.3, 0.4]
    class Engine:
        indexing = Indexing()
        embed = Embed()

    backend = PineconeBackend(FakeIndex(), dimension=3, engine=Engine())
    backend.node_upsert(ids=["n1"], documents=["doc"], metadatas=[{}], embeddings=[None])
    assert backend.two_stage_projection_capability.is_complete()
    backend.two_stage_projection_adapter.apply_embedding_job(entity_kind="node", entity_id="n1", op="UPSERT", payload_json=json.dumps({"source_fingerprint": "old"}))
    assert backend.node_get(ids=["n1"], include=["embeddings"])["embeddings"] == [None]
    backend.two_stage_projection_adapter.apply_embedding_job(entity_kind="node", entity_id="n1", op="UPSERT", payload_json=json.dumps({"source_fingerprint": "v1"}))
    assert backend.node_get(ids=["n1"], include=["embeddings"])["embeddings"] == [[0.2, 0.3, 0.4]]
