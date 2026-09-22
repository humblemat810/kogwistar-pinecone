from test_contract import FakeIndex
from kogwistar_pinecone import PineconeBackend


def test_core_050_pending_lifecycle_and_embedding_inspector():
    backend = PineconeBackend(FakeIndex(), dimension=3)
    backend.node_add(ids=["pending", "ready"], documents=["p", "r"], metadatas=[{"kind": "p"}, {"kind": "r"}], embeddings=[None, [1, 0, 0]])
    backend.node_update(ids=["pending"], metadatas=[{"lifecycle_status": "tombstoned"}])
    stored = backend.node_get(ids=["pending"], include=["embeddings", "metadatas"])
    assert stored["embeddings"] == [None]
    assert stored["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=10)["ids"] == [["ready"]]
    assert backend.embedding_storage_scope().startswith("pinecone:")
    assert backend.inspect_embedding_storage().backend_kind == "pinecone"
