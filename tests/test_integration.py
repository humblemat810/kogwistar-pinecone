import os

import pytest

from kogwistar_pinecone import PineconeBackend


@pytest.mark.integration
def test_pinecone_server_contract():
    host = os.getenv("PINECONE_INDEX_HOST")
    if not host:
        pytest.skip("PINECONE_INDEX_HOST not set")
    try:
        backend = PineconeBackend.from_env(index_host=host, dimension=3, prefix="integration")
    except Exception as exc:
        pytest.skip(f"Pinecone service unavailable: {exc}")
    backend.node_upsert(ids=["server-node"], documents=["server"], metadatas=[{"doc_id": "server-doc"}], embeddings=[[1, 0, 0]])
    assert backend.node_get(ids=["server-node"])["ids"] == ["server-node"]
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=1)["ids"][0] == ["server-node"]
