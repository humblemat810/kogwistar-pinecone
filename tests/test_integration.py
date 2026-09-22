import os
import time

import pytest

from kogwistar_pinecone import PineconeBackend


@pytest.mark.integration
def test_pinecone_server_contract():
    host = os.getenv("PINECONE_INDEX_HOST")
    if not host:
        pytest.skip("PINECONE_INDEX_HOST not set")
    backend = PineconeBackend.from_env(index_host=host, dimension=3, prefix="integration")
    backend.node_upsert(ids=["server-node"], documents=["server"], metadatas=[{"doc_id": "server-doc"}], embeddings=[[1, 0, 0]])

    # Pinecone is eventually consistent: wait for both fetch and ANN visibility.
    deadline = time.monotonic() + 30
    fetched = None
    queried = None
    while time.monotonic() < deadline:
        fetched = backend.node_get(ids=["server-node"])
        queried = backend.node_query(query_embeddings=[[1, 0, 0]], n_results=1)
        if fetched["ids"] == ["server-node"] and queried["ids"][0] == ["server-node"]:
            break
        time.sleep(0.5)

    assert fetched is not None and fetched["ids"] == ["server-node"]
    assert queried is not None and queried["ids"][0] == ["server-node"]
