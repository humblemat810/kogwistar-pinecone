from kogwistar_pinecone import PineconeBackend


class FakeIndex:
    def __init__(self):
        self.rows = {}

    def upsert(self, vectors, namespace):
        for row in vectors:
            self.rows[(namespace, row["id"])] = row

    def fetch(self, ids, namespace):
        return {"vectors": {i: self.rows[(namespace, i)] for i in ids if (namespace, i) in self.rows}}

    def query(self, vector, top_k, namespace, filter=None, include_metadata=True, include_values=False):
        matches = []
        for (ns, id_), row in self.rows.items():
            if ns != namespace:
                continue
            if filter and row["metadata"].get("doc_id") != filter.get("doc_id"):
                continue
            score = sum(a * b for a, b in zip(vector, row["values"]))
            matches.append({"id": id_, "score": score, "metadata": row["metadata"], "values": row["values"]})
        return {"matches": sorted(matches, key=lambda x: x["score"], reverse=True)[:top_k]}

    def update(self, namespace, id, values=None, set_metadata=None):
        row = self.rows[(namespace, id)]
        if values is not None:
            row["values"] = values
        if set_metadata:
            row["metadata"].update(set_metadata)

    def delete(self, ids=None, namespace=None, filter=None):
        if ids is not None:
            for id_ in ids:
                self.rows.pop((namespace, id_), None)
        elif filter:
            for key, row in list(self.rows.items()):
                if key[0] == namespace and row["metadata"].get("doc_id") == filter.get("doc_id"):
                    del self.rows[key]


def test_id_get_filter_vector_query_update_delete():
    backend = PineconeBackend(FakeIndex(), dimension=3)
    backend.node_add(
        ids=["n1", "n2"], documents=["alpha", "beta"],
        metadatas=[{"doc_id": "d1", "kind": "a"}, {"doc_id": "d1", "kind": "b"}],
        embeddings=[[1, 0, 0], [0, 1, 0]],
    )
    assert backend.node_get(ids=["n1"])["ids"] == ["n1"]
    assert set(backend.node_get(where={"doc_id": "d1"})["ids"]) == {"n1", "n2"}
    assert backend.node_query(query_embeddings=[[1, 0, 0]], n_results=1)["ids"][0] == ["n1"]
    backend.node_update(ids=["n1"], metadatas=[{"new": True}])
    assert backend.node_get(ids=["n1"])["metadatas"][0]["kind"] == "a"
    backend.node_delete(ids=["n2"])
    assert backend.node_get(ids=["n2"])["ids"] == []

