from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from pinecone import Pinecone

try:
    from kogwistar.engine_core.embedding_profile import EmbeddingStorageState
    from kogwistar.engine_core.storage_backend import TwoStageProjectionCapability
except ImportError:
    @dataclass(frozen=True)
    class EmbeddingStorageState:
        backend_kind: str
        storage_scope: str
        persistent: bool
        vector_count: int
        details: tuple[str, ...] = ()

COLLECTIONS = (
    "node_index", "node", "edge", "edge_endpoints", "document", "domain",
    "node_docs", "node_refs", "edge_refs",
)
DOCUMENT_KEY = "__gke_document"
METADATA_KEY = "__gke_metadata_json"
PENDING_KEY = "__gke_embedding_pending"


class NoopUnitOfWork:
    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


class AsyncNoopUnitOfWork:
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


try:
    from kogwistar.engine_core.storage_backend import TwoStageProjectionCapability
except ImportError:
    @dataclass(frozen=True)
    class TwoStageProjectionCapability:
        supports_two_stage: bool = False
        reason: str = "Pinecone adapter has no canonical event/revision promotion"


def _awaitable(value: Any) -> Any:
    class AwaitableValue:
        def __init__(self, value: Any) -> None:
            self.value = value
        def __await__(self):
            async def done() -> Any:
                return self.value
            return done().__await__()
        def __getattr__(self, name: str) -> Any:
            return getattr(self.value, name)
        def __getitem__(self, key: Any) -> Any:
            return self.value[key]
        def __iter__(self):
            return iter(self.value)
        def __len__(self) -> int:
            return len(self.value)
        def __eq__(self, other: Any) -> bool:
            return self.value == other
    return AwaitableValue(value)


def _provider_filter(where: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Validate/forward Chroma-compatible predicates accepted by Pinecone."""
    if not where:
        return None
    if "$and" in where or "$or" in where:
        for key in ("$and", "$or"):
            if key in where:
                _provider_filter_list(where[key])
    for key, value in where.items():
        if key.startswith("$"):
            continue
        if isinstance(value, Mapping):
            allowed = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"}
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unsupported Pinecone filter operators: {sorted(unknown)}")
    return where


def _provider_filter_list(items: Sequence[Mapping[str, Any]]) -> None:
    for item in items:
        _provider_filter(item)


def _response_value(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, Mapping):
        return response.get(name, default)
    return getattr(response, name, default)


class PineconeBackend:
    """Chroma-shaped adapter over one Pinecone index and logical namespaces.

    Pinecone has no transaction spanning this index and Kogwistar metadata.
    Each logical collection is a namespace. Every row carries a dense vector;
    relation/index rows use a zero sentinel because Pinecone records require a
    vector. Non-vector query is therefore a sentinel query plus metadata filter.
    """

    supports_transactions = False
    consistency = "eventual"

    def __init__(self, index: Any, *, dimension: int, prefix: str = "kogwistar", storage_scope: str | None = None, persistent: bool = True):
        self.index = index
        self.dimension = dimension
        self.prefix = prefix
        self.uow = NoopUnitOfWork()
        self.unit_of_work = self.uow
        self.async_unit_of_work = AsyncNoopUnitOfWork()
        self.supports_two_stage = False
        self.two_stage_projection_capability = TwoStageProjectionCapability()
        self._storage_scope = storage_scope or f"pinecone:{prefix}"
        self._persistent = persistent

    @classmethod
    def from_env(cls, *, index_host: str | None = None, dimension: int, prefix: str = "kogwistar") -> "PineconeBackend":
        api_key = os.environ.get("PINECONE_API_KEY", "pclocal")
        host = index_host or os.environ.get("PINECONE_INDEX_HOST")
        if not host:
            raise ValueError("PINECONE_INDEX_HOST is required")
        scope = f"pinecone:host:{hashlib.sha256(host.encode()).hexdigest()[:16]}"
        return cls(Pinecone(api_key=api_key).Index(host=host), dimension=dimension, prefix=prefix, storage_scope=scope, persistent=True)

    def embedding_storage_scope(self) -> str:
        return self._storage_scope

    def embedding_storage_scope_aliases(self) -> tuple[str, ...]:
        return ()

    def inspect_embedding_storage(self) -> dict[str, Any]:
        stats = self.index.describe_index_stats()
        namespaces = _response_value(stats, "namespaces", {}) or {}
        counts = {key: int(_response_value(namespaces.get(self._namespace(key), {}), "vector_count", 0)) for key in ("node_index", "node", "edge", "document", "domain")}
        return EmbeddingStorageState(backend_kind="pinecone", storage_scope=self._storage_scope, persistent=self._persistent, vector_count=sum(counts.values()), details=tuple(f"{key}={count}" for key, count in counts.items()))

    def _namespace(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def _vector(self, vector: Sequence[float] | None) -> list[float]:
        return list(vector) if vector is not None else [0.0] * self.dimension

    @staticmethod
    def _encode_metadata(metadata: Mapping[str, Any], document: str) -> dict[str, Any]:
        encoded: dict[str, Any] = {
            METADATA_KEY: json.dumps(dict(metadata), ensure_ascii=False, separators=(",", ":")),
            DOCUMENT_KEY: document,
        }
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                encoded[key] = value
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                encoded[key] = value
        return encoded

    @staticmethod
    def _decode_metadata(raw: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
        document = raw.get(DOCUMENT_KEY)
        blob = raw.get(METADATA_KEY)
        if isinstance(blob, str):
            try:
                metadata = json.loads(blob)
                if isinstance(metadata, dict):
                    return document, metadata
            except json.JSONDecodeError:
                pass
        return document, {k: v for k, v in raw.items() if k not in {DOCUMENT_KEY, METADATA_KEY}}

    @staticmethod
    def _include(include: Sequence[str] | None) -> set[str]:
        return set(include or ("documents", "metadatas"))

    def _record(self, item: Any, include: set[str]) -> dict[str, Any]:
        raw = _response_value(item, "metadata", {}) or {}
        document, metadata = self._decode_metadata(raw)
        return {
            "id": str(_response_value(item, "id")),
            "document": document,
            "metadata": metadata,
            "embedding": None if raw.get(PENDING_KEY) else _response_value(item, "values"),
            "distance": 1.0 - float(_response_value(item, "score", 0.0)),
        }

    def _flat(self, items: Sequence[Any], include: set[str]) -> dict[str, Any]:
        rows = [self._record(item, include) for item in items]
        out: dict[str, Any] = {"ids": [row["id"] for row in rows]}
        if "documents" in include:
            out["documents"] = [row["document"] for row in rows]
        if "metadatas" in include:
            out["metadatas"] = [row["metadata"] for row in rows]
        if "embeddings" in include:
            out["embeddings"] = [row["embedding"] for row in rows]
        return out

    def _fetch(self, key: str, ids: Sequence[str], include: set[str]) -> list[Any]:
        response = self.index.fetch(ids=list(ids), namespace=self._namespace(key))
        vectors = _response_value(response, "vectors", {}) or {}
        rows = list(vectors.values())
        by_id = {str(_response_value(item, "id")): item for item in rows}
        return [by_id[id_] for id_ in ids if id_ in by_id]

    def _fetch_by_metadata(self, key: str, where: Mapping[str, Any], limit: int) -> list[Any]:
        method = getattr(self.index, "fetch_by_metadata", None)
        if callable(method):
            response = method(filter=_provider_filter(where), namespace=self._namespace(key), limit=limit)
            return list(_response_value(response, "vectors", {}).values())
        return self._search(key, [0.0] * self.dimension, top_k=limit, where=where, include={"documents", "metadatas"})

    def _search(self, key: str, vector: Sequence[float], *, top_k: int, where: Mapping[str, Any] | None, include: set[str]) -> list[Any]:
        effective = dict(where or {})
        effective.setdefault(PENDING_KEY, {"$ne": True})
        response = self.index.query(
            vector=self._vector(vector), top_k=top_k, namespace=self._namespace(key),
            filter=_provider_filter(effective), include_metadata=True,
            include_values=("embeddings" in include),
        )
        return list(_response_value(response, "matches", []) or [])

    def get(self, key: str, *, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None, include: Sequence[str] | None = None, limit: int = 200) -> dict[str, Any]:
        inc = self._include(include)
        items = self._fetch(key, ids, inc) if ids is not None else self._fetch_by_metadata(key, where or {}, limit)
        return _awaitable(self._flat(items, inc))

    def query(self, key: str, *, query_embeddings: Sequence[Sequence[float]] | None = None, n_results: int = 10, where: Mapping[str, Any] | None = None, include: Sequence[str] | None = None) -> dict[str, Any]:
        inc = self._include(include) | {"documents", "metadatas"}
        vectors = query_embeddings or [[0.0] * self.dimension]
        batches = [self._search(key, vector, top_k=n_results, where=where, include=inc) for vector in vectors]
        out: dict[str, Any] = {"ids": [[str(_response_value(item, "id")) for item in batch] for batch in batches]}
        if "documents" in inc:
            out["documents"] = [[self._record(item, inc)["document"] for item in batch] for batch in batches]
        if "metadatas" in inc:
            out["metadatas"] = [[self._record(item, inc)["metadata"] for item in batch] for batch in batches]
        if "distances" in inc:
            out["distances"] = [[self._record(item, inc)["distance"] for item in batch] for batch in batches]
        return _awaitable(out)

    def upsert(self, key: str, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[Mapping[str, Any]], embeddings: Sequence[Sequence[float]] | None = None) -> None:
        vectors = embeddings or [None] * len(ids)
        records = [
            {"id": id_, "values": self._vector(vector), "metadata": {**self._encode_metadata(metadata, document), **({PENDING_KEY: True} if vector is None else {})}}
            for id_, document, metadata, vector in zip(ids, documents, metadatas, vectors, strict=True)
        ]
        return _awaitable(self.index.upsert(vectors=records, namespace=self._namespace(key)))

    add = upsert

    def update(self, key: str, *, ids: Sequence[str], documents: Sequence[str | None] | None = None, metadatas: Sequence[Mapping[str, Any]] | None = None, embeddings: Sequence[Sequence[float]] | None = None) -> None:
        old = self.get(key, ids=ids, include=["documents", "metadatas", "embeddings"])
        if hasattr(old, "value"):
            old = old.value
        positions = {id_: n for n, id_ in enumerate(old["ids"])}
        for n, id_ in enumerate(ids):
            if id_ not in positions:
                continue
            old_n = positions[id_]
            metadata = dict(old["metadatas"][old_n] or {})
            if metadatas is not None:
                metadata.update(metadatas[n])
            document = documents[n] if documents is not None and documents[n] is not None else old["documents"][old_n]
            vector = embeddings[n] if embeddings is not None else old["embeddings"][old_n]
            self.upsert(key, ids=[id_], documents=[document], metadatas=[metadata], embeddings=[vector])
        return _awaitable(None)

    def delete(self, key: str, *, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None) -> None:
        kwargs: dict[str, Any] = {"namespace": self._namespace(key)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        elif where is not None:
            kwargs["filter"] = _provider_filter(where)
        else:
            kwargs["delete_all"] = True
        return _awaitable(self.index.delete(**kwargs))

    def call(self, collection_key: str, method: str, **kwargs: Any) -> Any:
        if collection_key not in COLLECTIONS or method not in {"get", "query", "add", "upsert", "update", "delete"}:
            raise ValueError(f"unsupported collection/method: {collection_key}.{method}")
        return getattr(self, f"{collection_key}_{method}")(**kwargs)

    def __getattr__(self, name: str) -> Any:
        for key in COLLECTIONS:
            if name.startswith(key + "_") and name[len(key) + 1:] in {"get", "query", "add", "upsert", "update", "delete"}:
                method = name[len(key) + 1:]
                return lambda **kwargs: getattr(self, method)(key, **kwargs)
        raise AttributeError(name)
