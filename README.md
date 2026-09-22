# kogwistar-pinecone

Standalone Pinecone adapter for Kogwistar's Chroma-shaped backend surface.

## Local test

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```

Tests include a SQLite SQL-meta/outbox harness covering rollback, durable
pending work, requeue/retry, commit, and idempotent upsert replay. The harness
is contract coverage, not the production Kogwistar meta schema.

For Pinecone Local integration:

```powershell
docker compose up -d pinecone
$env:PINECONE_INDEX_HOST = "http://127.0.0.1:5081"
$env:PINECONE_API_KEY = "pclocal"
.\.venv\Scripts\python.exe -m pytest -m integration
docker compose down
```

The compose service uses Pinecone's official `pinecone-index` in-memory emulator
(dense, cosine, dimension 3). CI starts the same emulator and fails when the
real Pinecone SDK adapter contract fails. Data is intentionally non-persistent.

Pinecone is an eventually-consistent projection here. `transaction()` is a
no-op; authoritative graph/event state must remain in Kogwistar's transactional
store. One provider index is partitioned into namespaces, one per logical
collection. Non-vector materializations use a sentinel vector and filtered
query, because Pinecone records require a vector.
