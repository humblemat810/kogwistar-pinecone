# kogwistar-pinecone

Standalone Pinecone adapter for Kogwistar's Chroma-shaped backend surface.

## Local test

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
```

For Pinecone Local integration:

```powershell
docker compose up -d pinecone
$env:PINECONE_HOST = "http://127.0.0.1:5080"
$env:PINECONE_API_KEY = "pclocal"
# Create an index through the local control plane, then set PINECONE_INDEX_HOST.
.\.venv\Scripts\python.exe -m pytest -m integration
docker compose down
```

Pinecone is an eventually-consistent projection here. `transaction()` is a
no-op; authoritative graph/event state must remain in Kogwistar's transactional
store. One provider index is partitioned into namespaces, one per logical
collection. Non-vector materializations use a sentinel vector and filtered
query, because Pinecone records require a vector.

