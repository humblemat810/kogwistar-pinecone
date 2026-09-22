import sys

import pytest

pytestmark = pytest.mark.skipif(sys.implementation.name != "pypy", reason="PyPy compatibility lane only")


def test_provider_free_import_and_contract_surface():
    from kogwistar_pinecone import PineconeBackend

    backend = object.__new__(PineconeBackend)
    assert backend.__class__.__name__ == "PineconeBackend"
    assert hasattr(PineconeBackend, "from_env")
