"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from commerce_engine.embeddings import DeterministicEmbedder

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "test_commerce_products"


def qdrant_available() -> bool:
    """Check if Qdrant is reachable at localhost:6333."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at localhost:6333",
)


@pytest.fixture
def embedder():
    """Return a deterministic embedder for tests."""
    return DeterministicEmbedder()


@pytest.fixture
def qdrant_client():
    """Return a QdrantClient connected to local Qdrant, if available."""
    return QdrantClient(url=QDRANT_URL)


@pytest.fixture
def test_collection(qdrant_client):
    """Create a fresh test collection and clean up after."""
    from commerce_engine.qdrant_store import recreate_collection

    recreate_collection(qdrant_client, TEST_COLLECTION)
    yield TEST_COLLECTION
    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name=TEST_COLLECTION)
    except Exception:
        pass
