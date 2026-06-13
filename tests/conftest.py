"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from commerce_engine.embeddings import DeterministicEmbedder

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "test_commerce_products"


def qdrant_available() -> bool:
    """Always return True because we support fallback to in-memory Qdrant client."""
    return True


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant capabilities not available",
)


@pytest.fixture
def embedder():
    """Return a deterministic embedder for tests."""
    return DeterministicEmbedder()


@pytest.fixture
def qdrant_client():
    """Return a QdrantClient connected to local Qdrant, or in-memory fallback."""
    try:
        # Try connecting to local server first
        client = QdrantClient(url=QDRANT_URL, timeout=1)
        client.get_collections()
        return client
    except Exception:
        # Fallback to in-memory client
        client = QdrantClient(location=":memory:")
        from commerce_engine.qdrant_store import recreate_collection
        recreate_collection(client, "commerce_products")
        return client


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
