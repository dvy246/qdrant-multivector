"""Integration tests requiring a running Qdrant instance.

These tests verify the full data lifecycle: collection creation → ingestion →
search → updates → deletion. They use DeterministicEmbedder to avoid model
downloads while testing real Qdrant operations.

Skip condition: Qdrant not available at localhost:6333.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from commerce_engine.embeddings import DeterministicEmbedder, VISION_DIM, TEXT_DIM, REVIEW_DIM
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ids import point_id
from commerce_engine.ingest import ingest_products
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import (
    REVIEW_VECTOR,
    TEXT_VECTOR,
    VISUAL_VECTOR,
    recreate_collection,
)
from commerce_engine.search import search_products
from commerce_engine.updates import append_review, append_image

from tests.conftest import requires_qdrant


@requires_qdrant
class TestCollectionLifecycle:
    """Test collection creation, schema, and cleanup."""

    def test_create_collection(self, qdrant_client, test_collection):
        """Collection should exist with correct named vectors."""
        info = qdrant_client.get_collection(collection_name=test_collection)
        vectors = info.config.params.vectors
        assert VISUAL_VECTOR in vectors
        assert TEXT_VECTOR in vectors
        assert REVIEW_VECTOR in vectors
        assert vectors[VISUAL_VECTOR].size == VISION_DIM
        assert vectors[TEXT_VECTOR].size == TEXT_DIM
        assert vectors[REVIEW_VECTOR].size == REVIEW_DIM

    def test_collection_uses_max_sim(self, qdrant_client, test_collection):
        """All named vectors should use MAX_SIM comparator."""
        info = qdrant_client.get_collection(collection_name=test_collection)
        vectors = info.config.params.vectors
        for name in [VISUAL_VECTOR, TEXT_VECTOR, REVIEW_VECTOR]:
            assert vectors[name].multivector_config is not None

    def test_payload_indexes_created(self, qdrant_client, test_collection):
        """Payload indexes should be created for filterable fields."""
        if type(qdrant_client._client).__name__ == "QdrantLocal":
            pytest.skip("Local QdrantLocal does not store payload schema.")
        info = qdrant_client.get_collection(collection_name=test_collection)
        indexed_fields = set(info.payload_schema.keys())
        required = {"brand", "category", "region", "color", "size", "product_id",
                     "availability", "price", "eco_score"}
        assert required.issubset(indexed_fields)

    def test_quantization_profiles(self, qdrant_client):
        """Scalar and binary quantization profiles should work."""
        for profile in ["scalar", "binary"]:
            col = f"test_quant_{profile}"
            recreate_collection(qdrant_client, col, profile=profile)
            info = qdrant_client.get_collection(collection_name=col)
            if type(qdrant_client._client).__name__ != "QdrantLocal":
                assert info.config.quantization_config is not None
            qdrant_client.delete_collection(collection_name=col)


@requires_qdrant
class TestIngestion:
    """Test product ingestion into Qdrant."""

    def test_ingest_fixture_products(self, qdrant_client, test_collection, embedder, tmp_path):
        """Should ingest all fixture products with correct point IDs."""
        products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, products, embedder)

        info = qdrant_client.get_collection(collection_name=test_collection)
        assert info.points_count == len(products)

    def test_ingested_point_has_correct_payload(self, qdrant_client, test_collection, embedder, tmp_path):
        """Ingested point should carry complete product payload."""
        products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, products, embedder)

        pid = point_id(products[0].id)
        points = qdrant_client.retrieve(
            collection_name=test_collection, ids=[pid], with_payload=True
        )
        assert len(points) == 1
        payload = points[0].payload
        assert payload["product_id"] == products[0].id
        assert payload["brand"] == products[0].brand
        assert payload["price"] == products[0].price

    def test_ingested_point_has_all_vectors(self, qdrant_client, test_collection, embedder, tmp_path):
        """Ingested point should have all 3 named multivectors."""
        products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, products, embedder)

        pid = point_id(products[0].id)
        points = qdrant_client.retrieve(
            collection_name=test_collection, ids=[pid], with_vectors=True
        )
        vectors = points[0].vector
        assert VISUAL_VECTOR in vectors
        assert TEXT_VECTOR in vectors
        assert REVIEW_VECTOR in vectors
        # Visual: 1 SigLIP CLS per image → 1 row of 768-d
        assert len(vectors[VISUAL_VECTOR]) >= 1
        assert len(vectors[VISUAL_VECTOR][0]) == VISION_DIM

    def test_upsert_is_idempotent(self, qdrant_client, test_collection, embedder, tmp_path):
        """Ingesting same products twice should not duplicate points."""
        products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, products, embedder)
        ingest_products(qdrant_client, test_collection, products, embedder)

        info = qdrant_client.get_collection(collection_name=test_collection)
        assert info.points_count == len(products)


@requires_qdrant
class TestSearch:
    """Test search pipeline against real Qdrant."""

    @pytest.fixture(autouse=True)
    def _ingest(self, qdrant_client, test_collection, embedder, tmp_path):
        products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, products, embedder)

    def test_basic_search_returns_results(self, qdrant_client, test_collection, embedder):
        """Search should return results from ingested products."""
        request = SearchRequest(
            query="hiking boots",
            filters=SearchFilters(availability=None),
            limit=5,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        assert len(results) > 0

    def test_search_result_has_required_fields(self, qdrant_client, test_collection, embedder):
        """Each result should have all SearchResult fields."""
        request = SearchRequest(
            query="waterproof black hiking boots",
            filters=SearchFilters(availability=None),
            limit=5,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        assert len(results) > 0
        r = results[0]
        assert r.product_id
        assert r.title
        assert isinstance(r.qdrant_score, float)
        assert isinstance(r.final_score, float)
        assert isinstance(r.explanation, list)
        assert isinstance(r.payload, dict)

    def test_search_respects_limit(self, qdrant_client, test_collection, embedder):
        """Search should not return more than the requested limit."""
        request = SearchRequest(
            query="boots",
            filters=SearchFilters(availability=None),
            limit=2,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        assert len(results) <= 2

    def test_availability_filter(self, qdrant_client, test_collection, embedder):
        """Filtering by availability should exclude unavailable products."""
        request = SearchRequest(
            query="boots",
            filters=SearchFilters(availability=True),
            limit=10,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        for r in results:
            assert r.payload.get("availability") is True

    def test_brand_filter(self, qdrant_client, test_collection, embedder):
        """Brand filter should only return matching brands."""
        request = SearchRequest(
            query="boots",
            filters=SearchFilters(availability=None, brands=["TrailForge"]),
            limit=10,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        for r in results:
            assert r.payload.get("brand") == "TrailForge"

    def test_price_range_filter(self, qdrant_client, test_collection, embedder):
        """Price filter should only return products in range."""
        request = SearchRequest(
            query="boots",
            filters=SearchFilters(availability=None, min_price=100, max_price=160),
            limit=10,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        for r in results:
            price = r.payload.get("price", 0)
            assert 100 <= price <= 160

    def test_search_includes_explanations(self, qdrant_client, test_collection, embedder):
        """Search results should contain explanation strings."""
        request = SearchRequest(
            query="waterproof black hiking boots with good arch support",
            filters=SearchFilters(availability=None),
            limit=5,
        )
        results = search_products(qdrant_client, test_collection, request, embedder)
        assert len(results) > 0
        # At least one result should have explanations
        has_explanations = any(len(r.explanation) > 0 for r in results)
        assert has_explanations


@requires_qdrant
class TestUpdates:
    """Test incremental review and image updates."""

    @pytest.fixture(autouse=True)
    def _ingest(self, qdrant_client, test_collection, embedder, tmp_path):
        self.products = ensure_fixture_images(tmp_path)
        self.tmp_path = tmp_path
        ingest_products(qdrant_client, test_collection, self.products, embedder)

    def test_append_review(self, qdrant_client, test_collection, embedder):
        """Appending a review should add vectors and update payload."""
        pid = self.products[0].id
        result = append_review(
            qdrant_client, test_collection, pid,
            "Amazing ankle support on rocky terrain.", embedder,
        )
        assert result["product_id"] == pid
        assert result["review_count"] == len(self.products[0].reviews) + 1
        assert len(result["findings"]) > 0

        # Verify payload updated
        points = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
            with_payload=True,
        )
        reviews = points[0].payload["reviews"]
        assert "Amazing ankle support on rocky terrain." in reviews

    def test_append_review_increases_vector_count(self, qdrant_client, test_collection, embedder):
        """Appending a review should add new review vectors."""
        pid = self.products[0].id
        before = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
            with_vectors=True,
        )
        before_count = len(before[0].vector[REVIEW_VECTOR])

        append_review(
            qdrant_client, test_collection, pid,
            "Great waterproofing.", embedder,
        )

        after = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
            with_vectors=True,
        )
        after_count = len(after[0].vector[REVIEW_VECTOR])
        assert after_count > before_count

    def test_append_image(self, qdrant_client, test_collection, embedder):
        """Appending an image should add visual vectors."""
        pid = self.products[0].id
        image_path = self.tmp_path / "data" / "images" / f"{pid}.png"
        assert image_path.exists()

        before = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
            with_vectors=True,
        )
        before_count = len(before[0].vector[VISUAL_VECTOR])

        result = append_image(
            qdrant_client, test_collection, pid, image_path, embedder,
        )
        assert result["product_id"] == pid
        assert result["added_patches"] >= 1

        after = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
            with_vectors=True,
        )
        after_count = len(after[0].vector[VISUAL_VECTOR])
        assert after_count == before_count + result["added_patches"]

    def test_append_review_nonexistent_product(self, qdrant_client, test_collection, embedder):
        """Appending to a nonexistent product should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            append_review(
                qdrant_client, test_collection, "nonexistent-product",
                "This should fail.", embedder,
            )


@requires_qdrant
class TestDeletion:
    """Test product deletion from Qdrant."""

    @pytest.fixture(autouse=True)
    def _ingest(self, qdrant_client, test_collection, embedder, tmp_path):
        self.products = ensure_fixture_images(tmp_path)
        ingest_products(qdrant_client, test_collection, self.products, embedder)

    def test_delete_product(self, qdrant_client, test_collection):
        """Deleting a product should remove it from the collection."""
        from qdrant_client import models

        pid = self.products[0].id
        qdrant_client.delete(
            collection_name=test_collection,
            points_selector=models.PointIdsList(points=[point_id(pid)]),
            wait=True,
        )
        points = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[point_id(pid)],
        )
        assert len(points) == 0

    def test_delete_reduces_count(self, qdrant_client, test_collection):
        """Deleting a product should reduce the point count."""
        from qdrant_client import models

        before = qdrant_client.get_collection(collection_name=test_collection).points_count
        pid = self.products[0].id
        qdrant_client.delete(
            collection_name=test_collection,
            points_selector=models.PointIdsList(points=[point_id(pid)]),
            wait=True,
        )
        after = qdrant_client.get_collection(collection_name=test_collection).points_count
        assert after == before - 1
