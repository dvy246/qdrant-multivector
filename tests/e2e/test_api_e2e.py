"""End-to-end API tests against the FastAPI application.

These tests start the real FastAPI app using TestClient, mock the Qdrant
client and embedder dependencies via overrides to use the real local Qdrant
and the deterministic embedder.
"""
from __future__ import annotations
from PIL import Image
import numpy as np

import pytest
from fastapi.testclient import TestClient
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.api import app, get_client, get_embedder

from commerce_engine.embeddings import DeterministicEmbedder
from tests.conftest import qdrant_available, requires_qdrant


@pytest.fixture
def test_client(qdrant_client):
    """Provide a TestClient with overridden dependencies."""
    def override_get_client():
        return qdrant_client

    def override_get_embedder():
        return DeterministicEmbedder()

    app.dependency_overrides[get_client] = override_get_client
    app.dependency_overrides[get_embedder] = override_get_embedder
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class TestSystemEndpoints:
    def test_health(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @requires_qdrant
    def test_ready(self, test_client):
        response = test_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @requires_qdrant
    def test_metrics(self, test_client):
        test_client.post("/admin/init", json={"profile": "baseline"})
        response = test_client.get("/metrics")
        assert response.status_code == 200, response.text
        assert "points_count" in response.json()


@requires_qdrant
class TestE2EWorkflow:
    """Test full e2e workflow: init -> ingest -> search -> update -> delete."""

    def test_full_workflow(self, test_client, tmp_path):
        # 1. Init Collection
        response = test_client.post("/admin/init", json={"profile": "baseline"})
        assert response.status_code == 200
        assert "collection" in response.json()
        
        # 2. Ingest
        products = ensure_fixture_images(tmp_path)
        products_to_ingest = [p.model_dump(mode="json") for p in products]
        response = test_client.post("/ingest/products", json=products_to_ingest)
        assert response.status_code == 200
        assert response.json()["ingested"] == len(products_to_ingest)

        # 3. Search
        response = test_client.post(
            "/search",
            json={
                "query": "hiking boots",
                "filters": {},
                "limit": 5,
            },
        )
        assert response.status_code == 200
        results = response.json()
        assert len(results) > 0
        product_id = results[0]["product_id"]

        # 4. Append Review
        response = test_client.post(
            f"/products/{product_id}/reviews",
            json={"review": "Super comfy and reliable on long trails."},
        )
        assert response.status_code == 200
        assert response.json()["product_id"] == product_id
        assert response.json()["review_count"] >= 1

        # 5. Append Image
        # First write a dummy image
        img_dir = tmp_path / "images"
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / "test.png"
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(img_path)
        
        response = test_client.post(
            f"/products/{product_id}/images",
            json={"image_path": str(img_path)},
        )
        assert response.status_code == 200
        assert response.json()["added_patches"] >= 1

        # 6. Delete
        response = test_client.delete(f"/products/{product_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] == product_id


@requires_qdrant
class TestBenchmarks:
    def test_run_benchmark(self, test_client):
        """Test the benchmark endpoint."""
        response = test_client.post(
            "/benchmarks/run",
            json={"profile": "baseline"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query_count" in data
        assert "mrr" in data
