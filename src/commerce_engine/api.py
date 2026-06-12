from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import models

from commerce_engine.benchmark import result_dict, run_benchmark
from commerce_engine.config import get_settings
from commerce_engine.embeddings import create_embedder
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ids import point_id
from commerce_engine.ingest import ingest_products
from commerce_engine.models import Product, SearchRequest
from commerce_engine.qdrant_store import make_client, recreate_collection
from commerce_engine.search import search_products
from commerce_engine.updates import append_image, append_review

app = FastAPI(title="Multi-Aspect E-Commerce Semantic Engine")


class ReviewBody(BaseModel):
    review: str


class ImageBody(BaseModel):
    image_path: str


def dependencies():
    settings = get_settings()
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    embedder = create_embedder(
        settings.embedding_backend,
        settings.text_late_model,
        settings.review_model,
        settings.vision_model,
        settings.device,
    )
    return settings, client, embedder


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest/products")
def ingest(products: list[Product] | None = None) -> dict[str, int]:
    settings, client, embedder = dependencies()
    selected = products or ensure_fixture_images(Path.cwd())
    ingest_products(client, settings.qdrant_collection, selected, embedder)
    return {"ingested": len(selected)}


@app.post("/search")
def search(request: SearchRequest):
    settings, client, embedder = dependencies()
    return search_products(client, settings.qdrant_collection, request, embedder)


@app.post("/products/{product_id}/reviews")
def add_review(product_id: str, body: ReviewBody):
    settings, client, embedder = dependencies()
    return append_review(client, settings.qdrant_collection, product_id, body.review, embedder)


@app.post("/products/{product_id}/images")
def add_image(product_id: str, body: ImageBody):
    settings, client, embedder = dependencies()
    return append_image(
        client, settings.qdrant_collection, product_id, Path(body.image_path), embedder
    )


@app.delete("/products/{product_id}")
def delete_product(product_id: str) -> dict[str, str]:
    settings, client, _ = dependencies()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.PointIdsList(points=[point_id(product_id)]),
        wait=True,
    )
    return {"deleted": product_id}


@app.post("/benchmarks/run")
def benchmark(profile: str = "baseline"):
    settings, client, embedder = dependencies()
    return result_dict(
        run_benchmark(client, settings.qdrant_collection, Path.cwd(), embedder, profile)
    )


@app.post("/admin/init")
def init_qdrant(profile: str = "baseline") -> dict[str, str]:
    settings, client, _ = dependencies()
    recreate_collection(client, settings.qdrant_collection, profile=profile)
    return {"collection": settings.qdrant_collection, "profile": profile}
