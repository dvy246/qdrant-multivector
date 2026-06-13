"""Multi-Aspect E-Commerce Semantic Engine — FastAPI Application."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from commerce_engine.benchmark import result_dict, run_benchmark
from commerce_engine.config import get_settings, validate_startup, Settings
from commerce_engine.embeddings import Embedder, create_embedder
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ids import point_id
from commerce_engine.ingest import ingest_products
from commerce_engine.logging_config import (
    Timer,
    generate_request_id,
    get_logger,
    request_id_var,
    setup_logging,
)
from commerce_engine.models import Product, SearchRequest, SearchResult
from commerce_engine.qdrant_store import make_client, recreate_collection
from commerce_engine.search import search_products
from commerce_engine.updates import append_image, append_review

logger = get_logger("api")

# ── Response Models ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
    errors: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    collection: str
    points_count: int
    vectors_count: int
    status: str


class IngestResponse(BaseModel):
    ingested: int


class DeleteResponse(BaseModel):
    deleted: str


class InitResponse(BaseModel):
    collection: str
    profile: str


class ReviewBody(BaseModel):
    review: str = Field(..., description="Customer review text to append")


class ImageBody(BaseModel):
    image_path: str = Field(..., description="Path to product image file")


class ErrorResponse(BaseModel):
    detail: str


# ── App Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Aspect E-Commerce Semantic Engine",
    description=(
        "Production-grade multi-aspect product search using Qdrant multivectors. "
        "Combines visual (ViT + SigLIP), specification (ColBERT), and review (BGE) "
        "embeddings with MaxSim late interaction scoring."
    ),
    version="1.0.0",
)


# ── Singleton Dependencies ───────────────────────────────────────────

@lru_cache
def _get_client() -> QdrantClient:
    settings = get_settings()
    return make_client(settings.qdrant_url, settings.qdrant_api_key)


@lru_cache
def _get_embedder() -> Embedder:
    settings = get_settings()
    return create_embedder(
        settings.embedding_backend,
        settings.text_late_model,
        settings.review_model,
        settings.device,
        settings.vision_alignment_model,
    )


def get_client() -> QdrantClient:
    return _get_client()


def get_embedder() -> Embedder:
    return _get_embedder()


# ── Middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Inject request ID and log request latency."""
    req_id = generate_request_id()
    request_id_var.set(req_id)
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "request completed",
        extra={"extra_data": {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(elapsed_ms, 1),
            "request_id": req_id,
        }},
    )
    return response


# ── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("starting commerce engine API", extra={"extra_data": {
        "backend": settings.embedding_backend,
        "qdrant_url": settings.qdrant_url,
        "collection": settings.qdrant_collection,
    }})


# ── Health & Readiness ───────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness check",
    description="Returns 200 if the API process is running.",
)
def health() -> HealthResponse:
    return HealthResponse()


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["ops"],
    summary="Readiness check",
    description="Verifies Qdrant connectivity and collection availability.",
    responses={503: {"model": ErrorResponse}},
)
def ready(client: QdrantClient = Depends(get_client)):
    settings = get_settings()
    errors = validate_startup(settings)
    if errors:
        return ReadyResponse(
            status="not_ready",
            qdrant=settings.qdrant_url,
            collection=settings.qdrant_collection,
            errors=errors,
        )
    return ReadyResponse(
        status="ready",
        qdrant=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["ops"],
    summary="Collection metrics",
    description="Returns Qdrant collection statistics.",
    responses={503: {"model": ErrorResponse}},
)
def metrics(client: QdrantClient = Depends(get_client)):
    settings = get_settings()
    try:
        info = client.get_collection(collection_name=settings.qdrant_collection)
        return MetricsResponse(
            collection=settings.qdrant_collection,
            points_count=info.points_count or 0,
            vectors_count=getattr(info, "vectors_count", 0) or 0,
            status=str(info.status),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Core Endpoints ───────────────────────────────────────────────────

@app.post(
    "/search",
    response_model=list[SearchResult],
    tags=["search"],
    summary="Multi-aspect product search",
    description=(
        "Decomposes the query into text/visual/review aspects, searches Qdrant "
        "with multivector late interaction, and reranks with user personalization."
    ),
)
def search(
    request: SearchRequest,
    client: QdrantClient = Depends(get_client),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    return search_products(client, settings.qdrant_collection, request, embedder)


@app.post(
    "/ingest/products",
    response_model=IngestResponse,
    tags=["catalog"],
    summary="Ingest products",
    description="Generates embeddings and upserts products into Qdrant.",
)
def ingest(
    products: list[Product] | None = None,
    client: QdrantClient = Depends(get_client),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    selected = products or ensure_fixture_images(Path.cwd())
    ingest_products(client, settings.qdrant_collection, selected, embedder)
    return IngestResponse(ingested=len(selected))


@app.post(
    "/products/{product_id}/reviews",
    tags=["updates"],
    summary="Append a review",
    description=(
        "Extracts semantic findings from the review, embeds them, and appends "
        "to the product's review vectors without rebuilding the collection."
    ),
    responses={404: {"model": ErrorResponse}},
)
def add_review(
    product_id: str,
    body: ReviewBody,
    client: QdrantClient = Depends(get_client),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    try:
        return append_review(
            client, settings.qdrant_collection, product_id, body.review, embedder
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post(
    "/products/{product_id}/images",
    tags=["updates"],
    summary="Append an image",
    description=(
        "Generates patch embeddings from the image and appends to the product's "
        "visual vectors without rebuilding the collection."
    ),
    responses={404: {"model": ErrorResponse}},
)
def add_image(
    product_id: str,
    body: ImageBody,
    client: QdrantClient = Depends(get_client),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    try:
        return append_image(
            client, settings.qdrant_collection, product_id, Path(body.image_path), embedder
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete(
    "/products/{product_id}",
    response_model=DeleteResponse,
    tags=["catalog"],
    summary="Delete a product",
    description="Removes a product from the Qdrant collection by ID.",
)
def delete_product(
    product_id: str,
    client: QdrantClient = Depends(get_client),
):
    settings = get_settings()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.PointIdsList(points=[point_id(product_id)]),
        wait=True,
    )
    return DeleteResponse(deleted=product_id)


@app.post(
    "/benchmarks/run",
    tags=["ops"],
    summary="Run benchmark",
    description="Runs a benchmark against the specified quantization profile.",
)
def benchmark(
    profile: str = "baseline",
    client: QdrantClient = Depends(get_client),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    from commerce_engine.benchmark import write_benchmark_report
    result = run_benchmark(client, settings.qdrant_collection, Path.cwd(), embedder, profile)
    write_benchmark_report(result, Path.cwd())
    return result_dict(result)


@app.post(
    "/admin/init",
    response_model=InitResponse,
    tags=["admin"],
    summary="Initialize Qdrant collection",
    description="Drops and recreates the collection with the specified quantization profile.",
)
def init_qdrant(
    profile: str = "baseline",
    client: QdrantClient = Depends(get_client),
):
    settings = get_settings()
    recreate_collection(client, settings.qdrant_collection, profile=profile)
    return InitResponse(collection=settings.qdrant_collection, profile=profile)
