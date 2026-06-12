from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from commerce_engine.embeddings import Embedder
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ingest import product_point
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import recreate_collection, upsert_products
from commerce_engine.search import search_products


@dataclass
class BenchmarkResult:
    profile: str
    query_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    recall_at_3: float
    storage_size_bytes: int


def run_benchmark(
    client: QdrantClient,
    collection: str,
    root: Path,
    embedder: Embedder,
    profile: str,
) -> BenchmarkResult:
    benchmark_collection = f"{collection}_{profile}_benchmark"
    recreate_collection(
        client, benchmark_collection, profile=profile, disable_text_hnsw=profile != "hnsw"
    )
    products = ensure_fixture_images(root)
    points = [product_point(product, embedder) for product in products]
    upsert_products(client, benchmark_collection, points)
    storage_size_bytes = _estimated_vector_storage_bytes(points, profile)

    queries = [
        ("Waterproof black hiking boots with good arch support", "boot-001"),
        ("eco friendly waterproof hiking boots", "boot-002"),
        ("black travel sneakers for light rain", "shoe-003"),
    ]
    latencies: list[float] = []
    hits = 0
    for query, expected_id in queries:
        start = time.perf_counter()
        results = search_products(
            client,
            benchmark_collection,
            SearchRequest(
                query=query,
                user_id="user_a",
                filters=SearchFilters(availability=None),
                limit=3,
            ),
            embedder,
        )
        latencies.append((time.perf_counter() - start) * 1000)
        if expected_id in {result.product_id for result in results[:3]}:
            hits += 1

    return BenchmarkResult(
        profile=profile,
        query_count=len(queries),
        mean_latency_ms=statistics.mean(latencies),
        p95_latency_ms=max(latencies),
        recall_at_3=hits / len(queries),
        storage_size_bytes=storage_size_bytes,
    )


def result_dict(result: BenchmarkResult) -> dict:
    return asdict(result)


def _estimated_vector_storage_bytes(points, profile: str) -> int:
    total_floats = 0
    for point in points:
        for matrix in point.vector.values():
            total_floats += sum(len(vector) for vector in matrix)
    if profile == "scalar":
        return total_floats
    if profile == "binary":
        return (total_floats + 7) // 8
    return total_floats * 4
