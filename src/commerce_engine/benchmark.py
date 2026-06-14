"""Statistical benchmark framework for multi-aspect search.

Generates 100+ queries from product attribute combinations, measures
P50/P95/P99 latency, Recall@1/3/5, MRR, NDCG, and gets real storage
metrics from Qdrant.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from itertools import product as iterproduct
from pathlib import Path

from qdrant_client import QdrantClient

from commerce_engine.embeddings import Embedder
from commerce_engine.fixtures import FIXTURE_PRODUCTS, ensure_fixture_images
from commerce_engine.ingest import product_point
from commerce_engine.logging_config import Timer, get_logger
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import recreate_collection, upsert_products
from commerce_engine.search import search_products

logger = get_logger("benchmark")


@dataclass
class BenchmarkQuery:
    query_text: str
    expected_product_id: str
    category: str = ""


@dataclass
class BenchmarkResult:
    profile: str
    query_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg: float
    storage_size_bytes: int
    points_count: int
    vectors_count: int


def generate_benchmark_queries() -> list[BenchmarkQuery]:
    """Generate 100+ benchmark queries from product attribute combinations."""
    queries: list[BenchmarkQuery] = []

    for p in FIXTURE_PRODUCTS:
        pid = p.id
        cat = p.category

        # Category-only queries
        queries.append(BenchmarkQuery(f"{p.category}", pid, cat))
        queries.append(BenchmarkQuery(f"best {p.category}", pid, cat))
        queries.append(BenchmarkQuery(f"buy {p.category} online", pid, cat))

        # Color + category
        queries.append(BenchmarkQuery(f"{p.color} {p.category}", pid, cat))
        queries.append(BenchmarkQuery(f"{p.color} {p.category} for outdoor", pid, cat))

        # Brand + category
        queries.append(BenchmarkQuery(f"{p.brand} {p.category}", pid, cat))
        queries.append(BenchmarkQuery(f"{p.brand} {p.color} {p.category}", pid, cat))

        # Spec-based queries
        for spec_key, spec_val in p.specs.items():
            queries.append(BenchmarkQuery(f"{spec_key} {spec_val} {p.category}", pid, cat))
            queries.append(BenchmarkQuery(f"{p.category} with {spec_key} {spec_val}", pid, cat))

        # Review-inspired queries
        for review in p.reviews[:3]:
            words = review.lower().split()
            if len(words) >= 3:
                fragment = " ".join(words[:5])
                queries.append(BenchmarkQuery(f"{p.category} {fragment}", pid, cat))

        # Compound queries
        if p.specs:
            first_spec = next(iter(p.specs.values()))
            queries.append(BenchmarkQuery(f"{p.color} {p.category} {first_spec}", pid, cat))

        # Availability-focused
        if p.availability:
            queries.append(BenchmarkQuery(f"available {p.color} {p.category}", pid, cat))

        # Eco queries
        if p.is_sustainable:
            queries.append(BenchmarkQuery(f"eco friendly {p.category}", pid, cat))
            queries.append(BenchmarkQuery(f"sustainable {p.color} {p.category}", pid, cat))
            queries.append(BenchmarkQuery(f"green {p.category} environmentally friendly", pid, cat))

        # Price queries
        queries.append(BenchmarkQuery(f"affordable {p.category} under ${p.price + 50:.0f}", pid, cat))

        # Region queries
        queries.append(BenchmarkQuery(f"{p.category} available in {p.region}", pid, cat))

        # Multi-aspect compound queries
        for spec_key, spec_val in list(p.specs.items())[:2]:
            queries.append(BenchmarkQuery(
                f"{p.color} {p.category} with {spec_val} and good comfort",
                pid, cat,
            ))

    # Cross-product combination queries
    template_patterns = [
        "{color} {category}",
        "{brand} {category} for hiking",
        "comfortable {category} with good support",
        "durable {color} {category}",
        "{category} with excellent grip",
        "lightweight {color} {category}",
    ]
    for p in FIXTURE_PRODUCTS:
        for tpl in template_patterns:
            q = tpl.format(color=p.color, category=p.category, brand=p.brand)
            queries.append(BenchmarkQuery(q, p.id, p.category))

    # Deduplicate by query text (keep first occurrence)
    seen: set[str] = set()
    unique: list[BenchmarkQuery] = []
    for q in queries:
        key = q.query_text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    logger.info(f"generated {len(unique)} benchmark queries")
    return unique


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile using nearest-rank method."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, min(len(sorted_data) - 1, int(math.ceil(pct / 100.0 * len(sorted_data))) - 1))
    return sorted_data[idx]


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def run_benchmark(
    client: QdrantClient,
    collection: str,
    root: Path,
    embedder: Embedder,
    profile: str,
) -> BenchmarkResult:
    benchmark_collection = f"{collection}_{profile}_benchmark"

    logger.info(f"benchmark starting: profile={profile}")

    # Setup collection and ingest
    recreate_collection(
        client, benchmark_collection, profile=profile, disable_text_hnsw=profile != "hnsw"
    )
    products = ensure_fixture_images(root)
    points = [product_point(product, embedder) for product in products]
    upsert_products(client, benchmark_collection, points)

    # Generate queries
    queries = generate_benchmark_queries()
    assert len(queries) >= 100, f"Expected ≥100 queries, got {len(queries)}"

    # Run benchmark queries
    latencies: list[float] = []
    recall_1_hits = 0
    recall_3_hits = 0
    recall_5_hits = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []

    for bq in queries:
        start = time.perf_counter()
        results = search_products(
            client,
            benchmark_collection,
            SearchRequest(
                query=bq.query_text,
                user_id="user_a",
                filters=SearchFilters(availability=None),
                limit=5,
            ),
            embedder,
        )
        latencies.append((time.perf_counter() - start) * 1000)

        result_ids = [r.product_id for r in results]

        # Recall@K
        if bq.expected_product_id in result_ids[:1]:
            recall_1_hits += 1
        if bq.expected_product_id in result_ids[:3]:
            recall_3_hits += 1
        if bq.expected_product_id in result_ids[:5]:
            recall_5_hits += 1

        # MRR
        rr = 0.0
        for rank, rid in enumerate(result_ids, 1):
            if rid == bq.expected_product_id:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # NDCG
        relevances = [1.0 if rid == bq.expected_product_id else 0.0 for rid in result_ids]
        ndcg_scores.append(_ndcg(relevances, 5))

    storage_bytes = 0
    points_count = 0
    vectors_count = 0
    try:
        info = client.get_collection(collection_name=benchmark_collection)
        points_count = getattr(info, "points_count", 0) or 0
        vectors_count = getattr(info, "vectors_count", 0) or getattr(info, "indexed_vectors_count", 0) or 0
        try:
            if info.config and info.config.params and info.config.params.vectors:
                for vec_name, vec_params in info.config.params.vectors.items():
                    dim = vec_params.size
                    per_vector_bytes = dim * 4  # float32
                    if profile == "scalar":
                        per_vector_bytes = dim  # int8
                    elif profile == "binary":
                        per_vector_bytes = (dim + 7) // 8
                    storage_bytes += points_count * per_vector_bytes
        except Exception:
            storage_bytes = 0
    except Exception:
        pass

    n = len(queries)
    result = BenchmarkResult(
        profile=profile,
        query_count=n,
        mean_latency_ms=statistics.mean(latencies),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        recall_at_1=recall_1_hits / n,
        recall_at_3=recall_3_hits / n,
        recall_at_5=recall_5_hits / n,
        mrr=statistics.mean(reciprocal_ranks),
        ndcg=statistics.mean(ndcg_scores),
        storage_size_bytes=storage_bytes,
        points_count=points_count,
        vectors_count=vectors_count,
    )

    logger.info(
        "benchmark completed",
        extra={"extra_data": asdict(result)},
    )

    return result


def result_dict(result: BenchmarkResult) -> dict:
    return asdict(result)


def write_benchmark_report(result: BenchmarkResult, output_dir: Path) -> None:
    """Write benchmark_report.json and benchmark_report.md to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(asdict(result), indent=2))

    # Markdown report
    md_path = output_dir / "benchmark_report.md"
    md_content = f"""# Benchmark Report — Profile: {result.profile}

## Summary

| Metric | Value |
|--------|-------|
| Queries | {result.query_count} |
| Mean Latency | {result.mean_latency_ms:.1f} ms |
| P50 Latency | {result.p50_latency_ms:.1f} ms |
| P95 Latency | {result.p95_latency_ms:.1f} ms |
| P99 Latency | {result.p99_latency_ms:.1f} ms |

## Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall@1 | {result.recall_at_1:.1%} |
| Recall@3 | {result.recall_at_3:.1%} |
| Recall@5 | {result.recall_at_5:.1%} |
| MRR | {result.mrr:.4f} |
| NDCG@5 | {result.ndcg:.4f} |

## Storage

| Metric | Value |
|--------|-------|
| Points | {result.points_count} |
| Vectors | {result.vectors_count} |
| Est. Storage | {result.storage_size_bytes:,} bytes |
"""
    md_path.write_text(md_content)
    logger.info(f"benchmark report written to {output_dir}")
