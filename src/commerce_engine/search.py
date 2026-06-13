from __future__ import annotations

from qdrant_client import QdrantClient, models

from commerce_engine.embeddings import Embedder
from commerce_engine.filters import build_filter
from commerce_engine.fixtures import FIXTURE_USERS
from commerce_engine.logging_config import Timer, get_logger
from commerce_engine.models import SearchRequest, SearchResult, UserProfile
from commerce_engine.qdrant_store import REVIEW_VECTOR, TEXT_VECTOR, VISUAL_VECTOR
from commerce_engine.query import decompose_query
from commerce_engine.scoring import rerank

logger = get_logger("search")


def _profile(user_id: str) -> UserProfile:
    return FIXTURE_USERS.get(user_id, FIXTURE_USERS["user_a"])


def _build_explanation(plan, payload: dict) -> list[str]:
    """Generate dynamic per-aspect explanation strings."""
    explanations: list[str] = []

    if plan.text_terms:
        matched_specs = []
        specs = payload.get("specs", {})
        for term in plan.text_terms:
            for spec_key, spec_val in specs.items():
                if term.lower() in str(spec_val).lower() or term.lower() in spec_key.lower():
                    matched_specs.append(f"{spec_key}={spec_val}")
        if matched_specs:
            explanations.append(f"spec match: {', '.join(matched_specs)}")
        explanations.append(f"spec query terms: {', '.join(plan.text_terms)}")

    if plan.visual_terms:
        color = payload.get("color", "")
        category = payload.get("category", "")
        visual_hits = []
        for term in plan.visual_terms:
            if term.lower() in color.lower():
                visual_hits.append(f"color={color}")
            if term.lower() in category.lower():
                visual_hits.append(f"category={category}")
        if visual_hits:
            explanations.append(f"visual match: {', '.join(visual_hits)}")
        explanations.append(f"visual query terms: {', '.join(plan.visual_terms)}")

    if plan.review_terms:
        reviews = payload.get("reviews", [])
        review_hits = []
        for term in plan.review_terms:
            for review in reviews:
                if term.lower() in review.lower():
                    review_hits.append(term)
                    break
        if review_hits:
            explanations.append(f"review evidence: {', '.join(review_hits)}")
        explanations.append(f"review query terms: {', '.join(plan.review_terms)}")

    return explanations


def search_products(
    client: QdrantClient,
    collection: str,
    request: SearchRequest,
    embedder: Embedder,
) -> list[SearchResult]:
    plan = decompose_query(request.query)
    query_filter = build_filter(request.filters)

    logger.info(
        "search started",
        extra={"extra_data": {
            "query": request.query,
            "user": request.user_id,
            "text_terms": plan.text_terms,
            "visual_terms": plan.visual_terms,
            "review_terms": plan.review_terms,
        }},
    )

    with Timer() as embed_timer:
        text_query = embedder.text_late([plan.text_query])[0]
        review_query = embedder.review_findings([plan.review_query])
        visual_query = embedder.visual_query(plan.visual_query)

    candidate_limit = max(request.limit * 5, 20)

    prefetch = [
        models.Prefetch(
            query=text_query,
            using=TEXT_VECTOR,
            limit=candidate_limit,
            filter=query_filter,
        ),
        models.Prefetch(
            query=review_query,
            using=REVIEW_VECTOR,
            limit=candidate_limit,
            filter=query_filter,
        ),
    ]

    with Timer() as qdrant_timer:
        response = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=visual_query,
            using=VISUAL_VECTOR,
            query_filter=query_filter,
            limit=max(request.limit * 3, 10),
            with_payload=True,
        )

    raw_results: list[SearchResult] = []
    for point in response.points:
        payload = dict(point.payload or {})
        raw_results.append(
            SearchResult(
                product_id=str(payload.get("product_id", point.id)),
                title=str(payload.get("title", point.id)),
                brand=str(payload.get("brand", "")),
                category=str(payload.get("category", "")),
                qdrant_score=float(point.score),
                final_score=float(point.score),
                personalization_boost=0.0,
                explanation=_build_explanation(plan, payload),
                payload=payload,
            )
        )

    with Timer() as rerank_timer:
        results = rerank(raw_results, _profile(request.user_id))[: request.limit]

    logger.info(
        "search completed",
        extra={"extra_data": {
            "results_count": len(results),
            "embed_ms": round(embed_timer.elapsed_ms, 1),
            "qdrant_ms": round(qdrant_timer.elapsed_ms, 1),
            "rerank_ms": round(rerank_timer.elapsed_ms, 1),
            "total_ms": round(
                embed_timer.elapsed_ms + qdrant_timer.elapsed_ms + rerank_timer.elapsed_ms, 1
            ),
        }},
    )

    return results
