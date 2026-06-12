from __future__ import annotations

from qdrant_client import QdrantClient, models

from commerce_engine.embeddings import Embedder
from commerce_engine.filters import build_filter
from commerce_engine.fixtures import FIXTURE_USERS
from commerce_engine.models import SearchRequest, SearchResult, UserProfile
from commerce_engine.qdrant_store import REVIEW_VECTOR, TEXT_VECTOR, VISUAL_VECTOR
from commerce_engine.query import decompose_query
from commerce_engine.scoring import rerank


def _profile(user_id: str) -> UserProfile:
    return FIXTURE_USERS.get(user_id, FIXTURE_USERS["user_a"])


def search_products(
    client: QdrantClient,
    collection: str,
    request: SearchRequest,
    embedder: Embedder,
) -> list[SearchResult]:
    plan = decompose_query(request.query)
    query_filter = build_filter(request.filters)

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
                explanation=[
                    f"text route: {plan.text_query}",
                    f"visual route: {plan.visual_query}",
                    f"review route: {plan.review_query}",
                ],
                payload=payload,
            )
        )

    return rerank(raw_results, _profile(request.user_id))[: request.limit]
