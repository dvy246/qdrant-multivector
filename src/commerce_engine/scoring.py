from __future__ import annotations

import numpy as np

from commerce_engine.models import SearchResult, UserProfile


def maxsim_score(query_matrix: list[list[float]], doc_matrix: list[list[float]]) -> float:
    query = np.asarray(query_matrix, dtype=np.float32)
    doc = np.asarray(doc_matrix, dtype=np.float32)
    if query.ndim != 2 or doc.ndim != 2:
        raise ValueError("MaxSim inputs must be matrices")
    if query.shape[1] != doc.shape[1]:
        raise ValueError("Query and document vectors must have the same dimensionality")
    similarities = query @ doc.T
    return float(similarities.max(axis=1).sum())


def personalization_boost(payload: dict, profile: UserProfile) -> tuple[float, list[str]]:
    boost = 0.0
    reasons: list[str] = []

    if payload.get("brand") in profile.preferred_brands:
        boost += 0.25
        reasons.append(f"preferred brand: {payload['brand']}")

    price = float(payload.get("price", 0.0))
    low, high = profile.price_range
    if low <= price <= high:
        boost += 0.30
        reasons.append(f"price in user range: {low:g}-{high:g}")

    if profile.eco_preference and payload.get("is_sustainable"):
        boost += 0.30
        reasons.append("eco preference matched")

    if payload.get("category") in profile.favorite_categories:
        boost += 0.20
        reasons.append(f"favorite category: {payload['category']}")

    return boost, reasons


def rerank(results: list[SearchResult], profile: UserProfile) -> list[SearchResult]:
    reranked: list[SearchResult] = []
    for result in results:
        boost, reasons = personalization_boost(result.payload, profile)
        final_score = result.qdrant_score + boost
        reranked.append(
            result.model_copy(
                update={
                    "final_score": final_score,
                    "personalization_boost": boost,
                    "explanation": [*result.explanation, *reasons],
                }
            )
        )
    return sorted(reranked, key=lambda item: item.final_score, reverse=True)
