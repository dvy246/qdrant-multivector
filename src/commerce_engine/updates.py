from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from commerce_engine.embeddings import Embedder
from commerce_engine.ids import point_id
from commerce_engine.qdrant_store import REVIEW_VECTOR, VISUAL_VECTOR, update_named_vectors
from commerce_engine.reviews import extract_semantic_findings


def _get_point(client: QdrantClient, collection: str, product_id: str):
    points = client.retrieve(
        collection_name=collection,
        ids=[point_id(product_id)],
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        raise ValueError(f"product {product_id} not found")
    return points[0]


def append_review(
    client: QdrantClient,
    collection: str,
    product_id: str,
    review: str,
    embedder: Embedder,
) -> dict[str, Any]:
    point = _get_point(client, collection, product_id)
    payload = dict(point.payload or {})
    reviews = [*payload.get("reviews", []), review]
    findings = extract_semantic_findings([review])
    new_vectors = embedder.review_findings(findings)
    existing = (point.vector or {}).get(REVIEW_VECTOR, [])
    updated_review_vectors = [*existing, *new_vectors]

    update_named_vectors(client, collection, product_id, {REVIEW_VECTOR: updated_review_vectors})
    client.set_payload(
        collection_name=collection,
        payload={"reviews": reviews},
        points=[point_id(product_id)],
        wait=True,
    )
    return {"product_id": product_id, "findings": findings, "review_count": len(reviews)}


def append_image(
    client: QdrantClient,
    collection: str,
    product_id: str,
    image_path: Path,
    embedder: Embedder,
) -> dict[str, Any]:
    point = _get_point(client, collection, product_id)
    new_vectors = embedder.image_patches(image_path)
    existing = (point.vector or {}).get(VISUAL_VECTOR, [])
    updated_visual_vectors = [*existing, *new_vectors]

    update_named_vectors(client, collection, product_id, {VISUAL_VECTOR: updated_visual_vectors})
    return {
        "product_id": product_id,
        "added_patches": len(new_vectors),
        "total_patches": len(updated_visual_vectors),
    }
