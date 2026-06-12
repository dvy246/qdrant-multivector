from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models

from commerce_engine.embeddings import Embedder
from commerce_engine.ids import point_id
from commerce_engine.models import Product
from commerce_engine.qdrant_store import REVIEW_VECTOR, TEXT_VECTOR, VISUAL_VECTOR, upsert_products
from commerce_engine.reviews import extract_semantic_findings


def product_point(product: Product, embedder: Embedder) -> models.PointStruct:
    if product.image_path is None:
        raise ValueError(f"product {product.id} is missing image_path")

    text_matrix = embedder.text_late([product.text_document()])[0]
    review_matrix = embedder.review_findings(extract_semantic_findings(product.reviews))
    visual_matrix = embedder.image_patches(Path(product.image_path))

    return models.PointStruct(
        id=point_id(product.id),
        payload=product.payload(),
        vector={
            VISUAL_VECTOR: visual_matrix,
            TEXT_VECTOR: text_matrix,
            REVIEW_VECTOR: review_matrix,
        },
    )


def ingest_products(
    client: QdrantClient,
    collection: str,
    products: list[Product],
    embedder: Embedder,
) -> None:
    upsert_products(client, collection, (product_point(product, embedder) for product in products))
