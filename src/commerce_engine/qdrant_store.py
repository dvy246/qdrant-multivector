from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import QdrantClient, models

from commerce_engine.embeddings import REVIEW_DIM, TEXT_DIM, VISION_DIM
from commerce_engine.ids import point_id

VISUAL_VECTOR = "visual_vectors"
TEXT_VECTOR = "text_vectors"
REVIEW_VECTOR = "review_vectors"


def make_client(url: str, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def vector_params(size: int, *, hnsw_m: int | None = None) -> models.VectorParams:
    hnsw_config = models.HnswConfigDiff(m=hnsw_m) if hnsw_m is not None else None
    return models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        hnsw_config=hnsw_config,
    )


def quantization_config(profile: str):
    if profile == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        )
    if profile == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        )
    return None


def recreate_collection(
    client: QdrantClient,
    collection: str,
    *,
    profile: str = "baseline",
    disable_text_hnsw: bool = True,
) -> None:
    if client.collection_exists(collection_name=collection):
        client.delete_collection(collection_name=collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            VISUAL_VECTOR: vector_params(VISION_DIM),
            TEXT_VECTOR: vector_params(TEXT_DIM, hnsw_m=0 if disable_text_hnsw else None),
            REVIEW_VECTOR: vector_params(REVIEW_DIM),
        },
        quantization_config=quantization_config(profile),
    )
    create_payload_indexes(client, collection)


def create_payload_indexes(client: QdrantClient, collection: str) -> None:
    for field in ["brand", "category", "region", "color", "size", "product_id"]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    client.create_payload_index(
        collection_name=collection,
        field_name="availability",
        field_schema=models.PayloadSchemaType.BOOL,
    )
    for field in ["price", "eco_score"]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.FLOAT,
        )


def upsert_products(
    client: QdrantClient,
    collection: str,
    points: Iterable[models.PointStruct],
) -> None:
    client.upsert(collection_name=collection, points=list(points), wait=True)


def update_named_vectors(
    client: QdrantClient,
    collection: str,
    product_id: str,
    vectors: dict[str, list[list[float]]],
) -> None:
    client.update_vectors(
        collection_name=collection,
        points=[models.PointVectors(id=point_id(product_id), vector=vectors)],
        wait=True,
    )
