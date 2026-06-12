from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


def point_id(product_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"qdrant-multivector-commerce:{product_id}"))
