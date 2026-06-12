from __future__ import annotations

from qdrant_client import models

from commerce_engine.models import SearchFilters


def _any_match(key: str, values: list[str] | None) -> list[models.FieldCondition]:
    if not values:
        return []
    return [models.FieldCondition(key=key, match=models.MatchAny(any=values))]


def build_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None

    must: list[models.Condition] = []
    if filters.availability is not None:
        must.append(
            models.FieldCondition(
                key="availability", match=models.MatchValue(value=filters.availability)
            )
        )

    if filters.min_price is not None or filters.max_price is not None:
        must.append(
            models.FieldCondition(
                key="price",
                range=models.Range(gte=filters.min_price, lte=filters.max_price),
            )
        )

    for field, values in [
        ("brand", filters.brands),
        ("category", filters.categories),
        ("region", filters.regions),
        ("color", filters.colors),
        ("size", filters.sizes),
    ]:
        must.extend(_any_match(field, values))

    return models.Filter(must=must) if must else None
