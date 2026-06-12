from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Aspect(StrEnum):
    VISUAL = "visual"
    TEXT = "text"
    REVIEW = "review"


class Product(BaseModel):
    id: str
    title: str
    brand: str
    category: str
    price: float = Field(gt=0)
    availability: bool
    region: str
    color: str
    sizes: list[str]
    eco_score: float = Field(ge=0, le=1)
    is_sustainable: bool
    specs: dict[str, str]
    reviews: list[str]
    image_path: Path | None = None

    @field_validator("sizes")
    @classmethod
    def require_sizes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("sizes cannot be empty")
        return value

    def payload(self) -> dict[str, Any]:
        return {
            "product_id": self.id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "price": self.price,
            "availability": self.availability,
            "region": self.region,
            "color": self.color,
            "size": self.sizes,
            "eco_score": self.eco_score,
            "is_sustainable": self.is_sustainable,
            "specs": self.specs,
            "reviews": self.reviews,
        }

    def text_document(self) -> str:
        specs = " ".join(f"{key}: {value}" for key, value in sorted(self.specs.items()))
        return f"{self.title}. {specs}"


class UserProfile(BaseModel):
    id: str
    preferred_brands: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10_000.0)
    eco_preference: bool = False
    favorite_categories: list[str] = Field(default_factory=list)


class SearchFilters(BaseModel):
    min_price: float | None = None
    max_price: float | None = None
    availability: bool | None = True
    brands: list[str] | None = None
    categories: list[str] | None = None
    regions: list[str] | None = None
    colors: list[str] | None = None
    sizes: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str = "user_a"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=100)


class QueryPlan(BaseModel):
    original_query: str
    text_terms: list[str] = Field(default_factory=list)
    visual_terms: list[str] = Field(default_factory=list)
    review_terms: list[str] = Field(default_factory=list)

    @property
    def text_query(self) -> str:
        return " ".join(self.text_terms or [self.original_query])

    @property
    def visual_query(self) -> str:
        return " ".join(self.visual_terms or [self.original_query])

    @property
    def review_query(self) -> str:
        return " ".join(self.review_terms or [self.original_query])


class SearchResult(BaseModel):
    product_id: str
    title: str
    brand: str
    category: str
    qdrant_score: float
    final_score: float
    personalization_boost: float
    explanation: list[str]
    payload: dict[str, Any]
