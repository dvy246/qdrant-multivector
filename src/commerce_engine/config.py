from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "commerce_products"
    embedding_backend: Literal["production", "deterministic"] = "production"
    text_late_model: str = "answerdotai/answerai-colbert-small-v1"
    review_model: str = "BAAI/bge-small-en-v1.5"
    vision_model: str = "google/vit-base-patch16-224-in21k"
    device: str = "cpu"
    prefetch_limit: int = Field(default=50, ge=5)
    final_limit: int = Field(default=10, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
