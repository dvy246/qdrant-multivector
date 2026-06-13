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
    vision_alignment_model: str = "google/siglip-base-patch16-224"
    device: str = "cpu"
    prefetch_limit: int = Field(default=50, ge=5)
    final_limit: int = Field(default=10, ge=1)
    log_level: str = "INFO"


def validate_startup(settings: "Settings", client=None) -> list[str]:
    """Verify runtime dependencies are available. Returns list of errors (empty = OK)."""
    errors: list[str] = []

    # Check Qdrant connectivity
    try:
        if client is None:
            from commerce_engine.qdrant_store import make_client
            client = make_client(settings.qdrant_url, settings.qdrant_api_key)
        client.get_collections()
    except Exception as exc:
        errors.append(f"Qdrant unreachable at {settings.qdrant_url}: {exc}")

    # Check collection existence (warning, not fatal)
    try:
        if not client.collection_exists(collection_name=settings.qdrant_collection):
            errors.append(
                f"Collection '{settings.qdrant_collection}' not found. "
                f"Run: engine init-qdrant"
            )
    except Exception:
        pass  # Already reported above

    return errors


@lru_cache
def get_settings() -> Settings:
    return Settings()
