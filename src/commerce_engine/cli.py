from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from commerce_engine.benchmark import result_dict, run_benchmark
from commerce_engine.config import get_settings
from commerce_engine.embeddings import create_embedder
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ingest import ingest_products
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import make_client, recreate_collection
from commerce_engine.search import search_products
from commerce_engine.updates import append_image, append_review

app = typer.Typer(no_args_is_help=True)


def _deps():
    settings = get_settings()
    client = make_client(settings.qdrant_url, settings.qdrant_api_key)
    embedder = create_embedder(
        settings.embedding_backend,
        settings.text_late_model,
        settings.review_model,
        settings.device,
        settings.vision_alignment_model,
    )
    return settings, client, embedder


@app.command("init-qdrant")
def init_qdrant(profile: str = "baseline") -> None:
    settings, client, _ = _deps()
    recreate_collection(client, settings.qdrant_collection, profile=profile)
    typer.echo(f"initialized {settings.qdrant_collection} with profile={profile}")


@app.command("ingest")
def ingest(fixtures: bool = typer.Option(False, "--fixtures")) -> None:
    settings, client, embedder = _deps()
    if not fixtures:
        raise typer.BadParameter("only --fixtures ingestion is included in this portfolio build")
    products = ensure_fixture_images(Path.cwd())
    ingest_products(client, settings.qdrant_collection, products, embedder)
    typer.echo(f"ingested {len(products)} products")


@app.command("search")
def search(
    query: str,
    user: str = "user_a",
    limit: int = 10,
    available: bool | None = True,
) -> None:
    settings, client, embedder = _deps()
    results = search_products(
        client,
        settings.qdrant_collection,
        SearchRequest(
            query=query,
            user_id=user,
            filters=SearchFilters(availability=available),
            limit=limit,
        ),
        embedder,
    )
    print_json(json.dumps([result.model_dump() for result in results], default=str))


@app.command("update-review")
def update_review(product_id: str, review: str) -> None:
    settings, client, embedder = _deps()
    print_json(
        json.dumps(
            append_review(client, settings.qdrant_collection, product_id, review, embedder)
        )
    )


@app.command("update-image")
def update_image(product_id: str, image_path: Path) -> None:
    settings, client, embedder = _deps()
    print_json(
        json.dumps(
            append_image(client, settings.qdrant_collection, product_id, image_path, embedder)
        )
    )


@app.command("benchmark")
def benchmark(profile: str = "baseline") -> None:
    settings, client, embedder = _deps()
    from commerce_engine.benchmark import write_benchmark_report
    result = run_benchmark(client, settings.qdrant_collection, Path.cwd(), embedder, profile)
    write_benchmark_report(result, Path.cwd())
    print_json(json.dumps(result_dict(result)))
