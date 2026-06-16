import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient
from commerce_engine.embeddings import DeterministicEmbedder
from commerce_engine.fixtures import ensure_fixture_images
from commerce_engine.ingest import ingest_products
from commerce_engine.models import SearchFilters, SearchRequest
from commerce_engine.qdrant_store import recreate_collection
from commerce_engine.search import search_products

def main():
    print("Initializing in-memory Qdrant client...")
    client = QdrantClient(location=":memory:")
    collection_name = "test_commerce_products"
    
    print("Recreating collection...")
    recreate_collection(client, collection_name)
    
    print("Ingesting fixture products...")
    embedder = DeterministicEmbedder()
    products = ensure_fixture_images(Path(__file__).resolve().parents[1])
    ingest_products(client, collection_name, products, embedder)
    
    print("Running multi-aspect search...")
    req = SearchRequest(
        query="Waterproof black hiking boots with good arch support",
        user_id="user_a",
        filters=SearchFilters(availability=None),
        limit=5
    )
    results = search_products(client, collection_name, req, embedder)
    
    print(f"\nSearch returned {len(results)} results:")
    for rank, res in enumerate(results, 1):
        print(f"\n#{rank}: {res.title} (ID: {res.product_id})")
        print(f"  Final Score: {res.final_score:.4f}")
        print(f"  Qdrant Score: {res.qdrant_score:.4f}")
        print(f"  Personalization Boost: {res.personalization_boost:.4f}")
        print("  Explanation:")
        for line in res.explanation:
            print(f"    - {line}")

if __name__ == "__main__":
    main()
