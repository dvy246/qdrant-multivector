import pytest
from qdrant_client import QdrantClient, models

def test_in_memory():
    client = QdrantClient(location=":memory:")
    collection_name = "test_collection"
    
    # Create collection with named multivectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "vector_a": models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        }
    )
    
    # Upsert
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=1,
                vector={"vector_a": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]},
                payload={"name": "test"}
            )
        ]
    )
    
    # Retrieve
    res = client.retrieve(collection_name=collection_name, ids=[1])
    print("Retrieved:", res)

    # Query points
    query_res = client.query_points(
        collection_name=collection_name,
        query=[[1.0, 0.0, 0.0, 0.0]],
        using="vector_a",
        limit=1
    )
    print("Query result:", query_res)

if __name__ == "__main__":
    test_in_memory()
