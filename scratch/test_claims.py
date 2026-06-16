import torch
from transformers import AutoModel, AutoProcessor

print("Loading SigLIP model...")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

inputs = processor(text=["hello"], images=torch.zeros((3, 224, 224)), return_tensors="pt")
print("Calling get_text_features...")
text_features = model.get_text_features(input_ids=inputs["input_ids"])

print(f"type(text_features): {type(text_features)}")
if isinstance(text_features, torch.Tensor):
    print(f"text_features shape: {text_features.shape}")
else:
    print(f"text_features: {text_features}")

print("Calling get_image_features...")
image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
print(f"type(image_features): {type(image_features)}")
if isinstance(image_features, torch.Tensor):
    print(f"image_features shape: {image_features.shape}")
else:
    print(f"image_features: {image_features}")

print("\nChecking Qdrant...")
from qdrant_client import QdrantClient, models
client = QdrantClient(":memory:")

try:
    client.create_collection(
        collection_name="test_hnsw",
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE, hnsw_config=models.HnswConfigDiff(m=0, ef_construct=None))
    )
    print("Qdrant m=0 with ef_construct=None created successfully IN MEMORY.")
except Exception as e:
    print(f"Qdrant HNSW error: {e}")

try:
    client.create_collection(
        collection_name="test_bq",
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE, multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)),
        quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
    )
    print("Qdrant MAX_SIM with BQ created successfully IN MEMORY.")
except Exception as e:
    print(f"Qdrant BQ error: {e}")
