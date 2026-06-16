import inspect
from transformers import SiglipModel

print("SiglipModel get_image_features source:")
try:
    print(inspect.getsource(SiglipModel.get_image_features))
except Exception as e:
    print(f"Error: {e}")

print("\nSiglipModel get_text_features source:")
try:
    print(inspect.getsource(SiglipModel.get_text_features))
except Exception as e:
    print(f"Error: {e}")

print("\nSiglipModel forward source:")
try:
    print(inspect.getsource(SiglipModel.forward))
except Exception as e:
    print(f"Error: {e}")
