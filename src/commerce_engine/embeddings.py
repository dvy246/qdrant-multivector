from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

TEXT_DIM = 96
REVIEW_DIM = 384
VISION_DIM = 768


class Embedder(Protocol):
    def text_late(self, texts: list[str]) -> list[list[list[float]]]: ...

    def review_findings(self, findings: list[str]) -> list[list[float]]: ...

    def image_patches(self, image_path: Path) -> list[list[float]]: ...

    def visual_query(self, query: str) -> list[list[float]]: ...


def _unit_vector(seed: str, dim: int) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8)
    vector = (values.astype(np.float32) - 127.5) / 127.5
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


class DeterministicEmbedder:
    """Small deterministic embedder for unit tests and offline smoke runs."""

    def text_late(self, texts: list[str]) -> list[list[list[float]]]:
        matrices: list[list[list[float]]] = []
        for text in texts:
            tokens = text.lower().split()[:64] or [text]
            matrices.append([_unit_vector(f"text:{token}", TEXT_DIM) for token in tokens])
        return matrices

    def review_findings(self, findings: list[str]) -> list[list[float]]:
        return [_unit_vector(f"review:{finding.lower()}", REVIEW_DIM) for finding in findings]

    def image_patches(self, image_path: Path) -> list[list[float]]:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.asarray(image, dtype=np.float32)
        patches: list[list[float]] = []
        for y in range(0, 224, 56):
            for x in range(0, 224, 56):
                patch = arr[y : y + 56, x : x + 56]
                mean = patch.mean(axis=(0, 1))
                seed = f"image:{image_path.name}:{mean[0]:.1f}:{mean[1]:.1f}:{mean[2]:.1f}:{x}:{y}"
                patches.append(_unit_vector(seed, VISION_DIM))
        return patches

    def visual_query(self, query: str) -> list[list[float]]:
        tokens = query.lower().split()[:16] or [query]
        return [_unit_vector(f"visual-query:{token}", VISION_DIM) for token in tokens]


class ProductionEmbedder:
    def __init__(
        self,
        text_model: str,
        review_model: str,
        vision_model: str,
        device: str = "cpu",
    ) -> None:
        self.text_model_name = text_model
        self.review_model_name = review_model
        self.vision_model_name = vision_model
        self.device = device
        self._late_model = None
        self._review_model = None
        self._vision_processor = None
        self._vision_model = None

    @property
    def late_model(self):
        if self._late_model is None:
            from fastembed import LateInteractionTextEmbedding

            self._late_model = LateInteractionTextEmbedding(self.text_model_name)
        return self._late_model

    @property
    def review_model(self):
        if self._review_model is None:
            from fastembed import TextEmbedding

            self._review_model = TextEmbedding(self.review_model_name)
        return self._review_model

    def _load_vision(self) -> None:
        if self._vision_model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, ViTModel

        self._vision_processor = AutoImageProcessor.from_pretrained(self.vision_model_name)
        self._vision_model = ViTModel.from_pretrained(self.vision_model_name)
        self._vision_model.eval()
        self._vision_model.to(torch.device(self.device))

    def text_late(self, texts: list[str]) -> list[list[list[float]]]:
        return [embedding.astype(float).tolist() for embedding in self.late_model.embed(texts)]

    def review_findings(self, findings: list[str]) -> list[list[float]]:
        if not findings:
            return [_unit_vector("empty-review", REVIEW_DIM)]
        return [embedding.astype(float).tolist() for embedding in self.review_model.embed(findings)]

    def image_patches(self, image_path: Path) -> list[list[float]]:
        self._load_vision()
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._vision_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self._vision_model(**inputs)
        patch_tokens = output.last_hidden_state[:, 1:, :].squeeze(0)
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=1)
        return patch_tokens.cpu().numpy().astype(float).tolist()

    def visual_query(self, query: str) -> list[list[float]]:
        # For production image-query alignment, replace this with a CLIP/SigLIP text tower.
        # The project still stores real ViT patch matrices for product images.
        return [
            _unit_vector(f"visual-intent:{token}", VISION_DIM)
            for token in query.lower().split()
        ]


def create_embedder(
    backend: str,
    text_model: str,
    review_model: str,
    vision_model: str,
    device: str = "cpu",
) -> Embedder:
    if backend == "deterministic":
        return DeterministicEmbedder()
    return ProductionEmbedder(text_model, review_model, vision_model, device)
