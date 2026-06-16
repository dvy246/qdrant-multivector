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
    """Small deterministic embedder for unit tests and offline smoke runs.

    Produces the same shapes as ProductionEmbedder:
    - image_patches() returns 196 patch-level vectors (matching SigLIP patch token grid)
    - visual_query() returns 1 vector per query (matching SigLIP text tower)
    """

    def text_late(self, texts: list[str]) -> list[list[list[float]]]:
        matrices: list[list[list[float]]] = []
        for text in texts:
            tokens = text.lower().split()[:64] or [text]
            matrices.append([_unit_vector(f"text:{token}", TEXT_DIM) for token in tokens])
        return matrices

    def review_findings(self, findings: list[str]) -> list[list[float]]:
        return [_unit_vector(f"review:{finding.lower()}", REVIEW_DIM) for finding in findings]

    def image_patches(self, image_path: Path) -> list[list[float]]:
        """Return 1 vector matching production SigLIP pooled image embedding."""
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.asarray(image, dtype=np.float32)
        mean = arr.mean(axis=(0, 1))
        base_seed = f"siglip-patch:{image_path.name}:{mean[0]:.1f}:{mean[1]:.1f}:{mean[2]:.1f}"
        return [_unit_vector(f"{base_seed}:0", VISION_DIM)]

    def visual_query(self, query: str) -> list[list[float]]:
        """Return 1 vector per query, matching production SigLIP text tower."""
        return [_unit_vector(f"visual-query:{query.lower()}", VISION_DIM)]


class ProductionEmbedder:
    def __init__(
        self,
        text_model: str,
        review_model: str,
        device: str = "cpu",
        vision_alignment_model: str = "google/siglip-base-patch16-224",
    ) -> None:
        self.text_model_name = text_model
        self.review_model_name = review_model
        self.vision_alignment_model_name = vision_alignment_model
        self.device = device
        self._late_model = None
        self._review_model = None
        self._siglip_model = None
        self._siglip_processor = None

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

    def _load_siglip(self) -> None:
        if self._siglip_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self._siglip_processor = AutoProcessor.from_pretrained(
            self.vision_alignment_model_name
        )
        self._siglip_model = AutoModel.from_pretrained(
            self.vision_alignment_model_name
        )
        self._siglip_model.eval()
        self._siglip_model.to(torch.device(self.device))

    def text_late(self, texts: list[str]) -> list[list[list[float]]]:
        return [embedding.astype(float).tolist() for embedding in self.late_model.embed(texts)]

    def review_findings(self, findings: list[str]) -> list[list[float]]:
        if not findings:
            return [_unit_vector("empty-review", REVIEW_DIM)]
        return [embedding.astype(float).tolist() for embedding in self.review_model.embed(findings)]

    def image_patches(self, image_path: Path) -> list[list[float]]:
        """Generate SigLIP aligned pooled image embedding.

        Returns a matrix of shape [1, 768] containing the normalized pooled
        image representation from SigLIP's multihead attention pooling head.
        """
        self._load_siglip()
        import torch

        image = Image.open(image_path).convert("RGB")
        siglip_inputs = self._siglip_processor(images=image, return_tensors="pt")
        siglip_inputs = {k: v.to(self.device) for k, v in siglip_inputs.items()}
        with torch.no_grad():
            vision_outputs = self._siglip_model.vision_model(**siglip_inputs)
            # Shape: [1, 768] - pooled attention features
            pooled = vision_outputs.pooler_output
            # Normalize along the embedding dimension
            normalized = torch.nn.functional.normalize(pooled, dim=-1)
        return normalized.cpu().numpy().astype(float).tolist()

    def visual_query(self, query: str) -> list[list[float]]:
        """Encode visual query text using SigLIP text tower with projection head.

        Returns a 1-row matrix of shape [1, 768] containing the normalized
        projected text representation from SigLIP's linear projection head.
        """
        self._load_siglip()
        import torch

        text_inputs = self._siglip_processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_outputs = self._siglip_model.text_model(**text_inputs)
            # Shape: [1, 768] - EOS token projected features
            pooled = text_outputs.pooler_output
            normalized = torch.nn.functional.normalize(pooled, dim=-1)
        return normalized.cpu().numpy().astype(float).tolist()


def create_embedder(
    backend: str,
    text_model: str,
    review_model: str,
    device: str = "cpu",
    vision_alignment_model: str = "google/siglip-base-patch16-224",
) -> Embedder:
    if backend == "deterministic":
        return DeterministicEmbedder()
    return ProductionEmbedder(
        text_model, review_model, device, vision_alignment_model
    )
