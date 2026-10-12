"""Encoder abstraction.

The backend is deliberately *model-agnostic*. Anything that can turn a batch
of preprocessed image tensors into L2-normalized embeddings can plug in here.
This lets us swap DINOv2 for ResNet / SigLIP2 later without touching the API.

Two implementations ship today:

* :class:`DinoV2Encoder` — the real frozen DINOv2 ViT-S/14 used by the
  notebook (loaded via ``torch.hub``).
* :class:`MockEncoder` — a deterministic, offline encoder used by the test
  suite so tests run without downloading multi-hundred-MB weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class Encoder(ABC):
    """Turns a batch of preprocessed image tensors into embeddings."""

    name: str = "abstract"
    embedding_dim: int = 0

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings of shape ``(N, embedding_dim)``."""

    def to(self, device: torch.device | str) -> "Encoder":
        return self


class DinoV2Encoder(Encoder):
    """Frozen DINOv2 ViT-S/14 encoder (matches the notebook)."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "auto"):
        self.name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._model = torch.hub.load(
            "facebookresearch/dinov2", model_name
        )
        self._model = self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        # ViT-S/14 -> 384-D embedding.
        self.embedding_dim = 384

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        emb = self._model(images)
        emb = F.normalize(emb, dim=1)
        return emb.cpu()

    def to(self, device: torch.device | str) -> "DinoV2Encoder":
        self.device = str(device)
        self._model = self._model.to(self.device)
        return self


class MockEncoder(Encoder):
    """Deterministic offline encoder for tests and CI.

    Not a real visual model. It derives a stable embedding from each image's
    per-channel mean intensity (its "color signature") projected into the
    embedding space by a fixed random matrix. Test images whose base color
    encodes their class therefore cluster by class, giving the serving stack a
    meaningful (if fake) signal to classify and retrieve on. Identical inputs
    always yield identical embeddings, so a stored gallery image retrieved by
    its own bytes scores cosine ~= 1.0.
    """

    def __init__(self, embedding_dim: int = 384, seed: int = 0):
        self.name = "mock"
        self.embedding_dim = embedding_dim
        self._seed = seed
        # Fixed random projection from a 3-D color signature to embedding_dim.
        rng = np.random.default_rng(20240517 + seed)
        self._projection = rng.standard_normal((3, embedding_dim)).astype("float32")

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        out: List[np.ndarray] = []
        for row in images.detach().cpu().numpy():
            arr = np.asarray(row, dtype="float32")
            if arr.ndim != 3:
                # Fall back to a flat view if the shape is unexpected.
                arr = arr.reshape(3, -1)
            # Per-channel spatial mean -> 3-D color signature.
            sig = arr.reshape(arr.shape[0], -1).mean(axis=1)  # (C,)
            emb = sig @ self._projection[: sig.shape[0]]
            out.append(emb.astype("float32"))

        emb = torch.from_numpy(np.stack(out, axis=0))
        return F.normalize(emb, dim=1)


def build_encoder(kind: str, device: str = "auto", embedding_dim: int = 384) -> Encoder:
    """Factory selecting the encoder implementation by name."""

    kind = (kind or "").lower()
    if kind in {"dinov2", "dinov2_vits14"}:
        return DinoV2Encoder(device=device)
    if kind == "mock":
        return MockEncoder(embedding_dim=embedding_dim)
    raise ValueError(
        f"Unknown encoder '{kind}'. Expected 'dinov2' or 'mock'."
    )
