"""Image loading, validation and preprocessing.

The preprocessing contract (image size + normalization statistics) is *read
from the serving manifest*, never hard-coded here. This guarantees the backend
applies exactly the transform the classifier / FAISS index were trained and
built with — mirroring the notebook's ``eval_tf``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError

# Match the notebook: tolerate truncated JPEGs instead of hard-failing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Canonical ImageNet statistics, used when the manifest requests
# normalization == "imagenet". These are the exact values the notebook used.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}


class InvalidImageError(ValueError):
    """Raised when the uploaded payload cannot be decoded as an image."""


@dataclass(frozen=True)
class PreprocessSpec:
    """The preprocessing contract derived from the manifest."""

    image_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def build_spec_from_manifest(manifest: dict) -> PreprocessSpec:
    """Translate a serving ``manifest.json`` into a :class:`PreprocessSpec`.

    Unknown normalization schemes fall back to ImageNet statistics so a
    missing field never silently produces garbage, but we still require a
    positive integer image size.
    """

    size = int(manifest.get("image_size", 224))
    if size <= 0:
        raise ValueError(f"manifest image_size must be positive, got {size!r}")

    normalization = str(manifest.get("normalization", "imagenet")).lower()
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        # Allow an explicit override in the manifest, otherwise default.
        mean = tuple(manifest.get("mean", IMAGENET_MEAN))  # type: ignore[assignment]
        std = tuple(manifest.get("std", IMAGENET_STD))  # type: ignore[assignment]

    return PreprocessSpec(image_size=size, mean=mean, std=std)


def build_transform(spec: PreprocessSpec) -> "torch.nn.Module":
    """Build the deterministic eval transform matching the notebook.

    Deliberately identical to the notebook ``eval_tf``:
    ``Resize(H, W) -> ToTensor -> Normalize`` (no augmentation).
    """

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((spec.image_size, spec.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(spec.mean), std=list(spec.std)),
        ]
    )


def validate_upload(
    data: bytes, content_type: str | None, max_bytes: int
) -> None:
    """Reject obviously bad uploads before we spend CPU decoding them."""

    if not data:
        raise InvalidImageError("Empty file.")
    if len(data) > max_bytes:
        raise InvalidImageError(
            f"File too large: {len(data)} bytes exceeds limit of {max_bytes} bytes."
        )
    # Content-Type is advisory (clients lie); the real check is PIL decode.
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        # Allow anything, but only proceed if it decodes as an image below.
        pass


def load_pil_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image or raise :class:`InvalidImageError`."""

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    return image.convert("RGB")


def preprocess_images(
    pil_images: List[Image.Image], transform: "torch.nn.Module"
) -> torch.Tensor:
    """Apply ``transform`` to a batch of PIL images and stack into a tensor."""

    tensors = [transform(img) for img in pil_images]
    return torch.stack(tensors, dim=0)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner-product == cosine similarity.

    Matches the notebook which does ``F.normalize(z, dim=1)`` before storing
    gallery embeddings and building the FAISS ``IndexFlatIP``.
    """

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms
