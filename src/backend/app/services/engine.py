"""The model-agnostic search engine.

``AnimalSearchEngine`` is the stable interface the API depends on. Concrete
implementations bundle an :class:`~app.services.encoders.Encoder` with a
loaded :class:`~app.services.artifacts.ArtifactBundle` (classifier + FAISS +
gallery metadata). Swapping DINOv2 for another encoder only requires a new
implementation — the routes and response schema stay untouched.

Design notes
------------
* A single L2-normalized embedding feeds **both** classification and
  retrieval, exactly like the notebook: one DINOv2 representation serving a
  classifier head and an ``IndexFlatIP`` (cosine) index. ``search_and_predict``
  encodes the image once and reuses that embedding for both tasks.
* Confidence comes from the classifier's ``predict_proba``. Production
  artifacts must carry a calibrated classifier; the engine fails fast at
  construction if it does not (override with ``REQUIRE_CALIBRATED=false`` for
  local/synthetic artifacts, which then softmax ``decision_function``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .artifacts import ArtifactBundle
from .encoders import Encoder
from .preprocessing import normalize_embeddings


@dataclass
class Prediction:
    species: str
    confidence: float
    calibrated: bool


@dataclass
class SearchResult:
    id: str
    species: str
    similarity: float
    image_url: str


@dataclass
class SearchOutcome:
    prediction: Prediction
    results: List[SearchResult]


class AnimalSearchEngine(ABC):
    """Stable contract consumed by the HTTP layer."""

    encoder_name: str = "abstract"
    model_version: str = "unknown"

    @abstractmethod
    def predict(self, image_bytes: bytes) -> Prediction:
        """Classify a single uploaded image."""

    @abstractmethod
    def search(self, image_bytes: bytes, k: int) -> List[SearchResult]:
        """Return the top-``k`` gallery neighbours for an image."""

    def search_and_predict(self, image_bytes: bytes, k: int) -> SearchOutcome:
        """Convenience: run classification and retrieval on one embedding."""

        return SearchOutcome(self.predict(image_bytes), self.search(image_bytes, k))


class DinoSearchEngine(AnimalSearchEngine):
    """DINOv2 embedding + classifier head + FAISS cosine retrieval.

    The heavy encoder runs exactly **once** per request: a single L2-normalized
    embedding feeds both the classifier head and the FAISS cosine index.
    """

    def __init__(
        self,
        encoder: Encoder,
        bundle: ArtifactBundle,
        transform,
        gallery_url_prefix: str = "/gallery",
        require_calibrated: bool = True,
    ):
        self._encoder = encoder
        self._bundle = bundle
        self._transform = transform
        self._gallery_prefix = gallery_url_prefix.rstrip("/")

        self.encoder_name = bundle.manifest.get("model", encoder.name)
        self.model_version = bundle.manifest.get("version", "unknown")

        # Classifier probability support detection. Production artifacts must
        # carry a calibrated classifier (with predict_proba) so the exposed
        # confidence is a real probability. We fail fast at construction time
        # unless the caller explicitly opts out (dev / synthetic artifacts).
        self._has_predict_proba = hasattr(bundle.classifier, "predict_proba")
        if require_calibrated and not self._has_predict_proba:
            raise ValueError(
                "Loaded classifier does not expose predict_proba(); the "
                "serving package must contain a calibrated classifier "
                "(see scripts/serving_export.py). Set "
                "REQUIRE_CALIBRATED=false to allow an uncalibrated head."
            )
        self._classes = np.asarray(bundle.classifier.classes_)

        # label id -> species string, from class_mapping.
        self._id_to_label = bundle.class_mapping.get("id_to_label", {})

    # -- embedding -----------------------------------------------------------
    def _embed(self, image_bytes: bytes) -> np.ndarray:
        """Decode + preprocess + encode one image into a (1, D) unit vector."""

        from .preprocessing import load_pil_image, preprocess_images

        pil = load_pil_image(image_bytes)
        batch = preprocess_images([pil], self._transform)
        emb = self._encoder.encode(batch)  # torch (1, D), L2-normalized
        emb_np = emb.detach().cpu().numpy().astype("float32")
        return normalize_embeddings(emb_np)

    # -- classification (operates on a precomputed embedding) ----------------
    def _classify(self, emb: np.ndarray) -> Prediction:
        if self._has_predict_proba:
            proba = self._bundle.classifier.predict_proba(emb)[0]
        else:
            # Only reachable when require_calibrated=False. Softmax over the
            # decision scores — not a true probability.
            scores = self._bundle.classifier.decision_function(emb)
            scores = np.atleast_2d(scores)
            scores = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            proba = (exp / exp.sum(axis=1, keepdims=True))[0]

        top = int(np.argmax(proba))
        confidence = max(0.0, min(1.0, float(proba[top])))
        label_id = int(self._classes[top])
        species = self._id_to_label.get(str(label_id), str(label_id))

        return Prediction(
            species=species, confidence=confidence, calibrated=self._has_predict_proba
        )

    def predict(self, image_bytes: bytes) -> Prediction:
        return self._classify(self._embed(image_bytes))

    # -- retrieval (operates on a precomputed embedding) ---------------------
    def _retrieve(self, emb: np.ndarray, k: int) -> List[SearchResult]:
        k = max(1, min(int(k), self._bundle.gallery_size))

        # Inner product on L2-normalized vectors == cosine similarity.
        scores, indices = self._bundle.index.search(emb, k)
        scores = scores[0]
        indices = indices[0]

        results: List[SearchResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            meta = self._bundle.metadata[int(idx)]
            results.append(
                SearchResult(
                    id=str(meta["id"]),
                    species=str(meta["species"]),
                    similarity=float(score),
                    image_url=f"{self._gallery_prefix}/{meta['filename']}",
                )
            )
        return results

    def search(self, image_bytes: bytes, k: int) -> List[SearchResult]:
        return self._retrieve(self._embed(image_bytes), k)

    def search_and_predict(self, image_bytes: bytes, k: int) -> SearchOutcome:
        """Classify + retrieve from a *single* encoder pass."""

        emb = self._embed(image_bytes)  # encode ONCE
        return SearchOutcome(self._classify(emb), self._retrieve(emb, k))

    # -- info ----------------------------------------------------------------
    def info(self) -> Dict:
        m = self._bundle.manifest
        return {
            "encoder": self.encoder_name,
            "version": self.model_version,
            "embedding_dim": self._bundle.embedding_dim,
            "num_classes": self._bundle.num_classes,
            "gallery_size": self._bundle.gallery_size,
            "metric": m.get("metric", "cosine"),
            "index_type": m.get("index_type", "IndexFlatIP"),
            "image_size": int(m.get("image_size", 224)),
            "calibrated": self._has_predict_proba,
        }


def build_engine(settings) -> DinoSearchEngine:
    """Assemble a :class:`DinoSearchEngine` from application settings."""

    from .artifacts import load_artifacts
    from .encoders import build_encoder
    from .preprocessing import build_spec_from_manifest, build_transform

    bundle = load_artifacts(settings.artifacts_dir)
    spec = build_spec_from_manifest(bundle.manifest)
    transform = build_transform(spec)

    device = settings.resolve_device()
    encoder = build_encoder(
        settings.encoder, device=device, embedding_dim=bundle.embedding_dim
    )
    # If the encoder dim disagrees with the stored artifacts, that is a real
    # misconfiguration — surface it immediately.
    if encoder.embedding_dim != bundle.embedding_dim:
        raise ValueError(
            f"Encoder dim {encoder.embedding_dim} != artifact dim "
            f"{bundle.embedding_dim}. Check ENCODER / artifacts."
        )

    return DinoSearchEngine(
        encoder=encoder,
        bundle=bundle,
        transform=transform,
        gallery_url_prefix=settings.gallery_url_prefix,
        require_calibrated=getattr(settings, "require_calibrated", True),
    )

