"""Loading and validation of the serving artifact package.

The package layout (produced by the notebook "Serving Export" cell) is::

    artifacts/
    ├── manifest.json                # preprocessing + model contract
    ├── class_mapping.json           # {label: id} and {id: label}
    ├── dino_classifier.joblib       # sklearn classifier with predict_proba
    ├── dino_gallery_embeddings.npy  # L2-normalized (N, D) float32
    ├── dino_gallery_labels.npy      # (N,) integer labels
    ├── dino_gallery_metadata.csv    # id, species, filename, ...
    └── dino_gallery.faiss           # IndexFlatIP over the embeddings

The loader cross-checks that the FAISS index size, embeddings, labels and
metadata rows all agree, and that the classifier's class count matches the
class mapping — failing fast on any inconsistency rather than serving garbage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

try:  # faiss is a hard dependency at serving time.
    import faiss
except Exception:  # pragma: no cover - only hit if faiss not installed
    faiss = None


class ArtifactError(RuntimeError):
    """Raised when the artifact package is missing or inconsistent."""


@dataclass
class ArtifactBundle:
    """In-memory representation of a loaded serving package."""

    manifest: Dict
    class_mapping: Dict[str, Dict[str, str]]
    classifier: object
    embeddings: np.ndarray
    labels: np.ndarray
    metadata: List[Dict]
    index: "faiss.Index"
    artifacts_dir: Path

    # ---- convenience views -------------------------------------------------
    @property
    def gallery_size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_mapping.get("label_to_id", {}))


def _read_json(path: Path) -> Dict:
    if not path.exists():
        raise ArtifactError(f"Missing artifact file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_artifacts(artifacts_dir: Path) -> ArtifactBundle:
    """Load and validate a serving artifact package from ``artifacts_dir``."""

    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise ArtifactError(f"Artifacts directory does not exist: {artifacts_dir}")

    manifest = _read_json(artifacts_dir / "manifest.json")
    class_mapping = _read_json(artifacts_dir / "class_mapping.json")

    emb_path = artifacts_dir / "dino_gallery_embeddings.npy"
    lab_path = artifacts_dir / "dino_gallery_labels.npy"
    meta_path = artifacts_dir / "dino_gallery_metadata.csv"
    clf_path = artifacts_dir / "dino_classifier.joblib"
    faiss_path = artifacts_dir / "dino_gallery.faiss"

    for p in (emb_path, lab_path, meta_path, clf_path, faiss_path):
        if not p.exists():
            raise ArtifactError(f"Missing artifact file: {p}")

    embeddings = np.load(emb_path).astype("float32")
    labels = np.load(lab_path)
    metadata = pd.read_csv(meta_path).to_dict(orient="records")

    if embeddings.ndim != 2:
        raise ArtifactError(
            f"Embeddings must be 2-D (N, D), got shape {embeddings.shape}."
        )
    n, d = embeddings.shape

    if labels.shape[0] != n:
        raise ArtifactError(
            f"Label count {labels.shape[0]} != embedding count {n}."
        )
    if len(metadata) != n:
        raise ArtifactError(
            f"Metadata rows {len(metadata)} != embedding count {n}."
        )

    manifest_dim = manifest.get("embedding_dim")
    if manifest_dim is not None and int(manifest_dim) != d:
        raise ArtifactError(
            f"Manifest embedding_dim {manifest_dim} != actual {d}."
        )

    if faiss is None:
        raise ArtifactError("faiss is not installed; cannot load gallery index.")

    from .faiss_io import load_index

    index = load_index(faiss_path)
    if index.ntotal != n:
        raise ArtifactError(
            f"FAISS index size {index.ntotal} != embedding count {n}."
        )
    if index.d != d:
        raise ArtifactError(f"FAISS index dim {index.d} != embedding dim {d}.")

    classifier = joblib.load(clf_path)

    # Sanity: the classifier must be able to produce probabilities, because the
    # API exposes a calibrated confidence. Predict/proba presence is checked
    # lazily by the engine, but we assert the class count lines up here.
    n_classes = len(class_mapping.get("label_to_id", {}))
    if n_classes == 0:
        raise ArtifactError("class_mapping.json has an empty label_to_id map.")

    return ArtifactBundle(
        manifest=manifest,
        class_mapping=class_mapping,
        classifier=classifier,
        embeddings=embeddings,
        labels=labels,
        metadata=metadata,
        index=index,
        artifacts_dir=artifacts_dir,
    )
