"""Generate a small *synthetic* serving package for offline testing.

This does NOT use DINOv2 or any real data. It fabricates a tiny, internally
consistent artifact package (manifest, calibrated classifier, FAISS index,
gallery images + metadata) so the backend and its test-suite can run without
network access or multi-hundred-MB weights.

Gallery embeddings are produced by running each saved gallery image back
through the *exact* runtime path (PIL load -> eval transform -> MockEncoder),
so the stored embeddings are consistent with what the API computes at query
time. The generated ``manifest.json`` sets ``"model": "mock"`` so the backend's
encoder factory selects :class:`~app.services.encoders.MockEncoder`.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from PIL import Image

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

from sklearn.linear_model import LogisticRegression

# Import the real preprocessing + encoder so the synthetic gallery matches the
# serving path exactly (single source of truth).
from app.services.encoders import MockEncoder
from app.services.preprocessing import (
    build_spec_from_manifest,
    build_transform,
    load_pil_image,
    preprocess_images,
)

EMBEDDING_DIM = 384
IMAGE_SIZE = 224


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (x / norms).astype("float32")


def _make_species_image(species_idx: int, variant: int) -> Image.Image:
    """A solid-ish color image whose base color encodes the species index."""

    # Distinct base color per species so images cluster by class. The small
    # per-variant jitter keeps neighbours within a class similar but not
    # identical.
    r = (species_idx * 53) % 256
    g = (species_idx * 97 + variant * 5) % 256
    b = (species_idx * 151) % 256
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (r, g, b))
    return img


def _embed_image_bytes(data: bytes, transform, encoder) -> np.ndarray:
    """Run one image's bytes through the exact runtime serving path."""

    pil = load_pil_image(data)
    batch = preprocess_images([pil], transform)
    emb = encoder.encode(batch)  # torch (1, D), L2-normalized
    return emb.detach().cpu().numpy().astype("float32")



def make_synthetic_artifacts(
    output_dir,
    species: List[str] = None,
    per_class: int = 6,
    version: str = "synthetic-v1",
) -> Path:
    species = species or ["tiger", "wolf", "panda", "eagle", "dolphin"]
    output_dir = Path(output_dir)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Build the SAME preprocessing + encoder the backend will use at runtime,
    # derived from the manifest we are about to write. This keeps the synthetic
    # gallery consistent with the serving path.
    manifest = {
        "model": "mock",
        "image_size": IMAGE_SIZE,
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "embedding_dim": EMBEDDING_DIM,
        "metric": "cosine",
        "index_type": "IndexFlatIP",
        "classifier": "LogisticRegression",
        "num_classes": len(species),
        "version": version,
    }
    spec = build_spec_from_manifest(manifest)
    transform = build_transform(spec)
    encoder = MockEncoder(embedding_dim=EMBEDDING_DIM)

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    meta_rows = []

    for s_idx, name in enumerate(species):
        for v in range(per_class):
            img = _make_species_image(s_idx, v)
            # Gallery image filename + stable id.
            gid = f"{name}_{v:04d}"
            rel = f"{name}/{gid}.jpg"
            dst = gallery_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Save as JPEG, then re-read the *saved bytes* so the stored
            # embedding matches exactly what a client uploading this file
            # would produce at query time.
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            dst.write_bytes(buf.getvalue())

            embeddings.append(_embed_image_bytes(buf.getvalue(), transform, encoder)[0])
            labels.append(s_idx)
            meta_rows.append(
                {"id": gid, "species": name, "label": s_idx, "filename": rel}
            )

    emb = np.stack(embeddings, axis=0)
    emb = _normalize(emb)
    labels_arr = np.asarray(labels, dtype=np.int64)
    manifest["gallery_size"] = int(emb.shape[0])

    # A classifier that maps embedding -> species. Trained on the gallery.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb, labels_arr)
    joblib.dump(clf, output_dir / "dino_classifier.joblib")

    # FAISS index.
    if faiss is None:
        raise RuntimeError("faiss is required for the synthetic package.")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(emb)
    from app.services.faiss_io import save_index

    save_index(index, output_dir / "dino_gallery.faiss")

    np.save(output_dir / "dino_gallery_embeddings.npy", emb)
    np.save(output_dir / "dino_gallery_labels.npy", labels_arr)

    # metadata.csv
    with (output_dir / "dino_gallery_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "species", "label", "filename"])
        writer.writeheader()
        writer.writerows(meta_rows)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    label_to_id = {name: i for i, name in enumerate(species)}
    id_to_label = {str(i): name for name, i in label_to_id.items()}
    with (output_dir / "class_mapping.json").open("w", encoding="utf-8") as fh:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, fh, indent=2)

    print(f"✅ Synthetic artifacts written to: {output_dir}")
    print(f"   species={species}  gallery={emb.shape[0]}  dim={EMBEDDING_DIM}")
    return output_dir


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    make_synthetic_artifacts(target)
