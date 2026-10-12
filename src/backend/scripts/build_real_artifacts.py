"""Build a REAL serving package using the actual DINOv2 encoder.

Unlike make_synthetic_artifacts.py (which fakes everything offline), this
script:
  1. downloads real animal photos (Pexels CDN + pollinations for rarer
     species),
  2. encodes them with the real DINOv2 ViT-S/14 (torch.hub),
  3. trains a classifier head on those embeddings (LogisticRegression =>
     real calibrated probabilities),
  4. builds a FAISS cosine index,

producing the exact artifact layout the backend expects, with manifest
model="dinov2_vits14". Then run the backend with ENCODER=dinov2.

This is a small hand-built gallery (a handful of classes) to prove the real
model works end-to-end. The full 90-class gallery comes from the notebook's
Serving Export cell.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UA = {"User-Agent": "Mozilla/5.0 (AnimalVisualSearch educational demo)"}

# Real animal photos per species (stable Pexels CDN URLs, verified). Each
# species has a few images so the classifier has signal and retrieval has
# neighbours. num_classes in the manifest is derived from species that
# actually yielded at least one image (see build_real_artifacts).
GALLERY: Dict[str, List[str]] = {
    "cat": [
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/416160/pexels-photo-416160.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1741205/pexels-photo-1741205.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    "dog": [
        "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1108101/pexels-photo-1108101.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4681107/pexels-photo-4681107.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    "horse": [
        "https://images.pexels.com/photos/1996333/pexels-photo-1996333.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/635499/pexels-photo-635499.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1058277/pexels-photo-1058277.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    "bird": [
        "https://images.pexels.com/photos/326900/pexels-photo-326900.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1661179/pexels-photo-1661179.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/847393/pexels-photo-847393.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    "fish": [
        "https://images.pexels.com/photos/128756/pexels-photo-128756.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/1287089/pexels-photo-1287089.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
}

EMBEDDING_DIM = 384
IMAGE_SIZE = 224


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return (x / n).astype("float32")


def _build_faiss(emb: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def build_real_artifacts(output_dir, device: str = "cpu") -> Path:
    import csv
    import json

    import joblib
    from sklearn.linear_model import LogisticRegression

    from app.services.encoders import DinoV2Encoder
    from app.services.faiss_io import save_index
    from app.services.preprocessing import (
        build_spec_from_manifest,
        build_transform,
        load_pil_image,
        preprocess_images,
    )

    output_dir = Path(output_dir)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": "dinov2_vits14",
        "image_size": IMAGE_SIZE,
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "embedding_dim": EMBEDDING_DIM,
        "metric": "cosine",
        "index_type": "IndexFlatIP",
        "version": "real-v1",
    }
    spec = build_spec_from_manifest(manifest)
    transform = build_transform(spec)

    print("Loading real DINOv2 ViT-S/14...")
    encoder = DinoV2Encoder(device=device)

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    meta_rows = []
    # Map each source species to a contiguous class id, but ONLY for species
    # that actually yielded an image (so manifest num_classes matches reality).
    species_names = sorted(GALLERY.keys())
    species_to_class: Dict[str, int] = {}
    counters: Dict[str, int] = {}

    for name in species_names:
        for url in GALLERY[name]:
            try:
                data = _fetch(url)
                pil = load_pil_image(data)
            except Exception as exc:
                print(f"  [warn] skip {name}: {exc}")
                continue

            # Assign a class id the first time this species succeeds.
            if name not in species_to_class:
                species_to_class[name] = len(species_to_class)
            cls = species_to_class[name]

            gid = f"{name}_{counters.get(name, 0):04d}"
            counters[name] = counters.get(name, 0) + 1
            rel = f"{name}/{gid}.jpg"
            dst = gallery_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)

            batch = preprocess_images([pil], transform)
            emb = encoder.encode(batch).detach().cpu().numpy().astype("float32")
            embeddings.append(emb[0])
            labels.append(cls)
            meta_rows.append(
                {"id": gid, "species": name, "label": cls, "filename": rel}
            )
            print(f"  encoded {rel}")

    if not embeddings:
        raise RuntimeError("No images downloaded - check network / URLs.")

    emb = _normalize(np.stack(embeddings, axis=0))
    labels_arr = np.asarray(labels, dtype=np.int64)
    # Only species that actually contributed images.
    present = sorted(species_to_class, key=lambda n: species_to_class[n])
    manifest["num_classes"] = len(present)
    manifest["gallery_size"] = int(emb.shape[0])
    print(f"classes present: {present}")

    # Classifier head on DINOv2 embeddings (real calibrated probabilities).
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(emb, labels_arr)

    save_index(_build_faiss(emb), output_dir / "dino_gallery.faiss")

    joblib.dump(clf, output_dir / "dino_classifier.joblib")
    np.save(output_dir / "dino_gallery_embeddings.npy", emb)
    np.save(output_dir / "dino_gallery_labels.npy", labels_arr)

    with (output_dir / "dino_gallery_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "species", "label", "filename"])
        w.writeheader()
        w.writerows(meta_rows)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    label_to_id = {n: species_to_class[n] for n in present}
    id_to_label = {str(species_to_class[n]): n for n in present}
    with (output_dir / "class_mapping.json").open("w", encoding="utf-8") as fh:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, fh, indent=2)

    print(f"DONE real artifacts -> {output_dir} (gallery={emb.shape[0]}, classes={len(present)})")
    return output_dir


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts_real"
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    build_real_artifacts(target, device=dev)

