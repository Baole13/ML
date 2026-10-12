"""Serving Export — package the DINOv2 + classifier + FAISS + gallery.

This mirrors the notebook's in-memory objects. Paste the ``main(...)`` call as
the final cell of the Colab notebook (after the DINOv2 classifier and FAISS
index are built), pointing it at the notebook's variables. It writes the exact
artifact layout the backend expects:

    artifacts/
    ├── manifest.json
    ├── class_mapping.json
    ├── dino_classifier.joblib
    ├── dino_gallery_embeddings.npy
    ├── dino_gallery_labels.npy
    ├── dino_gallery_metadata.csv
    ├── dino_gallery.faiss
    └── gallery/<species>/<file>

It also emits a *calibrated* classifier: the notebook's raw ``LinearSVC`` is
wrapped in ``CalibratedClassifierCV`` so the API can report a real confidence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

from sklearn.calibration import CalibratedClassifierCV


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (x / norms).astype("float32")


def _save_faiss_index(index, path: Path) -> None:
    """Unicode-safe FAISS write.

    faiss.write_index(path) mangles non-ASCII paths on Windows (it uses a
    C-level char* interpreted with the ANSI code page). We serialize to a byte
    buffer and write through Python I/O so paths like "Học máy" work.
    """

    data = faiss.serialize_index(index)
    Path(path).write_bytes(np.ascontiguousarray(data, dtype="uint8").tobytes())


def build_gallery_metadata(
    gallery_paths: List[str], gallery_labels: np.ndarray, label_names: List[str]
) -> pd.DataFrame:
    """Assign stable gallery IDs and copy images into the package.

    Stable ID format: ``<species>_<4-digit per-species counter>`` and a copied
    filename ``<species>/<id>.jpg`` so retrieval results map to a portable URL.
    """

    rows = []
    counters: Dict[str, int] = {}

    for path, label in zip(gallery_paths, gallery_labels):
        species = label_names[int(label)]
        counters[species] = counters.get(species, 0) + 1
        gid = f"{species}_{counters[species]:04d}"
        rel_filename = f"{species}/{gid}.jpg"
        rows.append(
            {
                "id": gid,
                "species": species,
                "label": int(label),
                "filename": rel_filename,
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def export_serving_package(
    output_dir,
    gallery_dino: np.ndarray,
    gallery_labels_dino: np.ndarray,
    gallery_paths_dino: List[str],
    dino_clf,
    label_names: List[str],
    model_name: str = "dinov2_vits14",
    version: str = "clean-v1",
    image_size: int = 224,
    calibration_cv: int = 5,
) -> Path:
    """Write the full serving artifact package to ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # 1) Normalize embeddings (cosine via inner product), matching notebook.
    gallery_dino = _normalize(np.asarray(gallery_dino, dtype="float32"))
    n, d = gallery_dino.shape

    # 2) Calibrate the classifier so the API can report real confidence.
    #    CalibratedClassifierCV refits internal clones of the base estimator.
    #    We pass the estimator positionally so this works across sklearn
    #    versions (the kwarg was renamed base_estimator -> estimator in 1.2+).
    classifier = dino_clf
    try:
        calibrated = CalibratedClassifierCV(
            dino_clf, method="sigmoid", cv=calibration_cv
        )
        calibrated.fit(gallery_dino, gallery_labels_dino)
        # Only accept the calibrated head if it actually exposes probabilities.
        if hasattr(calibrated, "predict_proba"):
            classifier = calibrated
    except Exception:
        # Fall back to the raw classifier; the engine will softmax
        # decision_function and flag `calibrated=False`.
        classifier = dino_clf

    joblib.dump(classifier, output_dir / "dino_classifier.joblib")

    # 3) FAISS index (exact cosine = inner product on normalized vectors).
    if faiss is None:
        raise RuntimeError("faiss is required to build the gallery index.")
    index = faiss.IndexFlatIP(d)
    index.add(gallery_dino)
    # Unicode-safe write: faiss.write_index(path) mangles non-ASCII paths on
    # Windows (e.g. "Học máy"). Serialize to bytes and write via Python I/O.
    _save_faiss_index(index, output_dir / "dino_gallery.faiss")

    # 4) Embeddings + labels.
    np.save(output_dir / "dino_gallery_embeddings.npy", gallery_dino)
    np.save(output_dir / "dino_gallery_labels.npy", np.asarray(gallery_labels_dino))

    # 5) Metadata + copy images into the package (portable gallery).
    meta_df = build_gallery_metadata(
        gallery_paths_dino, np.asarray(gallery_labels_dino), label_names
    )
    for _, row in meta_df.iterrows():
        dst = gallery_dir / row["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(row["source_path"], dst)
        except Exception as exc:
            print(f"[warn] could not copy {row['source_path']}: {exc}")
    meta_df.drop(columns=["source_path"]).to_csv(
        output_dir / "dino_gallery_metadata.csv", index=False
    )

    # 6) manifest.json — the preprocessing contract the backend must honor.
    manifest = {
        "model": model_name,
        "image_size": image_size,
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "embedding_dim": d,
        "metric": "cosine",
        "index_type": "IndexFlatIP",
        "classifier": type(classifier).__name__,
        "num_classes": len(label_names),
        "gallery_size": int(n),
        "version": version,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # 7) class_mapping.json — id <-> species label.
    label_to_id = {name: i for i, name in enumerate(label_names)}
    id_to_label = {str(i): name for name, i in label_to_id.items()}
    with (output_dir / "class_mapping.json").open("w", encoding="utf-8") as fh:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, fh, indent=2)

    print(f"✅ Serving package exported to: {output_dir}")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# Notebook usage (paste as the final code cell):
#
# from serving_export import export_serving_package
# export_serving_package(
#     output_dir=OUTPUT_DIR / "artifacts",
#     gallery_dino=gallery_dino,
#     gallery_labels_dino=gallery_labels_dino,
#     gallery_paths_dino=gallery_paths_dino,
#     dino_clf=dino_clf,
#     label_names=label_names,
#     model_name="dinov2_vits14",
#     version="clean-v1",
#     image_size=IMG_SIZE_CNN,
# )
# ---------------------------------------------------------------------------

