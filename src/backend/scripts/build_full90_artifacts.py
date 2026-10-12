"""Build a FULL serving package from the real Animal-90 dataset on disk.

Reads the Animal-90 dataset from the local kagglehub cache, takes a
per-class slice as the serving gallery (default: all images), encodes
every gallery image with the real DINOv2 ViT-S/14, trains a calibrated
classifier head (CalibratedClassifierCV over LinearSVC, like the
notebook), and builds a FAISS cosine index.

Usage (from src/backend):
    python scripts/build_full90_artifacts.py artifacts_full90 --per-class 20
    python scripts/build_full90_artifacts.py artifacts_full90  # all/class

Progress is checkpointed to <out>/_emb_checkpoint.npz so an interrupted
run can be resumed with --resume. Gallery images are COPIED from the
kaggle cache into <out>/gallery/<species>/ so serving is self-contained.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KAGGLE_ANIMAL90 = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "iamsouravbanerjee"
    / "animal-image-dataset-90-different-animals"
    / "versions"
    / "5"
    / "animals"
    / "animals"
)

IMAGE_SIZE = 224
EMBEDDING_DIM = 384
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return (x / n).astype("float32")


def _build_faiss(emb: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def _collect_gallery(src: Path, per_class: int, seed: int) -> List[Tuple[str, Path]]:
    rng = np.random.default_rng(seed)
    items: List[Tuple[str, Path]] = []
    species_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    if not species_dirs:
        raise RuntimeError(f"No class folders found under {src}")
    for cls_dir in species_dirs:
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )
        if not files:
            print(f"  [warn] empty class folder: {cls_dir.name}")
            continue
        if per_class > 0 and len(files) > per_class:
            idx = rng.choice(len(files), size=per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        for f in files:
            items.append((cls_dir.name, f))
    return items


def build_full90_artifacts(
    output="artifacts_full90", per_class=0, seed=42,
    device="cpu", checkpoint_every=200, resume=False,
):
    import csv
    import json
    import shutil

    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    from app.services.encoders import DinoV2Encoder
    from app.services.faiss_io import save_index
    from app.services.preprocessing import (
        build_spec_from_manifest, build_transform,
        load_pil_image, preprocess_images,
    )

    output_dir = Path(output)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "_emb_checkpoint.npz"

    items = _collect_gallery(KAGGLE_ANIMAL90, per_class, seed)
    species_names = sorted({s for s, _ in items})
    species_to_class = {n: i for i, n in enumerate(species_names)}
    print(f"classes={len(species_names)} images={len(items)}")

    manifest = {
        "model": "dinov2_vits14",
        "image_size": IMAGE_SIZE,
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "embedding_dim": EMBEDDING_DIM,
        "metric": "cosine",
        "index_type": "IndexFlatIP",
        "version": "full90-v1",
        "num_classes": len(species_names),
        "gallery_size": len(items),
        "source": "animal-image-dataset-90-different-animals",
        "per_class": per_class,
        "seed": seed,
    }
    spec = build_spec_from_manifest(manifest)
    transform = build_transform(spec)

    print("Loading real DINOv2 ViT-S/14 ...")
    encoder = DinoV2Encoder(device=device)

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    meta_rows: List[Dict] = []
    if resume and ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        embeddings = [z["emb"][i] for i in range(z["emb"].shape[0])]
        labels = [int(v) for v in z["labels"]]
        meta_rows = [dict(r) for r in z["meta"].tolist()]
        print(f"resumed checkpoint: {len(embeddings)} embeddings")

    t0 = time.time()
    skipped = 0
    for pos, (name, src_path) in enumerate(items):
        cls = species_to_class[name]
        gid = f"{name}_{pos:05d}"
        rel = f"{name}/{gid}{src_path.suffix.lower()}"
        dst = gallery_dir / rel
        if resume and any(r.get("filename") == rel for r in meta_rows):
            skipped += 1
            continue
        try:
            data = src_path.read_bytes()
            pil = load_pil_image(data)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] skip {src_path}: {exc}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst)
        batch_t = preprocess_images([pil], transform)
        emb1 = encoder.encode(batch_t).detach().cpu().numpy().astype("float32")
        embeddings.append(emb1[0])
        labels.append(cls)
        meta_rows.append({"id": gid, "species": name,
                          "label": cls, "filename": rel})
        done_n = len(embeddings)
        if done_n % 50 == 0 or done_n == len(items):
            dt = time.time() - t0
            print(f"  encoded {done_n}/{len(items)} ({dt:.0f}s)", flush=True)
        if checkpoint_every > 0 and done_n % checkpoint_every == 0:
            np.savez_compressed(
                ckpt, emb=np.stack(embeddings),
                labels=np.asarray(labels, dtype=np.int64),
                meta=np.asarray(meta_rows, dtype=object))
            print(f"  checkpoint -> {ckpt} ({done_n})", flush=True)

    if skipped:
        print(f"skipped {skipped} already-encoded (resume)")
    if not embeddings:
        raise RuntimeError("No images encoded - check dataset path.")

    emb = _normalize(np.stack(embeddings, axis=0))
    labels_arr = np.asarray(labels, dtype=np.int64)
    manifest["gallery_size"] = int(emb.shape[0])

    print(f"training Calibrated LinearSVC on {emb.shape} ...")
    base = LinearSVC(max_iter=10000, C=1.0)
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(emb, labels_arr)
    print("classifier trained.")

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

    with (output_dir / "class_mapping.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"label_to_id": species_to_class,
             "id_to_label": {str(v): k for k, v in species_to_class.items()}},
            fh, indent=2)

    if ckpt.exists():
        ckpt.unlink()
    dt = time.time() - t0
    print(f"DONE full90 -> {output_dir} (gallery={emb.shape[0]}, "
          f"classes={len(species_names)}, {dt:.0f}s)")
    return output_dir


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build FULL Animal-90 artifacts")
    ap.add_argument("output", nargs="?", default="artifacts_full90")
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    build_full90_artifacts(
        output=args.output, per_class=args.per_class, seed=args.seed,
        device=args.device, checkpoint_every=args.checkpoint_every,
        resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

