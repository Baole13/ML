# Animal Visual Search

An end-to-end Machine Learning project for **animal species classification** and **content-based image retrieval**.

Given a single animal image, the system predicts the most likely species with a confidence score and returns the **Top-K visually related images** from a gallery. The project combines classical Machine Learning, transfer learning, self-supervised visual representations, cross-dataset evaluation, and a deployable FastAPI serving layer.

---

## 1. Project Objectives

The project is developed as a final course project for Machine Learning with the following goals:

- Use at least **two independently collected datasets**.
- Compare classical and modern image representations.
- Perform both **classification** and **image retrieval**.
- Evaluate models using appropriate metrics instead of classification accuracy alone.
- Analyze data quality, leakage, generalization, and model errors.
- Include modern ML techniques beyond the basic course content.
- Build an end-to-end application that can later be consumed by a web frontend.

The target user flow is:

```text
Animal image
    │
    ▼
Visual encoder
    │
    ├── Species classifier ──► Species + confidence
    │
    └── Retrieval index ─────► Top-K related images
```

---

## 2. Datasets

The clean benchmark notebook currently uses two Kaggle datasets:

### Animal Image Dataset — 90 Different Animals

Kaggle dataset:

```text
iamsouravbanerjee/animal-image-dataset-90-different-animals
```

This is the main dataset used for training, validation, testing, and the retrieval gallery.

### Wild Animals Images

Kaggle dataset:

```text
whenamancodes/wild-animals-images
```

This dataset is used as an external dataset for **cross-dataset retrieval evaluation** and domain-shift analysis.

When several resized versions of the same Wild Animals collection are available, the notebook prefers the `224 × 224` version to avoid counting the same content multiple times.

---

## 3. Clean Benchmark Protocol

The current benchmark notebook is:

```text
notebook/Animal_Visual_Search_Clean_Benchmark_v1.ipynb
```

The benchmark was redesigned to avoid optimistic results caused by duplicate or near-duplicate images.

The main rules are:

1. Audit images **before splitting**.
2. Detect exact duplicates using **SHA256**.
3. Detect perceptual near-duplicates using **dHash + pHash**.
4. Build duplicate groups and keep a clean representative set.
5. Remove ambiguous duplicate groups that contain conflicting class labels.
6. Create a fixed **Train / Validation / Test** split only after cleaning.
7. Verify that no duplicate leakage remains between the splits.
8. Use the **same fixed split for every model**.
9. Audit Animal-90 against Wild Animals before cross-dataset evaluation.
10. Report classification and retrieval metrics separately.

For the official experiment, set:

```python
QUICK_MODE = False
```

`QUICK_MODE=True` is only intended as a pipeline smoke test.

---

## 4. Models and Representations

The benchmark compares representations from traditional Machine Learning to modern self-supervised learning.

### PCA + KNN

```text
Image
  ↓
Resize / flatten
  ↓
PCA
  ↓
KNN
```

Used as a low-level classical ML baseline.

### HOG + Linear SVM

```text
Image
  ↓
HOG descriptor
  ↓
Linear SVM
```

Used to evaluate a handcrafted visual representation.

### ResNet18 Fine-Tuned

```text
Image
  ↓
ImageNet-pretrained ResNet18
  ↓
Fine-tuning
  ├── Classification head
  └── Deep visual embedding
```

Represents supervised deep feature learning with transfer learning.

### DINOv2 + Linear SVM / FAISS

```text
Image
  ↓
DINOv2 ViT-S/14
  ↓
384-D L2-normalized embedding
  ├── Linear / calibrated classifier
  └── FAISS IndexFlatIP
```

DINOv2 is used as the modern self-supervised representation.

For serving, the same embedding is designed to support both classification and retrieval.

---

## 5. Evaluation Metrics

### Classification

The project evaluates:

- Accuracy
- Macro-F1

### Image Retrieval

The project evaluates:

- Precision@5
- mAP@10

### Cross-Dataset Retrieval

Wild Animals queries are evaluated against the Animal-90 gallery using:

- Cross-dataset Precision@5
- Cross-dataset mAP@10

The notebook exports the official comparison table to:

```text
benchmark_results.csv
```

Data-integrity statistics are exported separately to:

```text
integrity_summary.csv
```

No preliminary Quick Mode metric should be treated as the final project result.

---

## 6. Experimental Results

### 6.1 Preliminary benchmark

The following results were obtained during the iterative development of the project before the final clean full-mode benchmark was frozen.

> **Important:** these values are kept as **preliminary experimental evidence only**.  
> A later integrity audit found perceptual duplicate leakage in the earlier split, so these numbers must **not** be reported as the final clean benchmark.

| Model | Classification Acc. | Macro-F1 | Retrieval P@5 | Retrieval mAP@10 | Cross-dataset P@5 | Cross-dataset mAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| PCA + KNN | 0.2333 | 0.2216 | — | — | — | — |
| HOG + Linear SVM | 0.2750 | 0.2617 | 0.0883 | 0.0397 | — | — |
| ResNet18 fine-tuned | 0.9250 | 0.9248 | 0.9183 | 0.9014 | 0.9020 | 0.8634 |
| DINOv2 frozen + Linear SVM / FAISS | **1.0000** | **1.0000** | **0.9617** | **0.9250** | **0.9703** | **0.9310** |

### 6.2 Preliminary observations

The preliminary runs show a clear progression in representation quality.

**Classical ML baselines.** PCA + KNN achieved only about 23% classification accuracy. Replacing low-level PCA features with HOG and a supervised Linear SVM improved classification modestly to 27.5%, but HOG remained weak for retrieval: P@5 was only 8.83% and mAP@10 was 3.97%. This suggests that handcrafted edge descriptors can support a discriminative decision boundary better than they support a semantic nearest-neighbour space.

**ResNet18.** Fine-tuned ResNet18 produced a large improvement over the classical baselines, reaching 92.5% preliminary classification accuracy and 91.83% P@5. Its cross-dataset P@5 of 90.20% also showed that the learned deep representation transferred reasonably well to images from a different dataset.

**DINOv2.** Frozen DINOv2 embeddings combined with a linear classifier and FAISS produced the strongest preliminary results. The most important trend was not only the perfect preliminary classification score, but also stronger retrieval and cross-dataset ranking performance: 96.17% P@5 and 93.10% cross-dataset mAP@10.

The main hypothesis suggested by these experiments is:

> **The quality of the visual representation is the dominant factor for both classification and semantic image retrieval, and self-supervised DINOv2 features appear to generalize better across datasets than the supervised ResNet18 representation.**

However, this statement remains a **preliminary hypothesis** until the full clean benchmark is completed.

### 6.3 Why the preliminary numbers are not final

During development, the data-integrity audit detected duplicate or perceptually identical samples across the original Train / Validation / Test split. In one audited run, duplicate candidates were found between:

```text
Train ↔ Validation
Train ↔ Test
```

This means that the earlier metrics may be optimistic.

The benchmark was therefore redesigned so that:

```text
Raw data
   ↓
Corrupted-image audit
   ↓
SHA256 + dHash + pHash duplicate grouping
   ↓
Clean representative dataset
   ↓
Fixed Train / Validation / Test split
   ↓
All models evaluated on the exact same split
```

The same principle is also applied to the external Wild Animals evaluation by removing cross-dataset duplicate candidates before computing cross-dataset retrieval metrics.

### 6.4 Official clean benchmark

The authoritative final results must come from:

```text
notebook/Animal_Visual_Search_Clean_Benchmark_v1.ipynb
```

with:

```python
QUICK_MODE = False
```

The final table is generated in **Phase 18 — Benchmark Dashboard** and exported as:

```text
benchmark_results.csv
```

The integrity report is exported as:

```text
integrity_summary.csv
```

At the current repository snapshot, the clean benchmark pipeline is implemented, but a completed full-mode `benchmark_results.csv` is not included in the repository. Therefore, this README intentionally does **not** invent or promote preliminary values as final results.

Once the full-mode benchmark is executed, this section should be replaced or extended with the authoritative table:

| Model | Classification Acc. | Macro-F1 | Retrieval P@5 | Retrieval mAP@10 | Cross-dataset P@5 | Cross-dataset mAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| PCA + KNN | TBD | TBD | TBD | TBD | — | — |
| HOG + Linear SVM | TBD | TBD | TBD | TBD | — | — |
| ResNet18 fine-tuned | TBD | TBD | TBD | TBD | TBD | TBD |
| DINOv2 frozen + Linear SVM / FAISS | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 7. System Architecture

The application is divided into three layers:

```text
┌─────────────────────────────┐
│          Frontend           │
│   Upload image / show Top-K │
└──────────────┬──────────────┘
               │ HTTP
               ▼
┌─────────────────────────────┐
│       FastAPI Backend       │
│ validation / API / gallery  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       ML Serving Layer      │
│                             │
│ DINOv2 encoder              │
│   ├── classifier            │
│   └── FAISS retrieval       │
└─────────────────────────────┘
```

The backend is intentionally **model-agnostic**. The public API should remain stable even if DINOv2 is replaced later by another encoder such as ResNet or SigLIP.

---

## 8. Repository Structure

```text
BTL/
├── docs/
│   └── plan.md
│
├── notebook/
│   └── Animal_Visual_Search_Clean_Benchmark_v1.ipynb
│
└── src/
    ├── backend/
    │   ├── app/
    │   │   ├── api/
    │   │   │   └── routes.py
    │   │   ├── services/
    │   │   │   ├── artifacts.py
    │   │   │   ├── encoders.py
    │   │   │   ├── engine.py
    │   │   │   └── preprocessing.py
    │   │   ├── config.py
    │   │   ├── main.py
    │   │   └── schemas.py
    │   │
    │   ├── scripts/
    │   │   ├── make_synthetic_artifacts.py
    │   │   └── serving_export.py
    │   │
    │   ├── tests/
    │   │   ├── conftest.py
    │   │   ├── test_api.py
    │   │   └── test_engine.py
    │   │
    │   ├── .env.example
    │   ├── NOTEBOOK_INTEGRATION.md
    │   ├── README.md
    │   └── requirements.txt
    │
    └── frontend/
        └── # frontend implementation
```

---

## 9. Serving Artifact Package

The notebook and `src/backend/scripts/serving_export.py` are designed to export a portable model package:

```text
artifacts/
├── manifest.json
├── class_mapping.json
├── dino_classifier.joblib
├── dino_gallery_embeddings.npy
├── dino_gallery_labels.npy
├── dino_gallery_metadata.csv
├── dino_gallery.faiss
└── gallery/
    └── <species>/
        └── <image>.jpg
```

### Why this package exists

The backend must not depend on temporary Colab or Kaggle cache paths.

The serving export therefore:

- assigns stable gallery IDs;
- copies gallery images into a portable folder;
- stores species metadata;
- stores the FAISS index;
- stores the classification head;
- records the preprocessing contract in `manifest.json`.

A typical manifest contains:

```json
{
  "model": "dinov2_vits14",
  "image_size": 224,
  "normalization": "imagenet",
  "embedding_dim": 384,
  "metric": "cosine",
  "index_type": "IndexFlatIP",
  "version": "clean-v1"
}
```

The backend reads this contract instead of hard-coding the image preprocessing configuration.

---

## 10. Backend

The serving API is implemented with **FastAPI**.

### Requirements

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- FAISS CPU
- FastAPI
- Uvicorn

Install the dependencies:

```bash
cd src/backend
pip install -r requirements.txt
```

### Environment Configuration

Copy the example environment file if needed:

```bash
cp .env.example .env
```

Important settings include:

```env
ARTIFACTS_DIR=./artifacts
ENCODER=dinov2
DEVICE=auto
MAX_K=20
MAX_UPLOAD_MB=10
CORS_ORIGINS=*
GALLERY_URL_PREFIX=/gallery
ENABLE_UNKNOWN_GATE=false
```

---

## 11. Running the Backend

### Real DINOv2 artifacts

First export the serving package from the notebook and place it under:

```text
src/backend/artifacts/
```

Then run:

```bash
cd src/backend

ENCODER=dinov2 \
ARTIFACTS_DIR=./artifacts \
uvicorn app.main:app --reload
```

The first DINOv2 run may download model weights through `torch.hub`.

### Offline Smoke Test

A synthetic artifact package can be generated without using the trained model:

```bash
cd src/backend

python scripts/make_synthetic_artifacts.py ./artifacts

ENCODER=mock \
ARTIFACTS_DIR=./artifacts \
uvicorn app.main:app --reload
```

The API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 12. API

### Search

```http
POST /api/v1/search
Content-Type: multipart/form-data
```

Form fields:

```text
image = image file
k     = number of neighbours, default 5
```

Example response:

```json
{
  "prediction": {
    "species": "tiger",
    "confidence": 0.927
  },
  "results": [
    {
      "id": "tiger_0012",
      "species": "tiger",
      "similarity": 0.941,
      "image_url": "/gallery/tiger/tiger_0012.jpg"
    }
  ],
  "model": {
    "encoder": "dinov2_vits14",
    "version": "clean-v1"
  },
  "is_unknown": null
}
```

### Health Check

```http
GET /health
```

Returns whether the API is alive and whether the model has been loaded.

### Model Information

```http
GET /api/v1/model-info
```

Returns information such as:

- encoder name;
- model version;
- embedding dimension;
- number of classes;
- gallery size;
- similarity metric;
- index type;
- input image size.

### Gallery Image

```http
GET /gallery/{filename}
```

Serves a retrieved gallery image.

The implementation validates the resolved path to prevent gallery path traversal.

---

## 13. Testing

The backend includes an offline test suite.

Run:

```bash
cd src/backend
pytest
```

The tests use the deterministic `MockEncoder` and synthetic artifacts, so they do not require GPU access or DINOv2 model weights.

Covered behaviors include:

- health endpoint;
- model information;
- search response schema;
- default Top-K;
- server-side K limit;
- invalid upload handling;
- missing image handling;
- gallery image serving;
- gallery path-traversal protection;
- inference/retrieval engine behavior.

---

## 14. Frontend

The frontend layer is the next application phase.

The planned frontend only depends on the stable backend API and does not need to know which ML encoder is currently active.

Recommended stack:

```text
React + TypeScript + Vite
```

Suggested MVP flow:

```text
Empty state
   ↓
Select / drag-and-drop animal image
   ↓
Image preview
   ↓
POST /api/v1/search
   ↓
Loading state
   ↓
Species + confidence
   ↓
Top-K similar image grid
```

Recommended frontend components:

```text
ImageUploader
PredictionCard
ResultGrid
ResultCard
ModelInfo
```

The frontend should focus on demonstrating the ML system rather than adding unrelated product features such as authentication or user profiles.

---

## 15. Notebook → Backend Integration

After the clean benchmark finishes, run the serving export using:

```python
from serving_export import export_serving_package

export_serving_package(
    output_dir=OUTPUT_DIR / "artifacts",
    gallery_dino=gallery_dino,
    gallery_labels_dino=gallery_labels_dino,
    gallery_paths_dino=gallery_paths_dino,
    dino_clf=dino_clf,
    label_names=label_names,
    model_name="dinov2_vits14",
    version="clean-v1",
    image_size=IMG_SIZE_CNN,
)
```

The full integration guide is available at:

```text
src/backend/NOTEBOOK_INTEGRATION.md
```

---

## 16. Current Project Status

| Area | Status |
|---|---|
| Dataset acquisition | ✅ |
| Data-quality audit | ✅ |
| Duplicate / near-duplicate strategy | ✅ |
| Fixed clean benchmark split | ✅ |
| PCA + KNN baseline | ✅ |
| HOG + SVM baseline | ✅ |
| ResNet18 experiment | ✅ |
| DINOv2 representation | ✅ |
| FAISS retrieval | ✅ |
| Cross-dataset evaluation pipeline | ✅ |
| Serving artifact exporter | ✅ |
| FastAPI backend | ✅ |
| API schemas and validation | ✅ |
| Backend tests | ✅ |
| Real serving artifact export | 🔄 Run from final notebook |
| Frontend | 🔄 Next phase |
| OOD / unknown-species detection | Planned |
| SigLIP comparison | Optional / planned |
| Grad-CAM and extended failure analysis | Optional / planned |

---

## 17. Planned Extensions

After the end-to-end application is working, possible research extensions include:

- ResNet18 from-scratch vs frozen vs fine-tuned comparison;
- confidence calibration analysis using ECE / reliability diagrams;
- open-set / OOD detection for unknown species;
- SigLIP2 comparison;
- retrieval failure taxonomy;
- Grad-CAM visualization;
- improved frontend visualization and model explanation.

These extensions should not change the main frontend API contract.

---

## 18. Reproducibility Notes

For official benchmark results:

1. Use the clean benchmark notebook.
2. Set:

   ```python
   QUICK_MODE = False
   ```

3. Restart the Colab runtime.
4. Run all cells from the beginning.
5. Keep the generated fixed split unchanged when comparing models.
6. Use the Phase 18 benchmark dashboard / `benchmark_results.csv`.
7. Do not report preliminary Quick Mode metrics as final results.

Important generated files include:

```text
fixed_clean_split.csv
benchmark_results.csv
integrity_summary.csv
resnet_test_predictions.csv
animal90_clean_metadata.csv
wild_animals_clean_external_metadata.csv
```

---

## 19. Tech Stack

### Machine Learning

- Python
- NumPy
- pandas
- scikit-learn
- scikit-image
- PyTorch / torchvision
- DINOv2
- FAISS
- Pillow

### Backend

- FastAPI
- Uvicorn
- Pydantic
- pytest

### Frontend

- React
- TypeScript
- Vite

---

## 20. Main Research Question

At a high level, the project investigates:

> **How does the choice of visual representation affect animal classification, image retrieval, and cross-dataset generalization?**

The experimental progression is:

```text
PCA
  ↓
HOG
  ↓
Supervised CNN representation
  ↓
Self-supervised DINOv2 representation
```

This allows the project to evaluate not only whether a model can classify an animal, but whether its learned representation creates a useful semantic space for retrieving related images from both in-domain and external data.

---

## License

This repository is intended for academic/coursework use. Dataset usage remains subject to the licenses and terms of the original dataset providers.
