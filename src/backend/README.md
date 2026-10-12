# Animal Visual Search — Backend

FastAPI serving layer for the animal classification + visual search model.
It is **model-agnostic**: the API, response schema and routes never change when
the underlying encoder (DINOv2 today, ResNet/SigLIP2 later) is swapped, because
everything is driven by a serving **artifact package** produced by the notebook.

## Architecture

```
Client
  │  POST /api/v1/search  (image + k)
  ▼
FastAPI  (app/main.py)
  │  image validation + preprocessing  (contract read from manifest.json)
  ▼
Encoder  (DINOv2 ViT-S/14 → 384-D, L2-normalized)   [services/encoders.py]
  ├────────────── embedding ──────────────┐
  ▼                                        ▼
Classifier head                     FAISS IndexFlatIP
(species + confidence)              (Top-K neighbours, cosine)
  └──────────────┬───────────────────────┘
                 ▼
          Gallery metadata → image URLs → JSON
```

### Key design decisions (see `docs/plan.md`)

- **Contract-driven preprocessing.** Image size + normalization are read from
  `manifest.json`, never hard-coded, so the backend applies exactly the
  transform the model was trained/evaluated with.
- **One embedding, two jobs.** The same 384-D L2-normalized vector feeds both
  the classifier and the FAISS cosine index.
- **Portable gallery.** Retrieval maps `FAISS position → stable id → metadata →
  /gallery/<file>`, independent of any Colab/Kaggle filesystem paths.
- **Calibrated confidence.** The export wraps the notebook's `LinearSVC` in
  `CalibratedClassifierCV` so the API returns a real probability. Production
  artifacts must ship a calibrated head: by default the engine **fails fast at
  startup** if the loaded classifier lacks `predict_proba`. Set
  `REQUIRE_CALIBRATED=false` only for local/synthetic artifacts, which then
  fall back to a softmax over `decision_function`.
- **Single encoder pass.** `POST /search` calls `search_and_predict()`, which
  encodes the uploaded image exactly once and reuses that embedding for both
  classification and retrieval (verified by a test).

## Layout

```
src/backend/
├── app/
│   ├── main.py               # FastAPI app + startup (lifespan) wiring
│   ├── config.py             # env-driven settings
│   ├── schemas.py            # Pydantic request/response models
│   ├── api/routes.py         # /search, /health, /model-info, /gallery
│   └── services/
│       ├── artifacts.py      # load + validate the artifact package
│       ├── preprocessing.py  # manifest → transform, image validation
│       ├── encoders.py       # Encoder ABC, DinoV2Encoder, MockEncoder
│       └── engine.py         # AnimalSearchEngine + DinoSearchEngine
├── scripts/
│   ├── serving_export.py     # notebook "Serving Export" cell
│   └── make_synthetic_artifacts.py  # offline test fixture generator
├── tests/                    # pytest (runs fully offline)
├── requirements.txt
├── .env.example
└── pytest.ini
```

## Artifact package (produced by the notebook)

```
artifacts/
├── manifest.json                # preprocessing + model contract
├── class_mapping.json           # {id <-> species}
├── dino_classifier.joblib       # calibrated sklearn classifier
├── dino_gallery_embeddings.npy  # (N, 384) float32, L2-normalized
├── dino_gallery_labels.npy      # (N,) int
├── dino_gallery_metadata.csv    # id, species, label, filename
├── dino_gallery.faiss           # IndexFlatIP
└── gallery/<species>/<id>.jpg
```

## API

### `POST /api/v1/search`
`multipart/form-data`: `image` (file), `k` (int, default 5).

```json
{
  "prediction": { "species": "tiger", "confidence": 0.927 },
  "results": [
    { "id": "tiger_0012", "species": "tiger",
      "similarity": 0.941, "image_url": "/gallery/tiger/tiger_0012.jpg" }
  ],
  "model": { "encoder": "dinov2_vits14", "version": "clean-v1" },
  "is_unknown": null
}
```

### `GET /health`
Liveness + model-loaded probe.

### `GET /model-info`
Details of the loaded serving package (dims, index type, metric, ...).

### `GET /gallery/{file}`
Serves a gallery image (path-traversal safe).

## Running

```bash
cd src/backend
pip install -r requirements.txt

# 1) Build artifacts. For real serving, run the notebook "Serving Export" cell.
#    For a quick offline smoke test, generate synthetic artifacts:
python scripts/make_synthetic_artifacts.py ./artifacts

# 2) Point the app at them and choose the encoder.
#    Real DINOv2 (downloads weights on first run):
ENCODER=dinov2 ARTIFACTS_DIR=./artifacts uvicorn app.main:app --reload
#    Offline mock (pairs with synthetic artifacts):
ENCODER=mock ARTIFACTS_DIR=./artifacts uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive Swagger UI.

## Testing

```bash
cd src/backend
pytest
```

The suite builds a synthetic artifact package and drives the full stack through
the offline `MockEncoder`, so it needs no GPU, no network and no DINOv2 weights.
