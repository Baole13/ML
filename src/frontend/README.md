# Animal Visual Search — Frontend

React 18 + TypeScript + Vite single-page app for the Animal Visual Search
backend. It uploads an animal image, shows the predicted species + confidence,
and renders the top-K most similar gallery images.

```
React (this)  ──HTTP──▶  FastAPI backend  ──▶  DINOv2 + classifier + FAISS
```

The frontend only knows the **frozen v1 API contract** — it is unaffected by
future backend model swaps (DINOv2 → SigLIP2, etc.).

## Stack

- **React 18** + **TypeScript** (strict) + **Vite 5**
- No UI framework — hand-written CSS (dark theme) in `src/index.css`
- `fetch` + `FormData` for uploads; no extra HTTP dependency

## Layout

```
src/
├── main.tsx                 # entry point
├── App.tsx                  # layout + state wiring
├── index.css                # all styling
├── types/index.ts           # types mirroring the API contract (frozen v1)
├── services/api.ts          # API client (search / modelInfo / health)
├── hooks/useSearch.ts       # state machine + model-info hook
└── components/
    ├── ImageUploader.tsx    # drag & drop + preview
    ├── KSelector.tsx        # choose K (1–20)
    ├── PredictionCard.tsx   # species + confidence bar
    ├── ResultGrid.tsx       # responsive grid of neighbours
    ├── ResultCard.tsx       # one neighbour (image + species + similarity)
    ├── ModelInfoPanel.tsx   # model / classes / retrieval info
    └── States.tsx           # Loading / Error / Empty
```

## UX states

```
EMPTY ──select──▶ SELECTED ──search──▶ LOADING ──▶ RESULT
                                             └────▶ ERROR
```

## Run

```bash
cd src/frontend
npm install

# 1) Start the backend first (see ../backend/README.md). For an offline demo:
#    cd ../backend
#    python scripts/make_synthetic_artifacts.py ./artifacts
#    ENCODER=mock ARTIFACTS_DIR=./artifacts uvicorn app.main:app --port 8000

# 2) Start the dev server (proxies /api and /gallery to :8000)
npm run dev
```

Open http://127.0.0.1:5173.

## Configuration

Copy `.env.example` to `.env`:

- `VITE_API_BASE_URL` — leave **empty** for dev (Vite proxy handles it). In
  production set it to the backend origin, e.g. `https://api.example.com`.
- `VITE_DEV_API_TARGET` — (dev only) where the proxy forwards API calls;
  default `http://127.0.0.1:8000`.

## Build

```bash
npm run build      # tsc -b && vite build  → dist/
npm run preview    # serve the production build locally
npm run lint       # tsc --noEmit (type-check only)
```

## API contract consumed

```
POST /api/v1/search   (multipart: image, k)  →  { prediction, results, model, is_unknown }
GET  /api/v1/model-info                     →  model metadata (info panel)
GET  /gallery/<file>                        →  gallery images
```

See `src/types/index.ts` for the exact TypeScript shapes. The API is treated as
a frozen contract: changing it is a breaking change on both sides.
