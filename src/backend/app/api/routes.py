"""HTTP routes for the Animal Visual Search API.

Endpoints
---------
* ``POST /api/v1/search``  — upload an image, get species + confidence + top-K.
* ``GET  /health``         — liveness / model-loaded probe.
* ``GET  /model-info``     — details of the loaded serving package.
* ``GET  /gallery/{file}`` — serve a gallery image by filename.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    SearchResponse,
)

api_router = APIRouter(prefix="/api/v1")
gallery_router = APIRouter()


def get_engine(request: Request):
    """Fetch the engine attached to app state, or 503 if not ready."""

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return engine


def get_settings(request: Request):
    return request.app.state.settings


@api_router.post(
    "/search",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def search(
    request: Request,
    image: UploadFile = File(..., description="Image file to search for."),
    k: int = Form(5, ge=1, description="Number of neighbours to return."),
    engine=Depends(get_engine),
):
    settings = get_settings(request)

    from ..services.preprocessing import InvalidImageError, validate_upload

    data = await image.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        validate_upload(data, image.content_type, max_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    k = min(k, settings.max_k)

    try:
        # Single encoder pass: classification + retrieval share one embedding.
        outcome = engine.search_and_predict(data, k)
        prediction = outcome.prediction
        results = outcome.results
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return SearchResponse(
        prediction={"species": prediction.species, "confidence": prediction.confidence},
        results=[
            {
                "id": r.id,
                "species": r.species,
                "similarity": r.similarity,
                "image_url": r.image_url,
            }
            for r in results
        ],
        model={"encoder": engine.encoder_name, "version": engine.model_version},
        is_unknown=None,
    )


@api_router.get("/model-info", response_model=ModelInfoResponse)
def model_info(engine=Depends(get_engine)):
    return engine.info()


@gallery_router.get("/health", response_model=HealthResponse)
def health(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return HealthResponse(status="starting", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        encoder=engine.encoder_name,
        version=engine.model_version,
    )


@gallery_router.get("/gallery/{filename:path}")
def gallery_file(request: Request, filename: str):
    """Serve a gallery image. Resolves safely under the gallery directory."""

    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    gallery_dir = Path(engine._bundle.artifacts_dir) / "gallery"
    # Prevent path traversal: resolve and ensure it stays under gallery_dir.
    safe_root = gallery_dir.resolve()
    target = (safe_root / filename).resolve()
    if not str(target).startswith(str(safe_root)) or not target.is_file():
        raise HTTPException(status_code=404, detail="Gallery image not found.")

    return FileResponse(str(target))
