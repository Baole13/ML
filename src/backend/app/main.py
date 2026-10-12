"""FastAPI application entrypoint.

The engine is built once during application startup (lifespan) and attached to
``app.state`` so every request reuses the same loaded model, FAISS index and
preprocessing contract. Startup failures (e.g. missing artifacts) are logged
but do not crash the process — ``/health`` reports the true state and the
search endpoints return 503 until the model is available.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import api_router, gallery_router
from .config import load_settings
from .schemas import ErrorResponse

logger = logging.getLogger("animal_search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.engine = None

    try:
        from .services.engine import build_engine

        engine = build_engine(settings)
        app.state.engine = engine
        logger.info(
            "Engine ready: encoder=%s version=%s gallery=%d classes=%d",
            engine.encoder_name,
            engine.model_version,
            engine.info()["gallery_size"],
            engine.info()["num_classes"],
        )
    except Exception as exc:  # pragma: no cover - depends on artifacts
        logger.error("Failed to load engine at startup: %s", exc)

    yield

    app.state.engine = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Animal Visual Search API",
        version="1.0.0",
        description=(
            "Upload an animal image to get a species prediction with "
            "confidence and the top-K most similar gallery images."
        ),
        lifespan=lifespan,
    )

    # CORS: default permissive for local dev; lock down via CORS_ORIGINS.
    try:
        settings = load_settings()
        origins = settings.cors_origins
    except Exception:
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root health + gallery live at the app root; versioned API under /api/v1.
    app.include_router(gallery_router)
    app.include_router(api_router)

    @app.get("/", tags=["meta"])
    def root():
        return {
            "service": "animal-visual-search",
            "docs": "/docs",
            "health": "/health",
            "search": "/api/v1/search",
        }

    return app


app = create_app()
