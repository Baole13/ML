"""Shared pytest fixtures.

Builds a synthetic serving package once per session and exposes a FastAPI
``TestClient`` wired to it. The synthetic manifest selects the offline
``MockEncoder`` so tests never touch the network or download DINOv2 weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the backend package importable when running pytest from anywhere.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    from scripts.make_synthetic_artifacts import make_synthetic_artifacts

    target = tmp_path_factory.mktemp("artifacts")
    make_synthetic_artifacts(target)
    return target


@pytest.fixture()
def settings(artifacts_dir, monkeypatch):
    from app.config import Settings

    # Force the offline mock encoder + point at the synthetic package.
    monkeypatch.setenv("ENCODER", "mock")
    s = Settings(artifacts_dir=artifacts_dir)
    s.encoder = "mock"
    return s


@pytest.fixture()
def engine(settings):
    from app.services.engine import build_engine

    return build_engine(settings)


@pytest.fixture()
def client(artifacts_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("ENCODER", "mock")
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    app = create_app()
    # TestClient triggers the lifespan (engine build) on context enter.
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """A JPEG belonging to species index 0 (same recipe as the generator)."""

    import io

    from scripts.make_synthetic_artifacts import _make_species_image

    img = _make_species_image(0, 0)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
