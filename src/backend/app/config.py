"""Application configuration.

Values are read from environment variables so the same container can be
pointed at different serving artifacts without code changes. The manifest
shipped inside the artifacts directory is the *authoritative* source for the
preprocessing contract; settings here only control infrastructure concerns
(paths, device, limits, CORS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class Settings:
    """Runtime settings for the backend service."""

    # Directory containing the serving artifact package produced by the
    # notebook "Serving Export" cell (manifest.json, classifier, faiss, ...).
    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ARTIFACTS_DIR",
                str(Path(__file__).resolve().parents[2] / "artifacts"),
            )
        )
    )

    # "auto" -> use CUDA if available, otherwise CPU. Can be forced to
    # "cpu"/"cuda" for reproducibility or constrained deployments.
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto").lower())

    # Encoder backend. "dinov2" loads the real DINOv2 ViT-S/14 model from
    # torch.hub. "mock" uses a deterministic offline encoder (used by tests).
    encoder: str = field(default_factory=lambda: os.getenv("ENCODER", "dinov2").lower())

    # Maximum number of neighbours the search endpoint will return.
    max_k: int = field(default_factory=lambda: _get_int("MAX_K", 20))

    # Maximum accepted upload size in megabytes.
    max_upload_mb: int = field(default_factory=lambda: _get_int("MAX_UPLOAD_MB", 10))

    # Comma separated list of allowed CORS origins, or "*".
    cors_origins: List[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "*").split(",")
            if o.strip()
        ]
    )

    # Public URL prefix under which gallery images are exposed.
    gallery_url_prefix: str = field(
        default_factory=lambda: os.getenv("GALLERY_URL_PREFIX", "/gallery")
    )

    # Reserved for a future OOD / calibration gate. Kept as a first-class flag
    # so the response schema never has to change when it is switched on.
    enable_unknown_gate: bool = field(
        default_factory=lambda: _get_bool("ENABLE_UNKNOWN_GATE", False)
    )

    # Production artifacts MUST carry a calibrated classifier (predict_proba).
    # When true (default), startup fails fast if the loaded head is
    # uncalibrated. Set to false only for local dev / synthetic artifacts.
    require_calibrated: bool = field(
        default_factory=lambda: _get_bool("REQUIRE_CALIBRATED", True)
    )

    def resolve_device(self) -> str:
        if self.device in {"cpu", "cuda"}:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def load_settings() -> Settings:
    return Settings()
