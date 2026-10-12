"""Service layer: artifact loading, preprocessing, encoders, search engine."""

from .artifacts import ArtifactBundle, load_artifacts
from .engine import (
    AnimalSearchEngine,
    DinoSearchEngine,
    Prediction,
    SearchResult,
    SearchOutcome,
    build_engine,
)

__all__ = [
    "ArtifactBundle",
    "load_artifacts",
    "AnimalSearchEngine",
    "DinoSearchEngine",
    "Prediction",
    "SearchResult",
    "SearchOutcome",
    "build_engine",
]
