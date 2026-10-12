"""Pydantic response schemas for the HTTP API.

The wire format mirrors the design in docs/plan.md. Field names and structure
are treated as a contract: changing them is a breaking API change.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Top-1 classification output."""

    species: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """A single gallery neighbour returned by the retrieval engine."""

    id: str
    species: str
    similarity: float
    image_url: str


class ModelInfo(BaseModel):
    """Metadata describing the loaded serving package."""

    encoder: str
    version: str


class SearchResponse(BaseModel):
    """Full response for POST /api/v1/search."""

    prediction: Prediction
    results: List[SearchResult]
    model: ModelInfo
    # Placeholder for a future OOD gate. Always present (null) so clients can
    # be written against the final shape now.
    is_unknown: Optional[bool] = None


class ModelInfoResponse(BaseModel):
    """Extended model info returned by GET /model-info."""

    encoder: str
    version: str
    embedding_dim: int
    num_classes: int
    gallery_size: int
    metric: str
    index_type: str
    image_size: int


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    model_loaded: bool
    encoder: Optional[str] = None
    version: Optional[str] = None


class ErrorResponse(BaseModel):
    """Uniform error body."""

    detail: str
