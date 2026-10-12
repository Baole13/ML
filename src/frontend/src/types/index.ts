// TypeScript types mirroring the FastAPI backend contract (v1, frozen for FE).
// Source of truth: src/backend/app/schemas.py and docs/plan.md.
// If these ever change, it is a breaking API change — update here too.

export interface Prediction {
  species: string;
  /** Calibrated probability in [0, 1]. */
  confidence: number;
}

export interface SearchResult {
  id: string;
  species: string;
  /** Cosine similarity in [-1, 1]. */
  similarity: number;
  /** Relative URL, e.g. "/gallery/tiger/tiger_0012.jpg". */
  image_url: string;
}

export interface ModelInfo {
  encoder: string;
  version: string;
}

export interface SearchResponse {
  prediction: Prediction;
  results: SearchResult[];
  model: ModelInfo;
  /** Placeholder for a future OOD gate; always null in v1. */
  is_unknown: boolean | null;
}

export interface ModelInfoResponse {
  encoder: string;
  version: string;
  embedding_dim: number;
  num_classes: number;
  gallery_size: number;
  metric: string;
  index_type: string;
  image_size: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  encoder: string | null;
  version: string | null;
}
