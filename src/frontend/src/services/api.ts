// Thin API client for the Animal Visual Search backend.
//
// In dev the Vite proxy forwards /api and /gallery to the backend, so an empty
// base URL works. In production set VITE_API_BASE_URL to the backend origin.
// Gallery URLs returned by the API are relative, so we resolve them against the
// same base.

const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Resolve a backend-relative URL (e.g. "/gallery/...") to a full src. */
export function resolveUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${BASE}${path}`;
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data && typeof data.detail === "string") return data.detail;
  } catch {
    /* fall through */
  }
  return `Request failed with status ${res.status}`;
}

/** Run a visual search: upload an image, get species + top-K neighbours. */
export async function search(
  image: File,
  k: number,
  signal?: AbortSignal,
): Promise<import("../types").SearchResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("k", String(k));

  const res = await fetch(`${BASE}/api/v1/search`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }
  return (await res.json()) as import("../types").SearchResponse;
}

/** Fetch metadata about the loaded serving package. */
export async function modelInfo(
  signal?: AbortSignal,
): Promise<import("../types").ModelInfoResponse> {
  const res = await fetch(`${BASE}/api/v1/model-info`, { signal });
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return (await res.json()) as import("../types").ModelInfoResponse;
}

/** Health probe. */
export async function health(
  signal?: AbortSignal,
): Promise<import("../types").HealthResponse> {
  const res = await fetch(`${BASE}/health`, { signal });
  if (!res.ok) throw new ApiError(await parseError(res), res.status);
  return (await res.json()) as import("../types").HealthResponse;
}
