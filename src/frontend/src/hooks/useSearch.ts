import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../services/api";
import type { ModelInfoResponse, SearchResponse } from "../types";

export type SearchStatus =
  | "empty" // no image chosen yet
  | "selected" // image chosen, not yet searched
  | "loading" // request in flight
  | "result" // got a response
  | "error"; // request failed

export interface UseSearchState {
  status: SearchStatus;
  file: File | null;
  previewUrl: string | null;
  k: number;
  response: SearchResponse | null;
  error: string | null;
  elapsedMs: number | null;
}

export interface UseSearch extends UseSearchState {
  selectFile: (file: File | null) => void;
  setK: (k: number) => void;
  run: () => void;
  reset: () => void;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_BYTES = 10 * 1024 * 1024; // must be <= backend MAX_UPLOAD_MB
const DEFAULT_K = 5;
const MAX_K = 20;

export function useSearch(): UseSearch {
  const [status, setStatus] = useState<SearchStatus>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [k, setK] = useState<number>(DEFAULT_K);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // Manage the object URL lifecycle for the query-image preview.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Abort any in-flight request on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const selectFile = useCallback((next: File | null) => {
    if (!next) {
      setFile(null);
      setResponse(null);
      setError(null);
      setStatus("empty");
      return;
    }
    if (!ACCEPTED.includes(next.type)) {
      setError(`Unsupported file type: ${next.type || "unknown"}. Use JPG/PNG/WebP.`);
      setStatus("error");
      return;
    }
    if (next.size > MAX_BYTES) {
      setError("Image is larger than 10 MB.");
      setStatus("error");
      return;
    }
    setError(null);
    setResponse(null);
    setFile(next);
    setStatus("selected");
  }, []);

  const run = useCallback(() => {
    if (!file) return;

    // Cancel any previous request.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("loading");
    setError(null);
    const started = performance.now();

    api
      .search(file, k, controller.signal)
      .then((res) => {
        setResponse(res);
        setElapsedMs(performance.now() - started);
        setStatus("result");
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message =
          err instanceof api.ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Something went wrong.";
        setError(message);
        setStatus("error");
      });
  }, [file, k]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setFile(null);
    setResponse(null);
    setError(null);
    setElapsedMs(null);
    setStatus("empty");
  }, []);

  const changeK = useCallback((next: number) => {
    const clamped = Math.max(1, Math.min(MAX_K, Math.round(next) || 1));
    setK(clamped);
  }, []);

  return {
    status,
    file,
    previewUrl,
    k,
    response,
    error,
    elapsedMs,
    selectFile,
    setK: changeK,
    run,
    reset,
  };
}

/** Fetch model info once on mount for the info panel. */
export function useModelInfo(): ModelInfoResponse | null {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api
      .modelInfo(controller.signal)
      .then(setInfo)
      .catch(() => setInfo(null)); // panel is optional; ignore failures
    return () => controller.abort();
  }, []);
  return info;
}
