import { useModelInfo, useSearch } from "./hooks/useSearch";
import { ImageUploader } from "./components/ImageUploader";
import { KSelector } from "./components/KSelector";
import { PredictionCard } from "./components/PredictionCard";
import { ResultGrid } from "./components/ResultGrid";
import { ModelInfoPanel } from "./components/ModelInfoPanel";
import { EmptyState, ErrorState, LoadingState } from "./components/States";

export default function App() {
  const s = useSearch();
  const modelInfo = useModelInfo();

  const busy = s.status === "loading";
  const canSearch = s.status === "selected" && !busy;

  return (
    <div className="app">
      <header className="app__header">
        <h1>🐾 Animal Visual Search</h1>
        <p className="muted">
          Upload an animal photo — get its species and the most similar gallery
          images.
        </p>
      </header>

      <main className="app__main">
        <section className="panel panel--controls">
          <ImageUploader
            previewUrl={s.previewUrl}
            fileName={s.file?.name ?? null}
            disabled={busy}
            onSelect={s.selectFile}
          />

          <KSelector value={s.k} disabled={busy} onChange={s.setK} />

          <div className="actions">
            <button
              type="button"
              className="btn btn--primary"
              disabled={!canSearch}
              onClick={s.run}
            >
              {busy ? "Searching…" : "Search"}
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              disabled={busy}
              onClick={s.reset}
            >
              Reset
            </button>
          </div>

          {s.status === "error" && s.error && (
            <ErrorState message={s.error} onRetry={canSearch ? s.run : undefined} onReset={s.reset} />
          )}
        </section>

        <section className="panel panel--results">
          {s.status === "loading" && <LoadingState />}

          {s.status === "empty" && <EmptyState />}

          {s.status === "selected" && (
            <EmptyState message="Ready — press Search to find similar images." />
          )}

          {s.status === "result" && s.response && (
            <>
              <PredictionCard prediction={s.response.prediction} />

              <div className="results__head">
                <h2>Similar images</h2>
                {s.elapsedMs != null && (
                  <span className="muted">{Math.round(s.elapsedMs)} ms</span>
                )}
              </div>

              <ResultGrid results={s.response.results} />

              <p className="muted model-line">
                {s.response.model.encoder} · v{s.response.model.version}
              </p>
            </>
          )}
        </section>

        <ModelInfoPanel info={modelInfo} />
      </main>

      <footer className="app__footer muted">
        Animal Visual Search — DINOv2 embeddings + FAISS cosine retrieval.
      </footer>
    </div>
  );
}
