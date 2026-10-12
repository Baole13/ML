import type { SearchResult } from "../types";
import { ResultCard } from "./ResultCard";

interface Props {
  results: SearchResult[];
}

export function ResultGrid({ results }: Props) {
  if (results.length === 0) {
    return <p className="muted">No similar images found.</p>;
  }
  return (
    <div className="result-grid">
      {results.map((r, i) => (
        <ResultCard key={`${r.id}-${i}`} result={r} rank={i + 1} />
      ))}
    </div>
  );
}
