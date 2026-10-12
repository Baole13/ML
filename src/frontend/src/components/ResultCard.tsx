import { useState } from "react";
import { resolveUrl } from "../services/api";
import type { SearchResult } from "../types";

interface Props {
  result: SearchResult;
  rank: number;
}

export function ResultCard({ result, rank }: Props) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  return (
    <figure className="result-card">
      <div className="result-card__media">
        {failed ? (
          <div className="result-card__fallback">🖼️</div>
        ) : (
          <img
            src={resolveUrl(result.image_url)}
            alt={`${result.species} gallery image ${result.id}`}
            loading="lazy"
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
            className={loaded ? "is-loaded" : ""}
          />
        )}
        <span className="result-card__rank">#{rank}</span>
      </div>
      <figcaption className="result-card__caption">
        <span className="result-card__species">{result.species}</span>
        <span className="result-card__sim">{result.similarity.toFixed(3)}</span>
      </figcaption>
    </figure>
  );
}
