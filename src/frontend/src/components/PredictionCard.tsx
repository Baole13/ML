import type { Prediction } from "../types";

interface Props {
  prediction: Prediction;
}

export function PredictionCard({ prediction }: Props) {
  const pct = Math.round(prediction.confidence * 1000) / 10; // one decimal
  // Colour the bar by confidence band.
  const band = prediction.confidence >= 0.8 ? "high" : prediction.confidence >= 0.5 ? "mid" : "low";

  return (
    <div className="prediction">
      <div className="prediction__label">Predicted species</div>
      <div className="prediction__species">{prediction.species}</div>
      <div className="confidence">
        <div className="confidence__track">
          <div
            className={`confidence__fill confidence__fill--${band}`}
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
        <div className="confidence__value">{pct.toFixed(1)}% confidence</div>
      </div>
    </div>
  );
}
