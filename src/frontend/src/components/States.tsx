interface LoadingProps {
  message?: string;
}

export function LoadingState({ message = "Searching the gallery…" }: LoadingProps) {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
}

interface ErrorProps {
  message: string;
  onRetry?: () => void;
  onReset?: () => void;
}

export function ErrorState({ message, onRetry, onReset }: ErrorProps) {
  return (
    <div className="state state--error" role="alert">
      <p className="state__title">⚠️ {message}</p>
      <div className="state__actions">
        {onRetry && (
          <button type="button" className="btn" onClick={onRetry}>
            Try again
          </button>
        )}
        {onReset && (
          <button type="button" className="btn btn--ghost" onClick={onReset}>
            Start over
          </button>
        )}
      </div>
    </div>
  );
}

interface EmptyProps {
  message?: string;
}

export function EmptyState({ message = "Upload an image to begin." }: EmptyProps) {
  return (
    <div className="state state--empty">
      <p className="muted">{message}</p>
    </div>
  );
}
