import { useCallback, useRef, useState } from "react";

interface Props {
  previewUrl: string | null;
  fileName: string | null;
  disabled: boolean;
  onSelect: (file: File | null) => void;
}

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp";

export function ImageUploader({ previewUrl, fileName, disabled, onSelect }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onSelect(files[0]);
    },
    [onSelect],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      handleFiles(e.dataTransfer.files);
    },
    [disabled, handleFiles],
  );

  return (
    <div className="uploader">
      <div
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
      >
        {previewUrl ? (
          <img className="preview" src={previewUrl} alt="Selected animal" />
        ) : (
          <div className="dropzone__hint">
            <span className="dropzone__icon">🐾</span>
            <p>
              <strong>Drop an animal image here</strong>
            </p>
            <p className="muted">or click to browse (JPG / PNG / WebP, ≤ 10 MB)</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="visually-hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {fileName && (
        <div className="uploader__meta">
          <span className="filename" title={fileName}>
            {fileName}
          </span>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            disabled={disabled}
            onClick={() => {
              onSelect(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
