import type { ModelInfoResponse } from "../types";

interface Props {
  info: ModelInfoResponse | null;
}

export function ModelInfoPanel({ info }: Props) {
  if (!info) {
    return (
      <aside className="info-panel">
        <h2 className="info-panel__title">Model</h2>
        <p className="muted">Loading model info…</p>
      </aside>
    );
  }

  const rows: Array<[string, string | number]> = [
    ["Encoder", info.encoder],
    ["Version", info.version],
    ["Classes", info.num_classes],
    ["Gallery", info.gallery_size],
    ["Embedding", `${info.embedding_dim}-D`],
    ["Retrieval", `${info.metric} · ${info.index_type}`],
    ["Input size", `${info.image_size}px`],
  ];

  return (
    <aside className="info-panel">
      <h2 className="info-panel__title">Model</h2>
      <dl className="info-panel__list">
        {rows.map(([k, v]) => (
          <div className="info-panel__row" key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}
