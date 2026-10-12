interface Props {
  value: number;
  disabled: boolean;
  onChange: (k: number) => void;
}

const MAX_K = 20;

export function KSelector({ value, disabled, onChange }: Props) {
  return (
    <div className="kselector">
      <label htmlFor="k-slider">
        Similar images <span className="kselector__value">K = {value}</span>
      </label>
      <input
        id="k-slider"
        type="range"
        min={1}
        max={MAX_K}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
