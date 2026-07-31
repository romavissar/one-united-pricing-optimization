export default function ProgressBar({ drawn = 0, total = 1, label }) {
  const pct = total > 0 ? Math.min(100, Math.round((drawn / total) * 100)) : 0;
  return (
    <div className="w-full max-w-md">
      {label && <p className="mb-1 text-xs text-muted">{label}</p>}
      <div className="h-1.5 w-full overflow-hidden bg-rule/60">
        <div
          className="h-full bg-signal transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 font-mono text-xs tabular-nums text-muted">
        {drawn.toLocaleString()} / {total.toLocaleString()}
      </p>
    </div>
  );
}
