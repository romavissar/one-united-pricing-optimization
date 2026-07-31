export default function Spinner({ label = "Working…" }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted">
      <span
        className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-rule border-t-signal motion-reduce:animate-none"
        aria-hidden
      />
      {label}
    </span>
  );
}
