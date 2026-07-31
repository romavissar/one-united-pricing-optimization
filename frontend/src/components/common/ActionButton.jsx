const LABELS = {
  idle: {
    fit: "Fit synthetic demand",
    optimize: "Optimize",
    simulate: "Simulate",
    sensitivity: "Run sensitivity",
  },
  busy: {
    fit: "Fitting…",
    optimize: "Optimizing…",
    simulate: "Simulating…",
    sensitivity: "Running…",
  },
  done: {
    fit: "Fitted",
    optimize: "Optimized",
    simulate: "Simulated",
    sensitivity: "Sensitivity ready",
  },
};

export default function ActionButton({
  action,
  busy,
  done,
  onClick,
  disabled,
  className = "",
}) {
  const isBusy = busy === action;
  let label = LABELS.idle[action] || action;
  if (isBusy) label = LABELS.busy[action];
  else if (done) label = LABELS.done[action];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || Boolean(busy)}
      className={[
        "inline-flex items-center justify-center border px-4 py-2 text-sm font-medium transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal",
        "disabled:cursor-not-allowed disabled:opacity-50",
        done && !isBusy
          ? "border-signal bg-signal/10 text-signal"
          : "border-ink bg-ink text-paper hover:bg-ink/90",
        className,
      ].join(" ")}
    >
      {label}
    </button>
  );
}
