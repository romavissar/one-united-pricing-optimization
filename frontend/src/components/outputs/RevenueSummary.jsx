import { useProject } from "../../context/ProjectContext.jsx";
import { formatPct, formatUsd } from "../../utils/format.js";
import EmptyState from "../common/EmptyState.jsx";

export default function RevenueSummary() {
  const { planResult, distribution } = useProject();

  if (!planResult && !distribution) {
    return (
      <EmptyState
        title="Revenue summary waits on a plan"
        body="Optimize for a point estimate, then Simulate for the distribution and uplift vs pricing at comps."
      />
    );
  }

  const d = distribution?.discounted_usd;
  // The paired per-draw uplift, not the gap between two separately-ranked
  // medians. Both arms share the same sale lottery, so this is the typical gain
  // from repricing and it comes with a band; a difference of medians has none.
  const uplift = distribution?.uplift_vs_baseline_usd || null;

  return (
    <section aria-labelledby="revenue-heading">
      <h2 id="revenue-heading" className="font-display text-lg font-semibold">
        Revenue summary
      </h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Plan objective (discounted)"
          value={formatUsd(planResult?.objective_usd)}
        />
        <Stat
          label="Simulated P50"
          value={d ? formatUsd(d.p50) : "—"}
          accent
        />
        <Stat
          label="90% band (P5–P95)"
          value={
            d
              ? `${formatUsd(d.p5)} – ${formatUsd(d.p95)}`
              : "—"
          }
        />
        <Stat
          label="Uplift vs pricing at comps (paired, P50)"
          value={
            uplift
              ? `${formatUsd(uplift.p50)} · P(beat)=${formatPct(
                  distribution.prob_beats_baseline
                )}`
              : "— (run Simulate)"
          }
        />
      </div>
      {uplift && (
        <p className="mt-3 text-sm text-muted">
          Uplift 90% band{" "}
          <span className="font-mono tabular-nums">
            {formatUsd(uplift.p5)} to {formatUsd(uplift.p95)}
          </span>
          . This is the same release schedule repriced, scored by the model the
          plan was optimized against — so a positive figure says the optimizer
          found its own optimum, not that the model is right about the market.
        </p>
      )}
      {distribution?.variance_decomposition && (
        <p className="mt-3 text-sm text-muted">
          Risk split — parameter{" "}
          <span className="font-mono tabular-nums">
            {formatPct(distribution.variance_decomposition.parameter_share, {
              digits: 0,
            })}
          </span>
          , lumpy sales{" "}
          <span className="font-mono tabular-nums">
            {formatPct(
              distribution.variance_decomposition.idiosyncratic_share,
              { digits: 0 }
            )}
          </span>
          .
        </p>
      )}
    </section>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className="border border-rule p-3">
      <p className="text-xs text-muted">{label}</p>
      <p
        className={[
          "mt-1 font-mono text-sm tabular-nums",
          accent ? "text-signal" : "text-ink",
        ].join(" ")}
      >
        {value}
      </p>
    </div>
  );
}
