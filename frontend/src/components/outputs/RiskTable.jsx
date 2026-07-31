import { useProject } from "../../context/ProjectContext.jsx";
import { formatPct, formatUsd } from "../../utils/format.js";
import EmptyState from "../common/EmptyState.jsx";

export default function RiskTable() {
  const { distribution } = useProject();
  const breaches = distribution?.phase_breach || [];

  if (!distribution) {
    return (
      <EmptyState
        title="Cash-flow risk"
        body="After Simulate, each phase shows how often realized revenue misses its floor."
      />
    );
  }

  if (!breaches.some((p) => p.cash_flow_floor_usd > 0)) {
    return (
      <p className="text-sm text-muted">
        No cash-flow floors were set — nothing to breach.
      </p>
    );
  }

  return (
    <section aria-labelledby="risk-heading">
      <h2 id="risk-heading" className="font-display text-lg font-semibold">
        Cash-flow breach risk
      </h2>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[32rem] border-collapse text-sm">
          <thead>
            <tr className="border-b border-rule text-left text-muted">
              <th className="py-2 pr-3 font-medium">Phase</th>
              <th className="py-2 pr-3 font-medium">Floor</th>
              <th className="py-2 pr-3 font-medium">P5 revenue</th>
              <th className="py-2 pr-3 font-medium">P(breach)</th>
              <th className="py-2 font-medium">Buffered floor</th>
            </tr>
          </thead>
          <tbody>
            {breaches.map((p) => (
              <tr key={p.index} className="border-b border-rule/70">
                <td className="py-2 pr-3">{p.name}</td>
                <td className="py-2 pr-3 font-mono tabular-nums">
                  {formatUsd(p.cash_flow_floor_usd)}
                </td>
                <td className="py-2 pr-3 font-mono tabular-nums">
                  {formatUsd(p.p5_revenue_usd)}
                </td>
                <td
                  className={[
                    "py-2 pr-3 font-mono tabular-nums",
                    p.is_alarming ? "text-warn" : "",
                  ].join(" ")}
                >
                  {formatPct(p.breach_probability)}
                </td>
                <td className="py-2 font-mono tabular-nums">
                  {p.buffered_floor_usd != null
                    ? formatUsd(p.buffered_floor_usd)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
