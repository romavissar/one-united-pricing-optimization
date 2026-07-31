import Plot from "react-plotly.js";
import { useProject } from "../../context/ProjectContext.jsx";
import { formatPct, formatUsdPerSqft } from "../../utils/format.js";
import EmptyState from "../common/EmptyState.jsx";

export default function DemandCurve() {
  const { demandCurve, planResult, selectedUnitId } = useProject();

  if (!selectedUnitId || !planResult) {
    return (
      <EmptyState
        title="Demand curve"
        body="Select a unit in the plan or heatmap. This is where elasticity becomes tangible — P(sell) vs $/sqft."
      />
    );
  }

  if (!demandCurve) {
    return (
      <EmptyState
        title="Loading demand curve…"
        body={`Scoring ${selectedUnitId} across its price ladder.`}
      />
    );
  }

  const chosen = planResult.plan.find(
    (r) => String(r.unit_id) === String(selectedUnitId)
  );

  return (
    <section aria-labelledby="curve-heading">
      <h2 id="curve-heading" className="font-display text-lg font-semibold">
        Demand curve
      </h2>
      <p className="mt-1 text-sm text-muted">
        <span className="font-mono tabular-nums text-ink">
          {demandCurve.unit_id}
        </span>
        {demandCurve.building ? ` · ${demandCurve.building}` : ""} · floor{" "}
        <span className="font-mono tabular-nums">{demandCurve.floor}</span>
        {chosen && (
          <>
            {" "}
            · plan{" "}
            <span className="font-mono tabular-nums text-signal">
              {formatUsdPerSqft(chosen.price_ppsf)}
            </span>{" "}
            at P(sell){" "}
            <span className="font-mono tabular-nums">
              {formatPct(chosen.sale_probability)}
            </span>
          </>
        )}
      </p>
      <Plot
        data={[
          {
            type: "scatter",
            mode: "lines+markers",
            x: demandCurve.prices_ppsf,
            y: demandCurve.probabilities,
            line: { color: "#0F6E5C", width: 2 },
            marker: { color: "#16202B", size: 6 },
            hovertemplate: "$%{x:.0f}/sqft → %{y:.1%}<extra></extra>",
          },
          ...(chosen
            ? [
                {
                  type: "scatter",
                  mode: "markers",
                  x: [chosen.price_ppsf],
                  y: [chosen.sale_probability],
                  marker: {
                    color: "#B4551F",
                    size: 12,
                    symbol: "diamond",
                  },
                  hovertemplate: "Plan: $%{x:.0f} · %{y:.1%}<extra></extra>",
                },
              ]
            : []),
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {
            family: "IBM Plex Sans, sans-serif",
            color: "#6B7684",
            size: 11,
          },
          margin: { l: 48, r: 16, t: 16, b: 48 },
          height: 280,
          showlegend: false,
          xaxis: {
            title: "Asking price ($/sqft)",
            tickprefix: "$",
            gridcolor: "#D8D4CC",
            zeroline: false,
          },
          yaxis: {
            title: `P(sell within ${demandCurve.horizon_days}d)`,
            tickformat: ".0%",
            range: [0, 1],
            gridcolor: "#D8D4CC",
            zeroline: false,
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
        useResizeHandler
        style={{ width: "100%" }}
      />
    </section>
  );
}
