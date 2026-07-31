import Plot from "react-plotly.js";
import { useProject } from "../../context/ProjectContext.jsx";
import EmptyState from "../common/EmptyState.jsx";

const layoutBase = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "IBM Plex Sans, sans-serif", color: "#6B7684", size: 11 },
  margin: { l: 48, r: 16, t: 24, b: 40 },
  showlegend: false,
};

export default function RevenueDistribution() {
  const { distribution } = useProject();
  const d = distribution?.discounted_usd;

  if (!d) {
    return (
      <EmptyState
        title="Revenue distribution"
        body="Run Simulate to see the histogram of discounted revenue with percentile markers."
      />
    );
  }

  // The API returns percentiles, not the draw sample, so there is no histogram
  // to draw. What follows is an interpolation through seven reported points
  // with assumed heights — the percentile MARKERS are measured, the curve
  // between them is not. Skew, bimodality, and tail shape are invisible here by
  // construction, which is why the caption below says so on screen rather than
  // only in the README.
  const xs = [d.minimum, d.p5, d.p25, d.p50, d.p75, d.p95, d.maximum];
  const heights = [0.15, 0.35, 0.7, 1, 0.7, 0.35, 0.15];

  const markers = [
    { x: d.p5, label: "P5" },
    { x: d.p50, label: "P50" },
    { x: d.p95, label: "P95" },
    { x: d.cvar5, label: "CVaR@5%" },
  ];

  return (
    <section aria-labelledby="dist-heading">
      <h2 id="dist-heading" className="font-display text-lg font-semibold">
        Revenue distribution
      </h2>
      <p className="mt-1 text-sm text-muted">
        Discounted realized revenue across{" "}
        <span className="font-mono tabular-nums">
          {distribution.n_draws.toLocaleString()}
        </span>{" "}
        draws.
      </p>
      <p className="mt-1 text-xs text-warn">
        Shape is illustrative. The marked percentiles are measured; the curve
        between them is interpolated, because the API returns percentiles rather
        than the draw sample. Read the numbers, not the silhouette.
      </p>
      <Plot
        data={[
          {
            type: "scatter",
            mode: "lines",
            x: xs,
            y: heights,
            fill: "tozeroy",
            line: { color: "#0F6E5C", width: 2 },
            fillcolor: "rgba(15,110,92,0.18)",
            hovertemplate: "$%{x:,.0f}<extra></extra>",
          },
          ...markers.map((m) => ({
            type: "scatter",
            mode: "lines",
            x: [m.x, m.x],
            y: [0, 1.05],
            line: {
              color: m.label === "CVaR@5%" ? "#B4551F" : "#16202B",
              width: 1,
              dash: m.label === "P50" ? "solid" : "dot",
            },
            hovertemplate: `${m.label}: $%{x:,.0f}<extra></extra>`,
          })),
        ]}
        layout={{
          ...layoutBase,
          height: 260,
          xaxis: {
            title: "Discounted revenue (USD)",
            tickprefix: "$",
            separatethousands: true,
            gridcolor: "#D8D4CC",
            zeroline: false,
          },
          yaxis: { visible: false, range: [0, 1.15] },
          annotations: markers.map((m, i) => ({
            x: m.x,
            y: 1.08 - (i % 2) * 0.08,
            text: m.label,
            showarrow: false,
            font: { size: 10, color: "#6B7684" },
          })),
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
        useResizeHandler
        style={{ width: "100%" }}
      />
    </section>
  );
}
