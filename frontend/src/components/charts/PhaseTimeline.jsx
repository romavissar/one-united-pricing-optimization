import Plot from "react-plotly.js";
import { useProject } from "../../context/ProjectContext.jsx";
import EmptyState from "../common/EmptyState.jsx";

export default function PhaseTimeline() {
  const { planResult } = useProject();
  const phases = planResult?.per_phase || [];

  if (!phases.length) {
    return (
      <EmptyState
        title="Phase timeline"
        body="Optimize to see units and expected revenue by release wave."
      />
    );
  }

  return (
    <section aria-labelledby="timeline-heading">
      <h2 id="timeline-heading" className="font-display text-lg font-semibold">
        Phase timeline
      </h2>
      <Plot
        data={[
          {
            type: "bar",
            name: "Units",
            x: phases.map((p) => p.name),
            y: phases.map((p) => p.units_released),
            marker: { color: "#16202B" },
            yaxis: "y",
            hovertemplate: "%{x}: %{y} units<extra></extra>",
          },
          {
            type: "scatter",
            mode: "lines+markers",
            name: "E[revenue]",
            x: phases.map((p) => p.name),
            y: phases.map((p) => p.expected_revenue_usd),
            marker: { color: "#0F6E5C", size: 8 },
            line: { color: "#0F6E5C", width: 2 },
            yaxis: "y2",
            hovertemplate: "%{x}: $%{y:,.0f}<extra></extra>",
          },
        ]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {
            family: "IBM Plex Sans, sans-serif",
            color: "#6B7684",
            size: 11,
          },
          margin: { l: 48, r: 56, t: 16, b: 40 },
          height: 260,
          legend: { orientation: "h", y: 1.15 },
          xaxis: { gridcolor: "#D8D4CC" },
          yaxis: {
            title: "Units",
            gridcolor: "#D8D4CC",
            zeroline: false,
          },
          yaxis2: {
            title: "USD",
            overlaying: "y",
            side: "right",
            tickprefix: "$",
            separatethousands: true,
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
