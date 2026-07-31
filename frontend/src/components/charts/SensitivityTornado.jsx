import Plot from "react-plotly.js";
import { useProject } from "../../context/ProjectContext.jsx";
import EmptyState from "../common/EmptyState.jsx";

export default function SensitivityTornado() {
  const { sensitivity } = useProject();
  const bars = sensitivity?.tornado?.bars || [];

  if (!bars.length) {
    return (
      <EmptyState
        title="Sensitivity tornado"
        body="Run sensitivity to see ±1σ swings on discounted expected revenue."
      />
    );
  }

  // The largest bar is routinely an assumption, not evidence: a user-chosen
  // absorption sigma can swing revenue several times harder than beta_price's
  // fitted standard error. Labelling that only in the caption leaves the chart
  // itself saying the opposite, so the distinction goes on the axis and into
  // every hover.
  const labels = bars
    .map((b) => `${b.channel}${b.fitted ? "" : " (assumed)"}`)
    .reverse();
  const low = bars.map((b) => b.low_delta_usd).reverse();
  const high = bars.map((b) => b.high_delta_usd).reverse();
  const fitted = bars.map((b) => b.fitted).reverse();
  // Assumed channels are drawn hollow so a fitted bar reads as the solid one.
  const barStyle = (color) => ({
    color: fitted.map((f) => (f ? color : "rgba(0,0,0,0)")),
    line: { color, width: fitted.map((f) => (f ? 0 : 1.5)) },
  });

  return (
    <section aria-labelledby="tornado-heading">
      <h2 id="tornado-heading" className="font-display text-lg font-semibold">
        Sensitivity tornado
      </h2>
      <p className="mt-1 text-sm text-muted">
        One channel at a time, with the plan held fixed — no re-optimization.
        Solid bars are fitted evidence; hollow bars marked{" "}
        <span className="italic">assumed</span> use a standard deviation you
        chose, so their length says as much about that choice as about the
        project.
      </p>
      <Plot
        data={[
          {
            type: "bar",
            orientation: "h",
            name: "−1σ",
            y: labels,
            x: low,
            marker: barStyle("#B4551F"),
            hovertemplate: "%{y} −1σ: $%{x:,.0f}<extra></extra>",
          },
          {
            type: "bar",
            orientation: "h",
            name: "+1σ",
            y: labels,
            x: high,
            marker: barStyle("#0F6E5C"),
            hovertemplate: "%{y} +1σ: $%{x:,.0f}<extra></extra>",
          },
        ]}
        layout={{
          barmode: "overlay",
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {
            family: "IBM Plex Sans, sans-serif",
            color: "#6B7684",
            size: 11,
          },
          margin: { l: 140, r: 24, t: 16, b: 40 },
          height: 280,
          legend: { orientation: "h", y: 1.15 },
          xaxis: {
            title: "Δ discounted expected revenue (USD)",
            tickprefix: "$",
            separatethousands: true,
            gridcolor: "#D8D4CC",
            zeroline: true,
            zerolinecolor: "#16202B",
          },
          yaxis: { automargin: true },
        }}
        config={{ displayModeBar: false, responsive: true }}
        className="w-full"
        useResizeHandler
        style={{ width: "100%" }}
      />
    </section>
  );
}
