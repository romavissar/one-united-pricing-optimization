import { useProject } from "../../context/ProjectContext.jsx";
import { SENTIMENT_PRESETS } from "../../utils/format.js";

const SOURCE_LABEL = {
  fred_bls: "Live · FRED + BLS",
  cache: "Cached · FRED + BLS",
  static_fallback: "Fallback · APIs unavailable",
};

const CHANNELS = [
  {
    key: "comps_drift_sd",
    label: "Comps drift σ",
    hint: "Case-Shiller Miami return volatility",
    step: 0.005,
  },
  {
    key: "absorption_log_hazard_sd",
    label: "Absorption σ",
    hint: "−Δlog(days-on-market) volatility",
    step: 0.01,
  },
  {
    key: "competing_listings_sd",
    label: "Competing σ",
    hint: "active-listing swing × baseline",
    step: 0.5,
  },
  {
    key: "completion_delay_months_sd",
    label: "Delay σ (mo)",
    hint: "operational — you set this",
    step: 0.5,
  },
];

function fmt(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(3);
}

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export default function MacroAssumptions() {
  const {
    macro,
    macroMode,
    derivedDispersions,
    customScenario,
    enterCustomMacro,
    useDataMacro,
    setCustomField,
    applySentimentPreset,
    sentiment,
    nDraws,
    setNDraws,
  } = useProject();

  const isCustom = macroMode === "custom";
  const source = macro?.source;
  const context = macro?.context || {};
  const shown = isCustom ? customScenario || derivedDispersions : derivedDispersions;

  return (
    <section aria-labelledby="macro-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="macro-heading" className="font-display text-lg font-semibold">
          Macro assumptions
        </h2>
        <span
          className={[
            "text-xs font-mono",
            source === "static_fallback" ? "text-warn" : "text-signal",
          ].join(" ")}
        >
          {isCustom ? "Custom · you own these" : SOURCE_LABEL[source] || "Loading…"}
        </span>
      </div>

      <p className="mt-1 text-sm text-muted">
        {isCustom
          ? "You are overriding the model. Each σ below is your assumption, not a measurement."
          : "Derived from market data — the model reads FRED and BLS and sets the simulation width. Only β̂'s standard error comes from the demand fit; construction delay is operational and stays yours."}
      </p>

      {source === "static_fallback" && !isCustom && (
        <p className="mt-2 border border-warn/40 bg-warn/5 px-3 py-2 text-xs text-warn">
          Macro data is unavailable, so these are documented fallback assumptions,
          not measured values. Set FRED_API_KEY, or click “Input custom”.
        </p>
      )}

      {/* Derived / custom σ */}
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {CHANNELS.map((c) => (
          <div key={c.key}>
            <dt className="text-muted">{c.label}</dt>
            {isCustom ? (
              <dd className="mt-1">
                <input
                  type="number"
                  step={c.step}
                  min="0"
                  value={shown?.[c.key] ?? ""}
                  onChange={(e) => setCustomField(c.key, Number(e.target.value))}
                  className="w-full border border-rule bg-paper px-2 py-1 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
                  aria-label={c.label}
                />
              </dd>
            ) : (
              <dd className="font-mono tabular-nums">{fmt(shown?.[c.key])}</dd>
            )}
            <p className="mt-0.5 text-xs text-muted">{c.hint}</p>
          </div>
        ))}
      </dl>

      {/* Mode toggle */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {isCustom ? (
          <button
            type="button"
            onClick={useDataMacro}
            className="border border-signal px-3 py-1.5 text-sm text-signal focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal hover:bg-signal/5"
          >
            Use data-derived
          </button>
        ) : (
          <button
            type="button"
            onClick={enterCustomMacro}
            className="border border-rule bg-paper px-3 py-1.5 text-sm text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal hover:border-ink/40"
          >
            Input custom
          </button>
        )}
        {isCustom &&
          Object.entries(SENTIMENT_PRESETS).map(([key, value]) => (
            <button
              key={key}
              type="button"
              onClick={() => applySentimentPreset(key)}
              className={[
                "border px-3 py-1.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal",
                sentiment === key
                  ? "border-ink bg-ink text-paper"
                  : "border-rule bg-paper text-ink hover:border-ink/40",
              ].join(" ")}
            >
              {value.label}
            </button>
          ))}
      </div>

      {/* Context the model read (both APIs), shown in data mode */}
      {!isCustom && macro && Object.keys(context).length > 0 && (
        <dl className="mt-4 grid grid-cols-2 gap-x-3 gap-y-1 border-t border-rule pt-3 text-xs sm:grid-cols-3">
          {context.home_price_appreciation_yoy_real != null && (
            <div>
              <dt className="text-muted">Real HPA (YoY)</dt>
              <dd className="font-mono tabular-nums">
                {pct(context.home_price_appreciation_yoy_real)}
              </dd>
            </div>
          )}
          {context.mortgage_rate_pct != null && (
            <div>
              <dt className="text-muted">30y mortgage</dt>
              <dd className="font-mono tabular-nums">
                {fmt(context.mortgage_rate_pct)}%
              </dd>
            </div>
          )}
          {context.unemployment_rate_pct != null && (
            <div>
              <dt className="text-muted">Unemployment</dt>
              <dd className="font-mono tabular-nums">
                {fmt(context.unemployment_rate_pct)}%
              </dd>
            </div>
          )}
        </dl>
      )}

      <label className="mt-4 block text-sm">
        <span className="text-muted">Monte Carlo draws</span>
        <input
          type="number"
          min="500"
          max="20000"
          step="500"
          value={nDraws}
          onChange={(e) => setNDraws(Number(e.target.value))}
          className="mt-1 w-40 border border-rule bg-paper px-3 py-2 font-mono tabular-nums focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
        />
      </label>
    </section>
  );
}
