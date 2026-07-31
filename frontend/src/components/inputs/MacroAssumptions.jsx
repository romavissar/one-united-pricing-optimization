import { useProject } from "../../context/ProjectContext.jsx";
import { SENTIMENT_PRESETS } from "../../utils/format.js";

export default function MacroAssumptions() {
  const { sentiment, setSentiment, nDraws, setNDraws } = useProject();
  const preset = SENTIMENT_PRESETS[sentiment];

  return (
    <section aria-labelledby="macro-heading">
      <h2 id="macro-heading" className="font-display text-lg font-semibold">
        Macro assumptions
      </h2>
      <p className="mt-1 text-sm text-muted">
        These set the width of the simulation — user beliefs about the future,
        not fitted parameters. Only β̂&apos;s standard error comes from the demand
        model.
      </p>

      <fieldset className="mt-4">
        <legend className="text-sm text-muted">Demand sentiment</legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {Object.entries(SENTIMENT_PRESETS).map(([key, value]) => (
            <button
              key={key}
              type="button"
              onClick={() => setSentiment(key)}
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
      </fieldset>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-muted">Absorption σ</dt>
          <dd className="font-mono tabular-nums">
            {preset.absorption_log_hazard_sd}
          </dd>
        </div>
        <div>
          <dt className="text-muted">Comps drift σ</dt>
          <dd className="font-mono tabular-nums">{preset.comps_drift_sd}</dd>
        </div>
        <div>
          <dt className="text-muted">Delay σ (mo)</dt>
          <dd className="font-mono tabular-nums">
            {preset.completion_delay_months_sd}
          </dd>
        </div>
        <div>
          <dt className="text-muted">Competing σ</dt>
          <dd className="font-mono tabular-nums">
            {preset.competing_listings_sd}
          </dd>
        </div>
      </dl>

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
