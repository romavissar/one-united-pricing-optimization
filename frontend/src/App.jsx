import CalibrationBanner from "./components/common/CalibrationBanner.jsx";
import ErrorBoundary from "./components/common/ErrorBoundary.jsx";
import ProgressBar from "./components/common/ProgressBar.jsx";
import ActionButton from "./components/common/ActionButton.jsx";
import InventoryUpload from "./components/inputs/InventoryUpload.jsx";
import Constraints from "./components/inputs/Constraints.jsx";
import MacroAssumptions from "./components/inputs/MacroAssumptions.jsx";
import ReleasePlanTable from "./components/outputs/ReleasePlanTable.jsx";
import RevenueSummary from "./components/outputs/RevenueSummary.jsx";
import RiskTable from "./components/outputs/RiskTable.jsx";
import BuildingHeatmap from "./components/charts/BuildingHeatmap.jsx";
import RevenueDistribution from "./components/charts/RevenueDistribution.jsx";
import PhaseTimeline from "./components/charts/PhaseTimeline.jsx";
import SensitivityTornado from "./components/charts/SensitivityTornado.jsx";
import DemandCurve from "./components/charts/DemandCurve.jsx";
import { ProjectProvider, useProject } from "./context/ProjectContext.jsx";
import { formatNumber } from "./utils/format.js";

function Workspace() {
  const {
    demandMeta,
    validation,
    planResult,
    distribution,
    sensitivity,
    busy,
    error,
    setError,
    simProgress,
    fitNotes,
    isCalibrated,
    runFit,
    runOptimize,
    runSimulate,
    runSensitivity,
  } = useProject();

  const beta = demandMeta?.beta_price || fitNotes?.beta_price;
  const hasModel = Boolean(demandMeta || fitNotes);
  const canOptimize = Boolean(validation?.is_valid && hasModel);
  const canSimulate = Boolean(planResult?.plan?.length);

  return (
    <div className="min-h-screen bg-paper text-ink">
      <CalibrationBanner visible={!isCalibrated} />

      <header className="border-b border-rule px-4 py-8 md:px-10">
        <p className="font-display text-2xl font-semibold tracking-tight md:text-3xl">
          Pricing & Release Optimizer
        </p>
        <p className="mt-2 max-w-2xl text-sm text-muted md:text-base">
          Price and release luxury inventory against a fitted own-price
          elasticity — then see the revenue distribution, not a point estimate
          dressed as certainty.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <ActionButton
            action="fit"
            busy={busy}
            done={hasModel}
            onClick={runFit}
          />
          <ActionButton
            action="optimize"
            busy={busy}
            done={Boolean(planResult)}
            onClick={runOptimize}
            disabled={!canOptimize}
          />
          <ActionButton
            action="simulate"
            busy={busy}
            done={Boolean(distribution)}
            onClick={runSimulate}
            disabled={!canSimulate}
          />
          <ActionButton
            action="sensitivity"
            busy={busy}
            done={Boolean(sensitivity)}
            onClick={runSensitivity}
            disabled={!canSimulate}
          />
        </div>

        {beta && (
          <p className="mt-4 text-sm text-muted">
            Active β_price{" "}
            <span className="font-mono tabular-nums text-ink">
              {Number(beta.value).toFixed(3)}
            </span>
            {beta.std_error != null && (
              <>
                {" "}
                ±{" "}
                <span className="font-mono tabular-nums">
                  {Number(beta.std_error).toFixed(3)}
                </span>
              </>
            )}
            {demandMeta?.provenance?.fitted_on && (
              <>
                {" "}
                · fitted on{" "}
                <span className="text-ink">
                  {demandMeta.provenance.fitted_on}
                </span>
              </>
            )}
          </p>
        )}

        {busy === "simulate" && simProgress && (
          <div className="mt-4">
            <ProgressBar
              drawn={simProgress.drawn || 0}
              total={simProgress.n_draws || 1}
              label="Monte Carlo draws"
            />
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="mt-4 max-w-2xl border border-warn/40 bg-warn/10 p-3 text-sm"
          >
            <p className="font-medium text-warn">{error}</p>
            <button
              type="button"
              onClick={() => setError(null)}
              className="mt-2 text-xs text-muted underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-signal"
            >
              Dismiss message
            </button>
          </div>
        )}
      </header>

      <main className="mx-auto grid max-w-[90rem] gap-12 px-4 py-10 md:px-10">
        <div className="grid gap-12 lg:grid-cols-2">
          <InventoryUpload />
          <div className="space-y-12">
            <Constraints />
            <MacroAssumptions />
          </div>
        </div>

        <ErrorBoundary>
          <BuildingHeatmap />
        </ErrorBoundary>

        <ErrorBoundary>
          <ReleasePlanTable />
        </ErrorBoundary>

        <div className="grid gap-12 lg:grid-cols-2">
          <ErrorBoundary>
            <RevenueSummary />
          </ErrorBoundary>
          <ErrorBoundary>
            <RiskTable />
          </ErrorBoundary>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          <ErrorBoundary>
            <RevenueDistribution />
          </ErrorBoundary>
          <ErrorBoundary>
            <PhaseTimeline />
          </ErrorBoundary>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          <ErrorBoundary>
            <SensitivityTornado />
          </ErrorBoundary>
          <ErrorBoundary>
            <DemandCurve />
          </ErrorBoundary>
        </div>

        {planResult?.caveats?.length > 0 && (
          <section>
            <h2 className="font-display text-lg font-semibold">Caveats</h2>
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">
              {planResult.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
              {distribution?.caveats?.slice(0, 5).map((c, i) => (
                <li key={`d-${i}`}>{c}</li>
              ))}
            </ul>
            {planResult.units_released != null && (
              <p className="mt-3 font-mono text-xs tabular-nums text-muted">
                Released {formatNumber(planResult.units_released)} units in{" "}
                {planResult.solve_seconds?.toFixed?.(2)}s
              </p>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ProjectProvider>
      <Workspace />
    </ProjectProvider>
  );
}
