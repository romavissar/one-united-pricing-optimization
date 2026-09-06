import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  fetchConfig,
  fetchCurrentDemand,
  fetchDemandCurve,
  fetchMacro,
  fetchSensitivity,
  fitDemand,
  optimizePlan,
  simulatePlan,
  validateInventory,
} from "../api/client.js";
import {
  SENTIMENT_PRESETS,
  defaultPhases,
  resolveMacroDispersions,
} from "../utils/format.js";

const ProjectContext = createContext(null);

export function ProjectProvider({ children }) {
  const [market] = useState("miami");
  const [config, setConfig] = useState(null);
  const [demandMeta, setDemandMeta] = useState(null);
  const [inventory, setInventory] = useState([]);
  const [validation, setValidation] = useState(null);
  const [phases, setPhases] = useState(defaultPhases);
  const [discountRate, setDiscountRate] = useState(0.12);
  const [presaleLeadMonths, setPresaleLeadMonths] = useState(24);
  const [projectStart, setProjectStart] = useState("2026-01-01");
  const [sentiment, setSentiment] = useState("base");
  const [nDraws, setNDraws] = useState(5000);

  // Macro assumptions are data-driven by default. `macro` is the derived
  // snapshot from GET /api/macro; `macroMode` is "data" (use the snapshot) or
  // "custom" (the user clicked "input custom" and owns the σ). `customScenario`
  // holds the user's overrides, seeded from the derived values so a custom edit
  // starts from data rather than from nothing.
  const [macro, setMacro] = useState(null);
  const [macroMode, setMacroMode] = useState("data");
  const [customScenario, setCustomScenario] = useState(null);

  const [planResult, setPlanResult] = useState(null);
  const [distribution, setDistribution] = useState(null);
  const [sensitivity, setSensitivity] = useState(null);
  const [selectedUnitId, setSelectedUnitId] = useState(null);
  const [demandCurve, setDemandCurve] = useState(null);

  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [simProgress, setSimProgress] = useState(null);
  const [fitNotes, setFitNotes] = useState(null);

  const provenance =
    planResult?.provenance ||
    distribution?.provenance ||
    demandMeta?.provenance ||
    null;

  // The banner is a statement about the numbers currently on screen, so it keys
  // off the provenance of whatever produced them — never off the market config.
  // config.is_calibrated_on_real_data only says a human has cleared the market
  // for real-data fits; a synthetic fit run afterwards still produces
  // illustrative numbers and must still be bannered. Absent provenance counts as
  // not calibrated, which is the safe direction to be wrong in.
  const isCalibrated = provenance?.is_calibrated_on_real_data === true;

  const refreshDemand = useCallback(async () => {
    try {
      const meta = await fetchCurrentDemand(market);
      setDemandMeta(meta);
      return meta;
    } catch {
      setDemandMeta(null);
      return null;
    }
  }, [market]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetchConfig(market);
        if (!cancelled) {
          setConfig(cfg);
          const d = cfg.defaults || {};
          if (d.discount_rate_annual != null) setDiscountRate(d.discount_rate_annual);
          if (d.presale_lead_months != null) setPresaleLeadMonths(d.presale_lead_months);
          if (d.monte_carlo_draws != null) setNDraws(d.monte_carlo_draws);
        }
        await refreshDemand();
        try {
          const snap = await fetchMacro(market);
          if (!cancelled) setMacro(snap);
        } catch {
          // Macro is a default, not a hard dependency: a fetch failure leaves
          // macro null and the panel offers "input custom" instead.
          if (!cancelled) setMacro(null);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [market, refreshDemand]);

  // The competing-listings baseline a relative macro swing is applied against:
  // the median of the phases' assumed competing-listings counts (mirrors the
  // backend). Used only to display and seed the resolved competing σ.
  const competingBaseline = useMemo(() => {
    const values = phases
      .map((p) => Number(p.competing_listings))
      .filter((v) => Number.isFinite(v));
    if (!values.length) return 30;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }, [phases]);

  // The derived σ resolved to spec units (competing_listings × baseline), for
  // display and for seeding a custom edit.
  const derivedDispersions = useMemo(
    () => resolveMacroDispersions(macro, competingBaseline),
    [macro, competingBaseline]
  );

  const enterCustomMacro = useCallback(() => {
    setCustomScenario((prev) => prev || { ...derivedDispersions });
    setMacroMode("custom");
  }, [derivedDispersions]);

  const useDataMacro = useCallback(() => setMacroMode("data"), []);

  const setCustomField = useCallback((field, value) => {
    setCustomScenario((prev) => ({
      ...(prev || {}),
      [field]: value,
    }));
  }, []);

  const applySentimentPreset = useCallback((key) => {
    const preset = SENTIMENT_PRESETS[key];
    if (!preset) return;
    setSentiment(key);
    setCustomScenario({
      absorption_log_hazard_sd: preset.absorption_log_hazard_sd,
      comps_drift_sd: preset.comps_drift_sd,
      completion_delay_months_sd: preset.completion_delay_months_sd,
      competing_listings_sd: preset.competing_listings_sd,
    });
  }, []);

  const basePayload = useCallback(() => {
    // Data mode: send only the operational channel; leaving the three macro
    // channels absent (null) tells the backend to derive them from FRED/BLS —
    // the whole point of the inversion. Custom mode: send the user's explicit σ
    // for every channel, and each supplied value wins server-side.
    const scenario =
      macroMode === "custom" && customScenario
        ? {
            absorption_log_hazard_sd: customScenario.absorption_log_hazard_sd,
            comps_drift_sd: customScenario.comps_drift_sd,
            competing_listings_sd: customScenario.competing_listings_sd,
            completion_delay_months_sd: customScenario.completion_delay_months_sd,
          }
        : {
            completion_delay_months_sd:
              derivedDispersions.completion_delay_months_sd,
          };
    return {
      market,
      project_start: projectStart,
      inventory,
      phases,
      discount_rate_annual: discountRate,
      presale_lead_months: presaleLeadMonths,
      scenario,
    };
  }, [
    market,
    projectStart,
    inventory,
    phases,
    discountRate,
    presaleLeadMonths,
    macroMode,
    customScenario,
    derivedDispersions,
  ]);

  const runFit = useCallback(async () => {
    setBusy("fit");
    setError(null);
    setFitNotes(null);
    try {
      const result = await fitDemand({
        dataset: "synthetic",
        market,
        n_listings: 4000,
        seed: 13,
        controls: true,
        building_fe: false,
      });
      setFitNotes(result);
      await refreshDemand();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  }, [market, refreshDemand]);

  const runValidate = useCallback(
    async (rows) => {
      setBusy("validate");
      setError(null);
      setPlanResult(null);
      setDistribution(null);
      setSensitivity(null);
      setDemandCurve(null);
      try {
        const result = await validateInventory(market, rows);
        setValidation(result);
        setInventory(result.units || []);
        if (!result.is_valid) {
          setError(
            `Inventory has ${result.errors.length} row error(s). Fix them before optimizing.`
          );
        }
        return result;
      } catch (err) {
        setError(err.message);
        setValidation(null);
        setInventory([]);
        throw err;
      } finally {
        setBusy(null);
      }
    },
    [market]
  );

  const runOptimize = useCallback(async () => {
    if (!inventory.length) {
      setError("Upload a valid inventory first.");
      return;
    }
    setBusy("optimize");
    setError(null);
    setDistribution(null);
    setSensitivity(null);
    try {
      const result = await optimizePlan({
        ...basePayload(),
        constraints: {},
      });
      setPlanResult(result);
      if (result.plan?.length) {
        setSelectedUnitId(result.plan[0].unit_id);
      }
    } catch (err) {
      setError(err.message);
      setPlanResult(null);
    } finally {
      setBusy(null);
    }
  }, [inventory, basePayload]);

  const runSimulate = useCallback(async () => {
    if (!planResult?.plan?.length) {
      setError("Optimize a plan before simulating.");
      return;
    }
    setBusy("simulate");
    setError(null);
    setSimProgress(null);
    try {
      const result = await simulatePlan(
        {
          ...basePayload(),
          plan: planResult.plan,
          n_draws: nDraws,
          seed: 13,
        },
        { onProgress: setSimProgress }
      );
      setDistribution(result);
    } catch (err) {
      setError(err.message);
      setDistribution(null);
    } finally {
      setBusy(null);
      setSimProgress(null);
    }
  }, [planResult, basePayload, nDraws]);

  const runSensitivity = useCallback(async () => {
    if (!planResult?.plan?.length) {
      setError("Optimize a plan before running sensitivity.");
      return;
    }
    setBusy("sensitivity");
    setError(null);
    try {
      const result = await fetchSensitivity({
        ...basePayload(),
        plan: planResult.plan,
        include_shadow_prices: true,
      });
      setSensitivity(result);
    } catch (err) {
      setError(err.message);
      setSensitivity(null);
    } finally {
      setBusy(null);
    }
  }, [planResult, basePayload]);

  const loadDemandCurve = useCallback(
    async (unitId) => {
      if (!unitId || !inventory.length || !phases.length) return;
      setSelectedUnitId(unitId);
      try {
        const curve = await fetchDemandCurve({
          market,
          project_start: projectStart,
          unit_id: unitId,
          inventory,
          phases,
          discount_rate_annual: discountRate,
          presale_lead_months: presaleLeadMonths,
          phase_index: 0,
        });
        setDemandCurve(curve);
      } catch (err) {
        setDemandCurve(null);
        setError(err.message);
      }
    },
    [inventory, phases, market, projectStart, discountRate, presaleLeadMonths]
  );

  useEffect(() => {
    if (selectedUnitId && planResult?.plan?.length) {
      loadDemandCurve(selectedUnitId);
    }
    // Intentionally depend on selected unit + plan identity, not the callback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUnitId, planResult]);

  const value = useMemo(
    () => ({
      market,
      config,
      demandMeta,
      inventory,
      validation,
      phases,
      setPhases,
      discountRate,
      setDiscountRate,
      presaleLeadMonths,
      setPresaleLeadMonths,
      projectStart,
      setProjectStart,
      sentiment,
      setSentiment,
      nDraws,
      setNDraws,
      macro,
      macroMode,
      derivedDispersions,
      competingBaseline,
      customScenario,
      enterCustomMacro,
      useDataMacro,
      setCustomField,
      applySentimentPreset,
      planResult,
      distribution,
      sensitivity,
      selectedUnitId,
      setSelectedUnitId,
      demandCurve,
      busy,
      error,
      setError,
      simProgress,
      fitNotes,
      provenance,
      isCalibrated,
      runFit,
      runValidate,
      runOptimize,
      runSimulate,
      runSensitivity,
      loadDemandCurve,
      refreshDemand,
    }),
    [
      market,
      config,
      demandMeta,
      inventory,
      validation,
      phases,
      discountRate,
      presaleLeadMonths,
      projectStart,
      sentiment,
      nDraws,
      macro,
      macroMode,
      derivedDispersions,
      competingBaseline,
      customScenario,
      enterCustomMacro,
      useDataMacro,
      setCustomField,
      applySentimentPreset,
      planResult,
      distribution,
      sensitivity,
      selectedUnitId,
      demandCurve,
      busy,
      error,
      simProgress,
      fitNotes,
      provenance,
      isCalibrated,
      runFit,
      runValidate,
      runOptimize,
      runSimulate,
      runSensitivity,
      loadDemandCurve,
      refreshDemand,
    ]
  );

  return (
    <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>
  );
}

export function useProject() {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error("useProject must be used within ProjectProvider");
  return ctx;
}
