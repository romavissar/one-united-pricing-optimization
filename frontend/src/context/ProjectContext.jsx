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
  fetchSensitivity,
  fitDemand,
  optimizePlan,
  simulatePlan,
  validateInventory,
} from "../api/client.js";
import { SENTIMENT_PRESETS, defaultPhases } from "../utils/format.js";

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
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [market, refreshDemand]);

  const basePayload = useCallback(() => {
    const preset = SENTIMENT_PRESETS[sentiment] || SENTIMENT_PRESETS.base;
    return {
      market,
      project_start: projectStart,
      inventory,
      phases,
      discount_rate_annual: discountRate,
      presale_lead_months: presaleLeadMonths,
      scenario: {
        absorption_log_hazard_sd: preset.absorption_log_hazard_sd,
        comps_drift_sd: preset.comps_drift_sd,
        completion_delay_months_sd: preset.completion_delay_months_sd,
        competing_listings_sd: preset.competing_listings_sd,
      },
    };
  }, [
    market,
    projectStart,
    inventory,
    phases,
    discountRate,
    presaleLeadMonths,
    sentiment,
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
