/** Format helpers — numerals use tabular mono in UI. */

export function formatUsd(value, { digits = 0 } = {}) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function formatUsdPerSqft(value) {
  return formatUsd(value, { digits: 0 });
}

export function formatPct(value, { digits = 1 } = {}) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatNumber(value, { digits = 0 } = {}) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function defaultPhases() {
  return [
    {
      name: "launch",
      start_month: 0,
      cash_flow_floor_usd: 0,
      max_units: 20,
      competing_listings: 30,
    },
    {
      name: "wave_two",
      start_month: 6,
      cash_flow_floor_usd: 0,
      max_units: 20,
      competing_listings: 30,
    },
    {
      name: "wave_three",
      start_month: 12,
      cash_flow_floor_usd: 0,
      max_units: 20,
      competing_listings: 30,
    },
    {
      name: "closeout",
      start_month: 18,
      cash_flow_floor_usd: 0,
      max_units: 20,
      competing_listings: 30,
    },
  ];
}

// Operational default for construction-slippage σ (months). Not a macro series
// — no public data measures it — so it stays user-owned even in data mode.
const DEFAULT_COMPLETION_DELAY_MONTHS_SD = 1.0;

/**
 * Resolve a macro snapshot's channel σ to scenario-spec units.
 *
 * The absolute channels (comps_drift, absorption) come straight off the
 * snapshot; competing_listings is stored as a relative swing and multiplied by
 * the plan's competing-listings baseline to become a count σ (mirrors the
 * backend). Falls back to the "base" preset numbers when no snapshot is present.
 */
export function resolveMacroDispersions(macro, competingBaseline = 30) {
  const base = SENTIMENT_PRESETS.base;
  if (!macro || !macro.dispersions) {
    return {
      absorption_log_hazard_sd: base.absorption_log_hazard_sd,
      comps_drift_sd: base.comps_drift_sd,
      competing_listings_sd: base.competing_listings_sd,
      completion_delay_months_sd: DEFAULT_COMPLETION_DELAY_MONTHS_SD,
    };
  }
  const rel = macro.relative_dispersions || {};
  return {
    absorption_log_hazard_sd: macro.dispersions.absorption_log_hazard_sd ?? 0,
    comps_drift_sd: macro.dispersions.comps_drift_sd ?? 0,
    competing_listings_sd: (rel.competing_listings_sd ?? 0) * competingBaseline,
    completion_delay_months_sd: DEFAULT_COMPLETION_DELAY_MONTHS_SD,
  };
}

/** Macro sentiment → scenario standard deviations (assumptions, not estimates). */
export const SENTIMENT_PRESETS = {
  pessimistic: {
    label: "Pessimistic",
    absorption_log_hazard_sd: 0.25,
    comps_drift_sd: 0.05,
    completion_delay_months_sd: 2.0,
    competing_listings_sd: 8.0,
  },
  base: {
    label: "Base",
    absorption_log_hazard_sd: 0.15,
    comps_drift_sd: 0.03,
    completion_delay_months_sd: 1.0,
    competing_listings_sd: 0.0,
  },
  optimistic: {
    label: "Optimistic",
    absorption_log_hazard_sd: 0.08,
    comps_drift_sd: 0.02,
    completion_delay_months_sd: 0.5,
    competing_listings_sd: 0.0,
  },
};
