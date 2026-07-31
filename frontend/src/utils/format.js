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
