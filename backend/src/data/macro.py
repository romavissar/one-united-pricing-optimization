"""Derive the Monte-Carlo scenario dispersions from macro data, not from a user.

`simulation/scenarios.py` documents five channels that perturb a release plan.
Exactly one — `beta_price` — is fitted from the listings. The other four were,
historically, the user's beliefs about the future, typed into a form. This
module replaces the *typing* for the three that a public macro series measures
directly, and leaves the fitted channel and the one genuinely operational
channel (`completion_delay_months`) alone.

The mappings are deliberately transparent, because `AGENTS.md` §1 forbids a
hardcoded estimated parameter and a fitted-looking coefficient smuggled in here
would be exactly that. None of the three is a fitted elasticity:

| channel | series | mapping | why it is not an invented coefficient |
|---|---|---|---|
| `comps_drift` | Case-Shiller FL-Miami (`MIXRSA`, FRED) | SD of log-returns, scaled to the horizon | `comps_drift` *is* a proportional move in the comp median; the index *is* that move. No coefficient. |
| `absorption` | Median days-on-market, Miami CBSA (`MEDDAYONMAR33100`, FRED) | SD of `−Δlog(DOM)` over the horizon | An accounting identity: hazard ≈ 1 / time-to-sale, so `Δlog(hazard) = −Δlog(DOM)`. Not an estimated rate→hazard sensitivity. |
| `competing_listings` | Active listing count, Miami CBSA (`ACTLISCOU33100`, FRED) | SD of the count change over the horizon | The covariate is a listing *count*; the series is a listing *count*. Same units. |

The cross-channel **correlations** among these three are estimated from the
aligned history rather than asserted (the old `DEFAULT_CORRELATIONS` held them
as sign priors). `beta_price`'s correlation to them stays a documented prior,
because `beta_price` is not a macro observable and nothing here measures how it
co-moves with days-on-market.

BLS enters as context the snapshot reports and reasons about, never as a
silent unit conversion of a dispersion: Miami-area CPI (`CUURS35CSA0`) turns the
nominal Case-Shiller drift into a real appreciation figure, and Miami-MSA
unemployment (`LAUMT123310000000003`) is the labour-market demand signal a
developer reads alongside absorption. `comps_drift_sd` stays **nominal**, because
the comps the optimizer divides a candidate price by are nominal $/sqft;
deflating the dispersion would be a units error.

Provenance is first-class. Every snapshot says where its numbers came from —
`fred_bls`, `cache`, or `static_fallback` — so a figure fetched live is never
confused with the documented fallback used when the APIs are unreachable. That
fallback is the only place round assumption magnitudes appear, and they are
labelled as fallback, not as measurement.

Units: `comps_drift_sd` decimal fraction; `absorption_log_hazard_sd` log points;
`competing_listings_sd` a count; correlations dimensionless in `[-1, 1]`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MarketConfig, validate_market_id
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_HTTP_TIMEOUT_SECONDS = 20.0
_TRADING_DAYS = 365.25  # calendar; horizons here are calendar days


# ── Documented fallback ────────────────────────────────────────────────────
# Used ONLY when the APIs are unreachable and no cache exists. These are the
# historical "base" assumptions from the frontend sentiment preset. They are the
# one place round magnitudes live, and the snapshot labels any result built on
# them `static_fallback` so they are never mistaken for a measurement. They are
# assumptions of last resort, not estimates.
_FALLBACK_DISPERSIONS: dict[str, float] = {
    "comps_drift_sd": 0.03,
    "absorption_log_hazard_sd": 0.15,
}
# Relative (dimensionless) fallback swing for competing_listings; 0.0 keeps the
# channel off, matching the historical "base" preset.
_FALLBACK_RELATIVE_DISPERSIONS: dict[str, float] = {
    "competing_listings_sd": 0.0,
}
# The old structural sign priors, kept for the pairs among the data-driven
# channels only as a fallback and for `beta_price`'s (never-observable) pairs.
_BETA_PRICE_CORRELATION_PRIORS: dict[tuple[str, str], float] = {
    ("beta_price", "absorption"): 0.50,
    ("beta_price", "competing_listings"): -0.30,
    ("beta_price", "comps_drift"): 0.20,
}
_FALLBACK_MACRO_CORRELATIONS: dict[tuple[str, str], float] = {
    ("absorption", "competing_listings"): -0.40,
    ("absorption", "comps_drift"): 0.30,
}

# Which channel each macro series informs. Each entry carries:
#   role       — the series role that measures it
#   sign       — aligns the series innovation with the channel's convention
#                (see scenarios.py "Sign conventions"). absorption is negated:
#                a market that slows (DOM rises) is a *lower* hazard of sale.
#   spec_field — the exact ScenarioSpec field the dispersion feeds
#   scale      — "absolute": the σ is already in the channel's units and feeds
#                the spec directly. "relative": the σ is a dimensionless
#                fractional swing that must be multiplied by the plan's baseline
#                level to become the spec's count σ. competing_listings is
#                relative because the metro active-listing *count* moves on a
#                completely different scale from the per-submarket
#                inventory_competition covariate; only its *percentage* swing
#                transfers, applied to whatever competing-listings baseline the
#                plan assumes.
_CHANNEL_SERIES = {
    "comps_drift": ("home_price_index", +1.0, "comps_drift_sd", "absolute"),
    "absorption": ("median_days_on_market", -1.0, "absorption_log_hazard_sd", "absolute"),
    "competing_listings": ("active_listings", +1.0, "competing_listings_sd", "relative"),
}
# Every data-driven channel is measured from log-returns: a price index and a
# duration move multiplicatively, and a listing count's *percentage* swing is
# the part that transfers across scales.
_CHANNEL_INNOVATION_KIND = "log_return"

# Baseline competing-listings level a relative swing is applied against when no
# plan-specific baseline is supplied. Matches PhaseInput.competing_listings.
_DEFAULT_COMPETING_LISTINGS_BASELINE = 30.0

# Default macro series wiring for a market, overridable from `config["macro"]`.
_DEFAULT_MACRO_CONFIG: dict[str, Any] = {
    "lookback_years": 12,
    "cache_days": 7,
    "fred_series": {
        "home_price_index": "MIXRSA",
        "median_days_on_market": "MEDDAYONMAR33100",
        "active_listings": "ACTLISCOU33100",
        "mortgage_rate": "MORTGAGE30US",
    },
    "bls_series": {
        "area_cpi": "CUURS35CSA0",
        "unemployment_rate": "LAUMT123310000000003",
    },
}


@dataclass(frozen=True)
class MacroSeries:
    """One fetched macro series, aligned to month starts where monthly.

    Attributes:
        series_id: the provider's series code.
        source: `"fred"` or `"bls"`.
        title: human-readable series name.
        dates: observation dates as ISO `YYYY-MM-DD` strings, ascending.
        values: observations, same length as `dates`, floats (NaN where the
            provider reported a missing value; never imputed).
        periods_per_year: nominal sampling frequency (12 monthly, 52 weekly,
            2 semi-annual), used to scale a per-period SD to a horizon.
        units: the series' own units, verbatim from the provider.
    """

    series_id: str
    source: str
    title: str
    dates: tuple[str, ...]
    values: tuple[float, ...]
    periods_per_year: float
    units: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "source": self.source,
            "title": self.title,
            "periods_per_year": self.periods_per_year,
            "units": self.units,
            "n_obs": len(self.values),
            "window": [self.dates[0], self.dates[-1]] if self.dates else [],
            "latest": self.values[-1] if self.values else None,
        }


@dataclass
class MacroSnapshot:
    """Data-derived scenario dispersions plus the evidence behind them.

    `dispersions` is what the scenario spec consumes; everything else exists so a
    reader can see how each number was produced and decide whether to trust it or
    click "input custom".

    Attributes:
        market: the market these numbers describe.
        horizon_days: the sale horizon the dispersions are scaled to. A σ over
            180 days and a σ over 360 days are different numbers, so the horizon
            is part of the result, not context.
        dispersions: `{channel_sd_name: value}` for the channels whose σ is
            already in the spec's units (comps_drift, absorption). Names match
            `ScenarioSpec` fields exactly.
        relative_dispersions: `{channel_sd_name: fraction}` for channels whose σ
            is a dimensionless swing needing a baseline to become a spec σ
            (competing_listings). Call `scenario_dispersions()` to resolve them.
        correlations: estimated pairwise correlations among the data-driven
            channels, keyed `"a|b"`. `beta_price`'s pairs are documented priors,
            carried here too so the caller can build one matrix.
        context: reported figures that inform judgement but do not set a
            dispersion — nominal vs real appreciation, the mortgage-rate level
            and its recent move, the unemployment rate and trend.
        series: metadata for each fetched series.
        source: `"fred_bls"`, `"cache"`, or `"static_fallback"`.
        fetched_at: ISO-8601 UTC timestamp of the fetch (or cache write).
        warnings: anything the reader must know — a series too short to estimate
            a correlation, a fallback in effect, a key missing.
    """

    market: str
    horizon_days: int
    dispersions: dict[str, float]
    correlations: dict[str, float]
    relative_dispersions: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    series: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: str = "fred_bls"
    fetched_at: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        """True when the dispersions were measured, not taken from fallback."""
        return self.source in {"fred_bls", "cache"}

    def scenario_dispersions(
        self, competing_listings_baseline: float | None = None
    ) -> dict[str, float]:
        """All channel σ resolved to `ScenarioSpec` units.

        Relative channels are multiplied by their baseline: a metro listing
        count that swings ±`r` over the horizon, applied to an assumed
        competing-listings level `b`, is a count σ of `r · b`.

        Args:
            competing_listings_baseline: the plan's assumed competing-listings
                level (e.g. the median `PhaseInput.competing_listings`). Defaults
                to 30, matching `PhaseInput`.

        Returns:
            `{spec_field: σ}` for every data-driven channel, ready to splat into
            `ScenarioSpec` / `ScenarioInput`.
        """
        baseline = (
            float(competing_listings_baseline)
            if competing_listings_baseline is not None
            else _DEFAULT_COMPETING_LISTINGS_BASELINE
        )
        out = dict(self.dispersions)
        for name, rel in self.relative_dispersions.items():
            out[name] = float(rel) * baseline
        return out

    def correlation_pairs(self) -> dict[tuple[str, str], float]:
        """Correlations as channel-pair tuples, for `ScenarioSpec.correlation`."""
        out: dict[tuple[str, str], float] = {}
        for key, value in self.correlations.items():
            a, b = key.split("|", 1)
            out[(a, b)] = float(value)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "horizon_days": self.horizon_days,
            "dispersions": self.dispersions,
            "relative_dispersions": self.relative_dispersions,
            "correlations": self.correlations,
            "context": self.context,
            "series": self.series,
            "source": self.source,
            "is_live": self.is_live,
            "fetched_at": self.fetched_at,
            "warnings": self.warnings,
        }


# ── Pure computation ───────────────────────────────────────────────────────

def _clean(values: tuple[float, ...]) -> np.ndarray:
    """Finite observations only, in order. Never imputes a gap."""
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def _period_innovations(series: MacroSeries, kind: str) -> np.ndarray:
    """Per-period innovations of a series: log-returns or first differences.

    `log_return` for a level that moves multiplicatively (a price index, a
    duration); `diff` for a count. Non-positive values make a log-return
    undefined, so those transitions are dropped rather than forced.
    """
    arr = _clean(series.values)
    if arr.size < 2:
        return np.empty(0, dtype="float64")
    if kind == "log_return":
        positive = arr > 0
        # Only consecutive positive pairs yield a defined log-return.
        both = positive[1:] & positive[:-1]
        return np.log(arr[1:][both]) - np.log(arr[:-1][both])
    if kind == "diff":
        return np.diff(arr)
    raise SchemaError(f"unknown innovation kind {kind!r}")


def _horizon_scale(periods_per_year: float, horizon_days: int) -> float:
    """√(periods in the horizon) — the random-walk scaling of a per-period SD.

    A per-period innovation SD scales to an h-day horizon by √(h / period
    length) under the standard independent-increments assumption. This is the
    same assumption the Monte Carlo already makes about the channels; it is not
    an extra estimate.
    """
    periods_in_horizon = periods_per_year * (horizon_days / _TRADING_DAYS)
    return float(np.sqrt(max(periods_in_horizon, 0.0)))


def channel_dispersion(series: MacroSeries, kind: str, horizon_days: int) -> float:
    """Horizon-scaled SD of a channel's innovations. Units follow `kind`.

    `log_return` → a fractional/log-point SD; `diff` → a level SD in the
    series' own units. Returns 0.0 when there is too little history to form an
    SD, which cleanly disables the channel rather than inventing dispersion.
    """
    innovations = _period_innovations(series, kind)
    if innovations.size < 2:
        return 0.0
    per_period_sd = float(np.std(innovations, ddof=1))
    return per_period_sd * _horizon_scale(series.periods_per_year, horizon_days)


def _aligned_innovations(
    series_by_role: dict[str, MacroSeries],
) -> tuple[list[str], np.ndarray]:
    """Align the three data-driven channels' innovations on common months.

    Returns the channel order and an `(n_common, n_channel)` matrix, sign-
    adjusted so each column already points the way its channel's convention
    does (absorption negated). Correlations come off this matrix. An empty
    matrix means no overlapping window — the caller falls back to priors.
    """
    per_channel_dates: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for channel, (role, sign, _spec_field, _scale) in _CHANNEL_SERIES.items():
        series = series_by_role.get(role)
        if series is None:
            continue
        arr = np.asarray(series.values, dtype="float64")
        dates = series.dates
        pairs: dict[str, float] = {}
        for i in range(1, len(arr)):
            prev, cur = arr[i - 1], arr[i]
            if not (np.isfinite(prev) and np.isfinite(cur)):
                continue
            if prev <= 0 or cur <= 0:  # all channels use log-returns
                continue
            innovation = np.log(cur) - np.log(prev)
            pairs[dates[i]] = sign * float(innovation)
        if pairs:
            per_channel_dates[channel] = pairs
            order.append(channel)

    if len(order) < 2:
        return order, np.empty((0, len(order)), dtype="float64")

    common = set.intersection(*(set(per_channel_dates[c]) for c in order))
    common_sorted = sorted(common)
    if len(common_sorted) < 3:
        return order, np.empty((0, len(order)), dtype="float64")

    matrix = np.array(
        [[per_channel_dates[c][d] for c in order] for d in common_sorted],
        dtype="float64",
    )
    return order, matrix


def estimate_correlations(
    series_by_role: dict[str, MacroSeries],
) -> tuple[dict[str, float], list[str]]:
    """Pearson correlations among the data-driven channels, from aligned history.

    Returns the correlation dict keyed `"a|b"` and any warnings. Pairs whose
    overlapping window is too short, or where one channel has no variation, fall
    back to the documented prior for that pair so the returned matrix is always
    complete over the observable channels.
    """
    warnings: list[str] = []
    order, matrix = _aligned_innovations(series_by_role)
    out: dict[str, float] = {}

    if matrix.shape[0] >= 3:
        corr = np.corrcoef(matrix, rowvar=False)
        n = matrix.shape[0]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                j = order.index(b)
                rho = corr[i, j]
                if np.isfinite(rho):
                    out[f"{a}|{b}"] = float(np.clip(rho, -1.0, 1.0))
        warnings.append(
            f"macro correlations among {order} estimated from {n} aligned "
            "monthly innovations."
        )
    else:
        warnings.append(
            "too little overlapping macro history to estimate channel "
            "correlations; falling back to documented sign priors."
        )

    # Fill any missing observable pair from the fallback priors.
    for (a, b), rho in _FALLBACK_MACRO_CORRELATIONS.items():
        key = f"{a}|{b}"
        if key not in out and f"{b}|{a}" not in out:
            out[key] = rho

    return out, warnings


def fallback_correlation_pairs() -> dict[tuple[str, str], float]:
    """Documented, PSD-by-design correlation priors as channel-pair tuples.

    The safety net when estimated correlations, combined with `beta_price`'s
    priors, fail positive semi-definiteness for a particular set of active
    channels. These are the historical sign priors and are known PSD.
    """
    out: dict[tuple[str, str], float] = dict(_FALLBACK_MACRO_CORRELATIONS)
    out.update(_BETA_PRICE_CORRELATION_PRIORS)
    return out


def _full_correlation_dict(
    macro_corr: dict[str, float] | dict[tuple[str, str], float],
) -> dict[str, float]:
    """Combine estimated macro correlations with beta_price's documented priors.

    Accepts either string (`"a|b"`) or tuple (`(a, b)`) keys and always returns
    string keys, which is the on-the-wire shape `MacroSnapshot.correlations` uses.
    """
    out: dict[str, float] = {}
    for key, value in macro_corr.items():
        skey = key if isinstance(key, str) else f"{key[0]}|{key[1]}"
        out[skey] = float(value)
    for (a, b), rho in _BETA_PRICE_CORRELATION_PRIORS.items():
        out[f"{a}|{b}"] = rho
    return out


# ── Fetching (I/O, isolated here like ingest_mls) ──────────────────────────

def _macro_config(config: MarketConfig | None) -> dict[str, Any]:
    """Merge market config's `macro:` block over the defaults."""
    merged = json.loads(json.dumps(_DEFAULT_MACRO_CONFIG))  # deep copy
    block = (config or {}).get("macro") if config else None
    if isinstance(block, dict):
        for key, value in block.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _read_keys() -> dict[str, str | None]:
    """FRED and BLS keys from the environment, loading `.env` if present.

    BLS works keyless (25 requests/day); FRED requires a key. Missing keys are
    returned as None and handled by the caller, not raised here — a missing key
    is a fallback path, not a crash.
    """
    try:  # python-dotenv is a locked dependency; load lazily and once.
        from dotenv import load_dotenv

        load_dotenv(_BACKEND_ROOT.parent / ".env")
        load_dotenv(_BACKEND_ROOT / ".env")
    except Exception:  # pragma: no cover - dotenv absent is non-fatal
        logger.debug("dotenv not loaded; reading os.environ directly")
    return {
        "fred": os.environ.get("FRED_API_KEY") or None,
        "bls": os.environ.get("BLS_API_KEY") or None,
    }


def _fred_frequency(series_values_meta: str) -> float:
    return {"D": 365.25, "W": 52.0, "M": 12.0, "Q": 4.0, "SA": 2.0, "A": 1.0}.get(
        series_values_meta, 12.0
    )


def _fetch_fred_series(
    role: str, series_id: str, api_key: str, start: str, client: Any
) -> MacroSeries:
    """Fetch one FRED series' observations. Raises on any transport/API error."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    resp = client.get(_FRED_BASE, params=params, timeout=_HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    obs = payload.get("observations", [])
    dates: list[str] = []
    values: list[float] = []
    for row in obs:
        dates.append(str(row.get("date")))
        raw = row.get("value")
        values.append(float(raw) if raw not in (None, ".", "") else float("nan"))
    freq = _infer_periods_per_year(dates)
    return MacroSeries(
        series_id=series_id,
        source="fred",
        title=role,
        dates=tuple(dates),
        values=tuple(values),
        periods_per_year=freq,
    )


def _infer_periods_per_year(dates: list[str]) -> float:
    """Median spacing of ISO dates → nominal periods per year."""
    if len(dates) < 3:
        return 12.0
    try:
        parsed = sorted(datetime.fromisoformat(d) for d in dates if d and d != "None")
    except ValueError:
        return 12.0
    if len(parsed) < 3:
        return 12.0
    gaps = np.diff(np.array([p.timestamp() for p in parsed]))
    median_days = float(np.median(gaps)) / 86400.0
    if median_days <= 0:
        return 12.0
    return float(np.clip(round(365.25 / median_days), 1.0, 365.0))


def _fetch_bls_series(
    series_ids: dict[str, str], api_key: str | None, start_year: int, end_year: int,
    client: Any,
) -> dict[str, MacroSeries]:
    """Fetch BLS series (keyless allowed). Returns role → MacroSeries.

    A BLS failure is not fatal to the whole snapshot — BLS is context, not a
    dispersion source — so this raises and the caller decides.
    """
    body: dict[str, Any] = {
        "seriesid": list(series_ids.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if api_key:
        body["registrationkey"] = api_key
    resp = client.post(_BLS_BASE, json=body, timeout=_HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise SchemaError(f"BLS request failed: {payload.get('message')}")
    by_id = {s["seriesID"]: s for s in payload.get("Results", {}).get("series", [])}
    out: dict[str, MacroSeries] = {}
    for role, series_id in series_ids.items():
        raw = by_id.get(series_id)
        if not raw:
            continue
        rows = list(reversed(raw.get("data", [])))  # BLS returns newest-first
        dates: list[str] = []
        values: list[float] = []
        for row in rows:
            period = row.get("period", "")
            if not period.startswith("M"):  # skip annual averages (M13)
                if period.startswith("S"):  # semiannual S01/S02 → mid-period month
                    month = "01" if period == "S01" else "07"
                else:
                    continue
            else:
                month = period[1:]
            if month == "13":
                continue
            dates.append(f"{row.get('year')}-{month}-01")
            try:
                values.append(float(row.get("value")))
            except (TypeError, ValueError):
                values.append(float("nan"))
        out[role] = MacroSeries(
            series_id=series_id,
            source="bls",
            title=role,
            dates=tuple(dates),
            values=tuple(values),
            periods_per_year=_infer_periods_per_year(dates),
        )
    return out


# ── Context figures ────────────────────────────────────────────────────────

def _annualized_return(series: MacroSeries) -> float | None:
    """Trailing 12-month log-return of a level series, as a decimal."""
    arr = _clean(series.values)
    ppy = int(round(series.periods_per_year))
    if arr.size <= ppy or arr[-1] <= 0 or arr[-1 - ppy] <= 0:
        return None
    return float(np.log(arr[-1]) - np.log(arr[-1 - ppy]))


def _build_context(series_by_role: dict[str, MacroSeries]) -> dict[str, Any]:
    """Reported figures that inform judgement but set no dispersion."""
    context: dict[str, Any] = {}

    hpi = series_by_role.get("home_price_index")
    cpi = series_by_role.get("area_cpi")
    if hpi is not None:
        nominal = _annualized_return(hpi)
        context["home_price_appreciation_yoy_nominal"] = nominal
        if cpi is not None and nominal is not None:
            inflation = _annualized_return(cpi)
            if inflation is not None:
                context["cpi_inflation_yoy"] = inflation
                context["home_price_appreciation_yoy_real"] = nominal - inflation

    rate = series_by_role.get("mortgage_rate")
    if rate is not None:
        arr = _clean(rate.values)
        if arr.size:
            context["mortgage_rate_pct"] = float(arr[-1])
        if arr.size > 52:  # ~1y of weekly obs
            context["mortgage_rate_change_1y_pp"] = float(arr[-1] - arr[-52])

    unemployment = series_by_role.get("unemployment_rate")
    if unemployment is not None:
        arr = _clean(unemployment.values)
        if arr.size:
            context["unemployment_rate_pct"] = float(arr[-1])
        if arr.size > 12:
            context["unemployment_change_1y_pp"] = float(arr[-1] - arr[-12])

    return context


# ── Orchestration ──────────────────────────────────────────────────────────

def _cache_path(market: str) -> Path:
    return _BACKEND_ROOT / "data" / "processed" / market / "macro_cache.json"


def _load_cache(market: str, horizon_days: int, max_age_days: int) -> MacroSnapshot | None:
    path = _cache_path(market)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if int(raw.get("horizon_days", -1)) != int(horizon_days):
        return None  # a cache scaled to a different horizon is the wrong number
    fetched_at = raw.get("fetched_at", "")
    try:
        age = datetime.now(UTC) - datetime.fromisoformat(fetched_at)
    except ValueError:
        return None
    if age > timedelta(days=max_age_days):
        return None
    return MacroSnapshot(
        market=raw["market"],
        horizon_days=raw["horizon_days"],
        dispersions=raw["dispersions"],
        relative_dispersions=raw.get("relative_dispersions", {}),
        correlations=raw["correlations"],
        context=raw.get("context", {}),
        series=raw.get("series", {}),
        source="cache",
        fetched_at=fetched_at,
        warnings=list(raw.get("warnings", [])),
    )


def _write_cache(snapshot: MacroSnapshot) -> None:
    path = _cache_path(snapshot.market)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = snapshot.as_dict()
        payload["source"] = "fred_bls"  # cache always stores the live provenance
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - cache write failure is non-fatal
        logger.warning("could not write macro cache: %s", exc)


def _fallback_snapshot(
    market: str, horizon_days: int, warnings: list[str]
) -> MacroSnapshot:
    """The documented last-resort snapshot when nothing could be fetched."""
    return MacroSnapshot(
        market=market,
        horizon_days=horizon_days,
        dispersions=dict(_FALLBACK_DISPERSIONS),
        relative_dispersions=dict(_FALLBACK_RELATIVE_DISPERSIONS),
        correlations=_full_correlation_dict(dict(_FALLBACK_MACRO_CORRELATIONS)),
        context={},
        series={},
        source="static_fallback",
        fetched_at=datetime.now(UTC).isoformat(),
        warnings=[
            "MACRO UNAVAILABLE — dispersions are documented fallback assumptions, "
            "not measured from data. Fetch failed and no fresh cache exists. "
            "Click 'input custom' to set your own, or configure FRED_API_KEY.",
            *warnings,
        ],
    )


def build_macro_snapshot(
    market: str = "miami",
    horizon_days: int = 180,
    *,
    config: MarketConfig | None = None,
    use_cache: bool = True,
    client: Any | None = None,
) -> MacroSnapshot:
    """Fetch macro series and derive the scenario dispersions for `market`.

    Args:
        market: market id; selects the series wiring and the cache file.
        horizon_days: the sale horizon the dispersions are scaled to. A σ is
            horizon-specific, so this is part of the answer.
        config: market config; its optional `macro:` block overrides series ids,
            `lookback_years`, and `cache_days`.
        use_cache: read a fresh on-disk snapshot before hitting the network.
        client: an httpx-like client (injectable for tests). When None, a real
            `httpx.Client` is created for the call.

    Returns:
        A `MacroSnapshot`. Never raises for a network or key problem — those
        degrade to `cache` then `static_fallback`, both labelled in `source`.
        Only a genuine programming error (bad market id) raises.
    """
    market = validate_market_id(market)
    horizon_days = int(horizon_days)
    if horizon_days < 1:
        raise SchemaError(f"horizon_days must be >= 1, got {horizon_days}")

    # An explicit offline switch, honored before cache or network. Keeps tests
    # hermetic and lets an operator run the stack with no outbound calls; the
    # result is the documented fallback, clearly labelled.
    if os.environ.get("MACRO_DISABLE_NETWORK", "").strip().lower() in {"1", "true", "yes"}:
        return _fallback_snapshot(
            market, horizon_days, ["MACRO_DISABLE_NETWORK is set; not fetching."]
        )

    macro_cfg = _macro_config(config)
    cache_days = int(macro_cfg.get("cache_days", 7))
    if use_cache:
        cached = _load_cache(market, horizon_days, cache_days)
        if cached is not None:
            logger.info("macro snapshot served from cache (%s)", cached.fetched_at)
            return cached

    keys = _read_keys()
    warnings: list[str] = []
    if not keys["fred"]:
        return _fallback_snapshot(
            market, horizon_days, ["FRED_API_KEY is not set."]
        )

    lookback_years = int(macro_cfg.get("lookback_years", 12))
    start_date = (datetime.now(UTC) - timedelta(days=365 * lookback_years)).date()
    start_year = start_date.year

    owns_client = client is None
    if owns_client:
        import httpx

        client = httpx.Client()
    try:
        series_by_role: dict[str, MacroSeries] = {}
        for role, series_id in macro_cfg["fred_series"].items():
            try:
                series_by_role[role] = _fetch_fred_series(
                    role, series_id, keys["fred"], start_date.isoformat(), client
                )
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash
                warnings.append(f"FRED series {series_id} ({role}) failed: {exc}")

        try:
            bls = _fetch_bls_series(
                macro_cfg["bls_series"], keys["bls"], start_year,
                datetime.now(UTC).year, client,
            )
            series_by_role.update(bls)
        except Exception as exc:  # noqa: BLE001 - BLS is context; degrade quietly
            warnings.append(f"BLS fetch failed (context only): {exc}")
    finally:
        if owns_client:
            client.close()

    # Any of the three FRED price/inventory series missing means we cannot honor
    # "fully data-driven" — fall back rather than silently disabling a channel
    # to zero, which would understate risk.
    required_roles = {role for role, *_ in _CHANNEL_SERIES.values()}
    missing = [r for r in required_roles if r not in series_by_role]
    if missing:
        return _fallback_snapshot(
            market, horizon_days,
            [f"required FRED series missing: {missing}.", *warnings],
        )

    dispersions: dict[str, float] = {}
    relative_dispersions: dict[str, float] = {}
    for _channel, (role, _sign, spec_field, scale) in _CHANNEL_SERIES.items():
        sigma = channel_dispersion(
            series_by_role[role], _CHANNEL_INNOVATION_KIND, horizon_days
        )
        if scale == "relative":
            relative_dispersions[spec_field] = sigma
        else:
            dispersions[spec_field] = sigma

    macro_corr, corr_warnings = estimate_correlations(series_by_role)
    warnings.extend(corr_warnings)

    snapshot = MacroSnapshot(
        market=market,
        horizon_days=horizon_days,
        dispersions=dispersions,
        relative_dispersions=relative_dispersions,
        correlations=_full_correlation_dict(macro_corr),
        context=_build_context(series_by_role),
        series={role: s.as_dict() for role, s in series_by_role.items()},
        source="fred_bls",
        fetched_at=datetime.now(UTC).isoformat(),
        warnings=warnings,
    )
    if use_cache:
        _write_cache(snapshot)
    return snapshot


def format_macro_snapshot(snapshot: MacroSnapshot) -> str:
    """Human-readable snapshot; the printed report is the CLI deliverable."""
    lines = [
        f"MACRO  market={snapshot.market}  horizon={snapshot.horizon_days}d  "
        f"source={snapshot.source}  fetched={snapshot.fetched_at}",
        "",
        "DISPERSIONS (scenario σ, data-derived)",
    ]
    for name, value in snapshot.dispersions.items():
        lines.append(f"  {name}: {value:.5f}")
    for name, value in snapshot.relative_dispersions.items():
        lines.append(f"  {name}: {value:.5f} (relative swing; × competing-listings baseline)")
    lines.append("")
    lines.append("CORRELATIONS")
    for key, value in snapshot.correlations.items():
        lines.append(f"  {key}: {value:+.3f}")
    if snapshot.context:
        lines.append("")
        lines.append("CONTEXT")
        for key, value in snapshot.context.items():
            shown = f"{value:+.4f}" if isinstance(value, float) else str(value)
            lines.append(f"  {key}: {shown}")
    if snapshot.series:
        lines.append("")
        lines.append("SERIES")
        for role, meta in snapshot.series.items():
            lines.append(
                f"  {role}: {meta.get('series_id')} [{meta.get('source')}] "
                f"n={meta.get('n_obs')} window={meta.get('window')}"
            )
    if snapshot.warnings:
        lines.append("")
        lines.append("WARNINGS")
        for w in snapshot.warnings:
            lines.append(f"  ⚠  {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.config import load_market_config

    parser = argparse.ArgumentParser(description="Derive macro scenario dispersions")
    parser.add_argument("--market", default="miami")
    parser.add_argument("--horizon-days", type=int, default=180)
    parser.add_argument("--no-cache", action="store_true", help="Force a fresh fetch")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_market_config(args.market)
    except SchemaError:
        config = None
    snapshot = build_macro_snapshot(
        args.market, args.horizon_days, config=config, use_cache=not args.no_cache
    )
    print(format_macro_snapshot(snapshot))
    return 0 if snapshot.is_live else 2


if __name__ == "__main__":
    raise SystemExit(main())
