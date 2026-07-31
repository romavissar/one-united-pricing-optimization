"""Synthetic MLS generator with planted parameters.

Every constant in this module is a *planted* ground truth for a data-generating
process, not an estimate. `AGENTS.md` §1 exempts synthetic generators from the
no-hardcoded-parameters rule precisely so that estimators can be verified by
round-trip: generate with a known `true_beta_price`, fit, assert recovery. The
planted values are returned in `SyntheticTruth` rather than hidden in the code.

Units:
- Price: `$/sqft` internally; `original_list_price` / `close_price` are USD totals.
- Area: sqft.
- Duration / hazard: days and 1/days.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MarketConfig, load_market_config
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

# --- Planted hedonic surface -------------------------------------------------
_TIER_JITTER_SD = 0.05

# fair_ppsf multiplier for floor: (1 + alpha * ln(floor + 1)) ** gamma
_FLOOR_ALPHA = 0.06
_FLOOR_GAMMA = 1.0

# View premium spans this range across the ordered view_categories list.
_VIEW_PREMIUM_TOP = 1.35
_VIEW_PREMIUM_BOTTOM = 0.88

# (sqft / reference) ** size_elasticity — larger units carry a mild $/sqft premium.
_SIZE_REFERENCE_SQFT = 1400.0
_SIZE_ELASTICITY = 0.08

# --- Planted hazard ----------------------------------------------------------
_SUBMARKET_LOG_HAZARD_SD = 0.15
_MONTH_LOG_HAZARD_SD = 0.12
# Ceiling for listings allowed to run past the horizon; stays under the
# bad_duration filter in clean.py (1095 days).
_MAX_CENSOR_DAYS = 900.0

# --- Planted price behaviour -------------------------------------------------
_BASE_CONCESSION = 0.02
_CONCESSION_SLOPE = 0.35
_PRICE_CUT_PROB = 0.35
_PRICE_CUT_RANGE = (0.02, 0.12)

# --- Listing cosmetics -------------------------------------------------------
_BUILDINGS_PER_SUBMARKET = 12
_CENSORED_STATUS_SPLIT: dict[str, float] = {
    "Expired": 0.40,
    "Cancelled": 0.35,
    "Withdrawn": 0.25,
}
_DEFAULT_MISSINGNESS: dict[str, float] = {
    "floor": 0.03,
    "hoa_monthly": 0.02,
    "year_built": 0.01,
    "baths_half": 0.05,
}

_STREET_NAMES = (
    "Brickell Ave",
    "Collins Ave",
    "Biscayne Blvd",
    "Ocean Dr",
    "Bayshore Dr",
    "Alton Rd",
    "Coral Way",
    "Grove Isle Dr",
    "Harbour Dr",
    "Sunny Isles Blvd",
)
_BUILDING_WORDS = (
    "Aria",
    "Solstice",
    "Vantage",
    "Marea",
    "Costa",
    "Palazzo",
    "Meridian",
    "Lumen",
    "Cascade",
    "Verano",
    "Arbor",
    "Continuum",
)


@dataclass(frozen=True)
class AvailabilityProfile:
    """What an export actually carries, plus market behaviour that matches it.

    A generator that emits every canonical field is a poor stand-in for an
    export that omits half of them: features verified against the rich case
    silently disappear on real data. `like_export` mirrors the field
    availability, price level, censoring, and status mix actually observed in
    the Miami MLS pull, so Phase 2+ can be verified under both.

    absent_fields are omitted from the frame entirely, not nulled — that is how
    a missing column presents to `map_columns`.

    building_quality_share is the fraction of fair-value-noise *variance* that
    sits at the building rather than the unit. It decides whether building
    fixed effects can absorb the quality contamination in rel_price_premium:
    at 1.0 they absorb all of it, at 0.0 none. 0.65 is the working assumption
    for a market where brand, amenities, and construction quality vary far more
    between towers than between units within one.
    """

    name: str
    absent_fields: tuple[str, ...]
    tier_base_ppsf: Mapping[int, float]
    fair_value_noise_sd: float
    building_quality_share: float
    target_sold_share: float
    pending_share_of_events: float
    censor_mean_horizons: float
    overrun_share: float
    distressed_share: float
    rental_share: float


# Fields the Miami export omits. `unit_number` is absent too: that export hides
# the unit inside the address string, which is why extract_unit_from_address
# exists.
_EXPORT_ABSENT_FIELDS = (
    "unit_number",
    "property_type",
    "sale_type",
    "new_construction",
    "hoa_frequency",
    "tax_annual",
    "city",
    "list_agent_name",
    "list_office_name",
    "view_description",
)

PROFILES: dict[str, AvailabilityProfile] = {
    "rich": AvailabilityProfile(
        name="rich",
        absent_fields=(),
        tier_base_ppsf={0: 1600.0, 1: 1150.0, 2: 850.0},
        fair_value_noise_sd=0.12,
        building_quality_share=0.65,
        target_sold_share=0.65,
        pending_share_of_events=0.0,
        censor_mean_horizons=2.0,
        overrun_share=0.0,
        distressed_share=0.015,
        rental_share=0.005,
    ),
    # Levels below are matched to the observed Miami export: median list $/sqft
    # ~683, p90/p10 ~4.0, 43.8% of listings reach an event, 13.2% of those
    # events are PENDING rather than closed, longest listing ~720 days.
    "like_export": AvailabilityProfile(
        name="like_export",
        absent_fields=_EXPORT_ABSENT_FIELDS,
        tier_base_ppsf={0: 820.0, 1: 590.0, 2: 440.0},
        fair_value_noise_sd=0.47,
        building_quality_share=0.65,
        target_sold_share=0.438,
        pending_share_of_events=0.132,
        censor_mean_horizons=2.0,
        overrun_share=0.06,
        # The export carries no sale_type or property_type column, so distressed
        # and rental listings cannot be identified, let alone filtered.
        distressed_share=0.0,
        rental_share=0.0,
    ),
}


@dataclass
class SyntheticTruth:
    """Planted parameters the downstream estimators must recover.

    beta_price: coefficient on rel_price_premium (a ratio minus one).
    submarket_base_ppsf: base fair value in $/sqft.
    base_hazard_daily: baseline Weibull rate, 1/days.
    """

    beta_price: float
    submarket_base_ppsf: dict[str, float]
    floor_premium_alpha: float
    floor_premium_gamma: float
    view_premium: dict[str, float]
    size_elasticity: float
    fair_value_noise_sd: float
    building_quality_share: float
    aggressiveness_sd: float
    weibull_shape: float
    base_hazard_daily: float
    target_sold_share: float
    horizon_days: int
    # "realized": hazard driven by rel_price_premium, the same quantity the
    # estimator regresses on, so beta is recoverable without controls.
    # "latent": hazard driven by the seller's own aggressiveness, so quality
    # contamination in rel_price_premium attenuates a naive fit.
    hazard_basis: str = "realized"
    profile: str = "rich"
    # False when view_description is withheld: the view premium is still baked
    # into fair value, so it becomes genuine unobserved quality.
    view_observable: bool = True
    submarket_log_hazard: dict[str, float] = field(default_factory=dict)
    month_log_hazard: dict[str, float] = field(default_factory=dict)


@dataclass
class SyntheticMLS:
    """Generated export plus the latent variables needed to verify recovery.

    frame: canonical MLS_SCHEMA columns, export-shaped (dates as strings).
    latent: per-row ground truth (aggressiveness, fair value noise, realized
        rel_price_premium, uncensored time-to-sale, censoring time).
    """

    frame: pd.DataFrame
    truth: SyntheticTruth
    latent: pd.DataFrame


def _submarket_bases(
    submarkets: Mapping[str, Mapping[str, object]],
    tier_base_ppsf: Mapping[int, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    bases: dict[str, float] = {}
    for name, meta in submarkets.items():
        tier = int(meta.get("tier", 1)) if isinstance(meta, Mapping) else 1
        base = tier_base_ppsf.get(tier, tier_base_ppsf[1])
        bases[str(name)] = float(base * np.exp(rng.normal(0.0, _TIER_JITTER_SD)))
    return bases


def _view_premiums(categories: list[str]) -> dict[str, float]:
    """Monotone decreasing premium across the ordered view_categories list."""
    if not categories:
        return {}
    if len(categories) == 1:
        return {categories[0]: 1.0}
    step = (_VIEW_PREMIUM_TOP - _VIEW_PREMIUM_BOTTOM) / (len(categories) - 1)
    return {cat: _VIEW_PREMIUM_TOP - i * step for i, cat in enumerate(categories)}


def _solve_base_hazard(
    target_sold_share: float,
    horizon_days: int,
    shape: float,
    censor_mean_days: float,
) -> float:
    """Baseline Weibull rate (1/days) hitting `target_sold_share` at the reference unit.

    A listing sells when T <= min(C, horizon), so solving P(T <= horizon) alone
    overshoots the censoring and undershoots the target. This integrates
    P(sold) = int_0^H f_T(t) * P(C > t) dt for the reference case
    (rel_price_premium = 0, no submarket or month effect) and bisects on the
    rate. Submarket, month, and price effects then move the realized share
    around this reference, which is what keeps the share responsive to
    `true_beta_price`.
    """
    if not 0.0 < target_sold_share < 1.0:
        raise SchemaError(f"target_sold_share must be in (0, 1), got {target_sold_share}")

    grid = np.linspace(1e-9, float(horizon_days), 4001)
    survival_c = np.exp(-grid / censor_mean_days)

    def _sold_share(rate: float) -> float:
        scaled = rate * grid
        density = shape * rate * scaled ** (shape - 1.0) * np.exp(-(scaled**shape))
        return float(np.trapezoid(density * survival_c, grid))

    lo, hi = 1e-8, 10.0
    if _sold_share(hi) < target_sold_share:
        raise SchemaError(
            f"target_sold_share={target_sold_share} unreachable with "
            f"horizon_days={horizon_days} and censor_mean_days={censor_mean_days}"
        )
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if _sold_share(mid) < target_sold_share:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _build_buildings(
    submarkets: Mapping[str, Mapping[str, object]], rng: np.random.Generator
) -> pd.DataFrame:
    """One row per synthetic building: name, address, stories, year built, quality.

    `quality_z` is a standard-normal building-level draw. It is scaled into the
    fair-value noise by `building_quality_share` so that a portion of unobserved
    quality is constant within a building and therefore absorbable by building
    fixed effects.
    """
    rows: list[dict[str, object]] = []
    for sub_i, (name, meta) in enumerate(submarkets.items()):
        zips = list(meta.get("zips") or []) if isinstance(meta, Mapping) else []
        if not zips:
            raise SchemaError(f"Submarket {name} has no zips in market config")
        for b in range(_BUILDINGS_PER_SUBMARKET):
            word = _BUILDING_WORDS[(sub_i * _BUILDINGS_PER_SUBMARKET + b) % len(_BUILDING_WORDS)]
            rows.append(
                {
                    "submarket": str(name),
                    "zip_code": str(rng.choice(zips)),
                    "building_name": f"{word} {str(name).replace('_', ' ').title()} {b + 1}",
                    "street_address": (
                        f"{int(rng.integers(100, 9999))} "
                        f"{_STREET_NAMES[int(rng.integers(0, len(_STREET_NAMES)))]}"
                    ),
                    "total_stories": int(rng.integers(8, 71)),
                    "year_built": int(rng.integers(1990, 2025)),
                    "quality_z": float(rng.normal(0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def _list_dates(n: int, end_month: str, months: int, rng: np.random.Generator) -> pd.Series:
    """Spread list dates uniformly across `months` calendar months."""
    end = pd.Period(end_month, freq="M")
    periods = [(end - (months - 1 - i)) for i in range(months)]
    picks = rng.integers(0, months, size=n)
    out: list[pd.Timestamp] = []
    for p in picks:
        period = periods[int(p)]
        day = int(rng.integers(1, period.days_in_month + 1))
        out.append(pd.Timestamp(year=period.year, month=period.month, day=day))
    return pd.Series(out)


def generate_synthetic_mls(
    n: int = 6000,
    seed: int = 42,
    true_beta_price: float = -1.6,
    horizon_days: int = 180,
    market: str = "miami",
    config: MarketConfig | None = None,
    profile: str = "rich",
    target_sold_share: float | None = None,
    weibull_shape: float = 1.0,
    aggressiveness_sd: float = 0.08,
    building_quality_share: float | None = None,
    hazard_basis: str = "realized",
    confound_strength: float = 0.0,
    missingness: Mapping[str, float] | None = None,
    months: int = 24,
    end_month: str = "2025-12",
) -> SyntheticMLS:
    """Generate a synthetic MLS export from a known data-generating process.

    Args:
        n: number of listings.
        seed: RNG seed; identical seeds produce identical frames.
        true_beta_price: planted coefficient on rel_price_premium, which is a
            ratio minus one (-1.6 means a listing priced 10% above its
            submarket-month median has ~15% lower sale hazard).
        horizon_days: administrative censoring horizon, days.
        profile: data-availability profile, `"rich"` or `"like_export"`. The
            latter mirrors the fields, price level, censoring, and status mix of
            the real Miami export so downstream phases are verified against a
            faithful stand-in as well as a complete one.
        target_sold_share: overrides the profile. Drives the baseline hazard so
            the event mix is an input rather than a tuned constant.
        weibull_shape: 1.0 gives a constant (exponential) hazard.
        aggressiveness_sd: sd of the seller's pricing deviation a_i. This is the
            variation that identifies elasticity.
        building_quality_share: overrides the profile. Fraction of fair-value
            noise *variance* placed at the building rather than the unit. It
            sets how much of the quality contamination in rel_price_premium
            building fixed effects can absorb, and therefore how far a
            controlled fit under `hazard_basis="latent"` can close on the
            planted beta. At 1.0 the residual error is zero and recovery is
            exact; at 0.0 fixed effects buy nothing.
        hazard_basis: which price signal drives the hazard.
            `"realized"` uses rel_price_premium, the same quantity the estimator
            regresses on, so recovery is an exact test of estimator code.
            `"latent"` uses a_i instead. Because rel_price_premium is dominated
            by unit quality rather than pricing choice (under 2% of its variance
            is a_i in the `like_export` profile), a naive fit under `"latent"`
            is attenuated toward zero and only hedonic controls recover the
            planted beta. Use it to test specification, not estimator code.
        confound_strength: correlation forced between a_i and the unobserved
            fair-value noise. 0.0 keeps the case cleanly identified; raise it to
            deliberately break identification in adversarial tests.
        missingness: per-field null injection rates, to exercise the
            never-impute path.

    Returns:
        SyntheticMLS with an export-shaped frame, the planted truth, and the
        latent per-row variables.
    """
    if n <= 0:
        raise SchemaError(f"n must be positive, got {n}")
    if not 0.0 <= confound_strength < 1.0:
        raise SchemaError(f"confound_strength must be in [0, 1), got {confound_strength}")
    if profile not in PROFILES:
        raise SchemaError(f"Unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    if hazard_basis not in {"realized", "latent"}:
        raise SchemaError(
            f"hazard_basis must be 'realized' or 'latent', got {hazard_basis!r}"
        )

    prof = PROFILES[profile]
    sold_target = prof.target_sold_share if target_sold_share is None else target_sold_share
    quality_share = (
        prof.building_quality_share
        if building_quality_share is None
        else float(building_quality_share)
    )
    if not 0.0 <= quality_share <= 1.0:
        raise SchemaError(
            f"building_quality_share must be in [0, 1], got {quality_share}"
        )

    cfg = config if config is not None else load_market_config(market)
    submarkets = cfg.get("submarkets") or {}
    if not submarkets:
        raise SchemaError(f"Market config for {market} defines no submarkets")
    view_categories = [str(v) for v in (cfg.get("view_categories") or [])]

    rng = np.random.default_rng(seed)
    rates = {**_DEFAULT_MISSINGNESS, **dict(missingness or {})}

    base_ppsf = _submarket_bases(submarkets, prof.tier_base_ppsf, rng)
    view_premium = _view_premiums(view_categories)
    buildings = _build_buildings(submarkets, rng)

    # --- 1. Draw units -------------------------------------------------------
    b_idx = rng.integers(0, len(buildings), size=n)
    units = buildings.iloc[b_idx].reset_index(drop=True)

    total_stories = units["total_stories"].to_numpy()
    floor = rng.integers(1, total_stories + 1)
    living_area_sqft = np.round(rng.lognormal(np.log(1400.0), 0.35, size=n)).clip(600, 8000)
    beds = np.clip(np.round(living_area_sqft / 750.0), 1, 6).astype(int)
    baths_full = np.clip(beds + rng.integers(0, 2, size=n), 1, 7).astype(int)
    baths_half = rng.integers(0, 2, size=n)
    view = (
        rng.choice(view_categories, size=n)
        if view_categories
        else np.array(["city_skyline"] * n)
    )

    # --- 2. Fair value from the hedonic surface ------------------------------
    base = np.array([base_ppsf[s] for s in units["submarket"]])
    floor_mult = (1.0 + _FLOOR_ALPHA * np.log(floor + 1.0)) ** _FLOOR_GAMMA
    view_mult = np.array([view_premium.get(str(v), 1.0) for v in view])
    size_mult = (living_area_sqft / _SIZE_REFERENCE_SQFT) ** _SIZE_ELASTICITY
    # Unobserved quality splits into a building component, constant within a
    # tower and therefore absorbable by building fixed effects, and a unit
    # component that no fixed effect can reach. Total variance is unchanged by
    # the split, so the marginal distribution of fair value does not move with
    # quality_share — only what a controlled fit can recover does.
    z_building = units["quality_z"].to_numpy()
    z_unit = rng.normal(0.0, 1.0, size=n)
    z_eps = np.sqrt(quality_share) * z_building + np.sqrt(1.0 - quality_share) * z_unit
    eps = prof.fair_value_noise_sd * z_eps
    # The view premium is always in fair value. When the profile withholds the
    # column it simply becomes unobserved quality, which is the honest depiction
    # of an export that carries no view field.
    fair_ppsf = base * floor_mult * view_mult * size_mult * np.exp(eps)

    # --- 3. Listing aggressiveness, independent of fair-value noise ----------
    z_indep = rng.normal(0.0, 1.0, size=n)
    z_a = confound_strength * z_eps + np.sqrt(1.0 - confound_strength**2) * z_indep
    aggressiveness = aggressiveness_sd * z_a
    list_ppsf = fair_ppsf * (1.0 + aggressiveness)
    original_list_price = np.round(list_ppsf * living_area_sqft, -3)

    list_date = _list_dates(n, end_month=end_month, months=months, rng=rng)
    list_month = list_date.dt.to_period("M")

    # --- 4. rel_price_premium, computed exactly as Phase 2 will compute it ---
    realized_ppsf = original_list_price / living_area_sqft
    cells = pd.DataFrame(
        {
            "submarket": units["submarket"].to_numpy(),
            "list_month": list_month.to_numpy(),
            "list_ppsf": realized_ppsf,
        }
    )
    cell_median = cells.groupby(["submarket", "list_month"], observed=True)[
        "list_ppsf"
    ].transform("median")
    rel_price_premium = (realized_ppsf / cell_median - 1.0).to_numpy()

    # --- 5. Weibull hazard ---------------------------------------------------
    censor_mean_days = prof.censor_mean_horizons * horizon_days
    base_hazard = _solve_base_hazard(
        sold_target, horizon_days, weibull_shape, censor_mean_days
    )
    sub_names = sorted({str(s) for s in units["submarket"]})
    sub_effect = {
        s: float(rng.normal(0.0, _SUBMARKET_LOG_HAZARD_SD)) for s in sub_names
    }
    month_names = sorted({str(m) for m in list_month})
    month_effect = {m: float(rng.normal(0.0, _MONTH_LOG_HAZARD_SD)) for m in month_names}

    price_signal = rel_price_premium if hazard_basis == "realized" else aggressiveness
    log_rate = (
        np.log(base_hazard)
        + np.array([sub_effect[str(s)] for s in units["submarket"]])
        + np.array([month_effect[str(m)] for m in list_month])
        + true_beta_price * price_signal
    )
    rate = np.exp(log_rate)

    u = rng.uniform(1e-12, 1.0, size=n)
    time_to_sale = ((-np.log(u)) ** (1.0 / weibull_shape)) / rate

    # Independent right-censoring, capped administratively at the horizon. A
    # slice of listings is allowed to run past it, because real ones do.
    overrun = rng.uniform(size=n) < prof.overrun_share
    cap = np.where(overrun, _MAX_CENSOR_DAYS, float(horizon_days))
    censor_time = np.minimum(rng.exponential(censor_mean_days, size=n), cap)

    # --- 6. Outcomes ---------------------------------------------------------
    sold = time_to_sale <= censor_time
    duration_days = np.where(sold, time_to_sale, censor_time)
    duration_days = np.maximum(np.round(duration_days), 1.0)

    # A slice of events is still under contract at export time: status PENDING
    # with a pending_date but no close. MLS_SCHEMA §3 counts these as events.
    still_pending = sold & (rng.uniform(size=n) < prof.pending_share_of_events)
    closed = sold & ~still_pending

    concession = _BASE_CONCESSION + _CONCESSION_SLOPE * np.maximum(aggressiveness, 0.0)
    close_price = np.where(
        closed, np.round(original_list_price * (1.0 - concession), -3), np.nan
    )

    escrow_days = rng.integers(30, 61, size=n)
    pending_date = list_date + pd.to_timedelta(duration_days, unit="D")
    close_date = pending_date + pd.to_timedelta(escrow_days, unit="D")
    off_market_date = list_date + pd.to_timedelta(duration_days, unit="D")

    censored_labels = list(_CENSORED_STATUS_SPLIT.keys())
    censored_probs = np.array(list(_CENSORED_STATUS_SPLIT.values()), dtype=float)
    censored_probs = censored_probs / censored_probs.sum()
    censored_status = rng.choice(censored_labels, size=n, p=censored_probs)
    status = np.where(closed, "Closed", np.where(still_pending, "Pending", censored_status))

    cut = rng.uniform(*_PRICE_CUT_RANGE, size=n) * (rng.uniform(size=n) < _PRICE_CUT_PROB)
    last_list_price = np.round(original_list_price * (1.0 - cut), -3)

    list_year = list_date.dt.year.to_numpy()
    year_built = units["year_built"].to_numpy()
    hoa_monthly = np.round(living_area_sqft * rng.uniform(0.8, 1.8, size=n), 0)

    # --- 7. Assemble the export ---------------------------------------------
    def _iso(series: pd.Series, keep: np.ndarray) -> list[str | None]:
        return [
            ts.strftime("%Y-%m-%d") if k else None
            for ts, k in zip(series, keep, strict=True)
        ]

    # Unit numbers and the floor blackout are resolved before the frame is built,
    # because the address may have to carry the unit (see below).
    unit_number = [f"{int(f)}{int(rng.integers(1, 10)):02d}" for f in floor]
    unit_floor: list[int | None] = [int(f) for f in floor]
    floor_missing = rng.uniform(size=n) < float(rates.get("floor", 0.0))
    for i in np.flatnonzero(floor_missing):
        # Blanking a floor means blanking both carriers: an ambiguous two-digit
        # unit number is what an export looks like when the floor is unknowable.
        unit_floor[i] = None
        unit_number[i] = str(int(rng.integers(10, 100)))

    street_address = units["street_address"].to_numpy()
    if "unit_number" in prof.absent_fields:
        # No unit column: the unit rides inside the address, Matrix style, and
        # extract_unit_from_address has to recover it.
        street_address = np.array(
            [f"{addr} {unit}, MIAMI FL" for addr, unit in zip(street_address, unit_number)]
        )

    frame = pd.DataFrame(
        {
            "mls_number": [f"S{seed:04d}{i:06d}" for i in range(n)],
            "status": status,
            "original_list_price": original_list_price,
            "list_date": list_date.dt.strftime("%Y-%m-%d"),
            "living_area_sqft": living_area_sqft,
            "zip_code": units["zip_code"].to_numpy(),
            "close_price": close_price,
            "close_date": _iso(close_date, closed),
            "last_list_price": last_list_price,
            "days_on_market": duration_days.astype(int),
            "off_market_date": _iso(off_market_date, ~sold),
            "beds": beds,
            "baths_full": baths_full,
            "street_address": street_address,
            "unit_number": unit_number,
            "building_name": units["building_name"].to_numpy(),
            "property_type": "Condominium",
            "sale_type": "Standard",
            "unit_floor": unit_floor,
            "cumulative_days_on_market": duration_days.astype(int),
            "pending_date": _iso(pending_date, sold),
            "baths_half": baths_half,
            "year_built": year_built,
            "new_construction": np.where(list_year - year_built <= 2, "Y", "N"),
            "total_stories": total_stories,
            "hoa_monthly": hoa_monthly,
            "hoa_frequency": "Monthly",
            "tax_annual": np.round(original_list_price * rng.uniform(0.015, 0.022, size=n), 0),
            "subdivision": units["building_name"].to_numpy(),
            "city": "Miami",
            "list_agent_name": "Synthetic Agent",
            "list_office_name": "Synthetic Brokerage",
            "waterfront": np.where(rng.uniform(size=n) < 0.35, "Y", "N"),
            "view_description": view,
        }
    )

    # --- 8. Missingness, contamination, and field availability ---------------
    for column in ("hoa_monthly", "year_built", "baths_half"):
        rate_i = float(rates.get(column, 0.0))
        if rate_i > 0:
            frame.loc[rng.uniform(size=n) < rate_i, column] = None

    # Contamination the cleaning filters are supposed to catch. Only possible
    # where the profile actually carries the column that marks it.
    if prof.distressed_share > 0 and "sale_type" not in prof.absent_fields:
        hit = rng.uniform(size=n) < prof.distressed_share
        frame.loc[hit, "sale_type"] = rng.choice(
            ["Short Sale", "Foreclosure", "REO"], size=int(hit.sum())
        )
    if prof.rental_share > 0 and "property_type" not in prof.absent_fields:
        hit = rng.uniform(size=n) < prof.rental_share
        frame.loc[hit, "property_type"] = "Residential Rental"

    frame = frame.drop(columns=list(prof.absent_fields), errors="ignore")

    truth = SyntheticTruth(
        beta_price=true_beta_price,
        submarket_base_ppsf=base_ppsf,
        floor_premium_alpha=_FLOOR_ALPHA,
        floor_premium_gamma=_FLOOR_GAMMA,
        view_premium=view_premium,
        size_elasticity=_SIZE_ELASTICITY,
        fair_value_noise_sd=prof.fair_value_noise_sd,
        building_quality_share=quality_share,
        aggressiveness_sd=aggressiveness_sd,
        weibull_shape=weibull_shape,
        base_hazard_daily=base_hazard,
        target_sold_share=sold_target,
        horizon_days=horizon_days,
        hazard_basis=hazard_basis,
        profile=prof.name,
        view_observable="view_description" not in prof.absent_fields,
        submarket_log_hazard=sub_effect,
        month_log_hazard=month_effect,
    )

    latent = pd.DataFrame(
        {
            "mls_number": frame["mls_number"],
            "submarket": units["submarket"].to_numpy(),
            "building_name": units["building_name"].to_numpy(),
            "aggressiveness": aggressiveness,
            "fair_value_noise": eps,
            "building_quality": prof.fair_value_noise_sd
            * np.sqrt(quality_share)
            * z_building,
            "unit_quality": prof.fair_value_noise_sd
            * np.sqrt(1.0 - quality_share)
            * z_unit,
            "fair_ppsf": fair_ppsf,
            "list_ppsf": realized_ppsf,
            "cell_median_ppsf": cell_median.to_numpy(),
            "rel_price_premium": rel_price_premium,
            "hazard_price_signal": price_signal,
            "time_to_sale_days": time_to_sale,
            "censor_time_days": censor_time,
            "duration_days": duration_days,
            "event_sold": sold.astype(int),
            "still_pending": still_pending.astype(int),
        }
    )

    logger.info(
        "Generated %d synthetic listings (profile=%s, beta_price=%.3f, event share=%.3f)",
        n,
        prof.name,
        true_beta_price,
        float(sold.mean()),
    )
    return SyntheticMLS(frame=frame, truth=truth, latent=latent)


def write_synthetic_mls(result: SyntheticMLS, path: Path) -> Path:
    """Write the synthetic export to CSV so it can be ingested like a real file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic MLS export")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=-1.6, help="true_beta_price")
    parser.add_argument("--horizon-days", type=int, default=180)
    parser.add_argument("--market", default="miami")
    parser.add_argument("--profile", default="rich", choices=sorted(PROFILES))
    parser.add_argument("--hazard-basis", default="realized", choices=["realized", "latent"])
    parser.add_argument(
        "--building-quality-share",
        type=float,
        default=None,
        help="share of fair-value noise variance placed at the building (default: profile)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = generate_synthetic_mls(
        n=args.n,
        seed=args.seed,
        true_beta_price=args.beta,
        horizon_days=args.horizon_days,
        market=args.market,
        profile=args.profile,
        hazard_basis=args.hazard_basis,
        building_quality_share=args.building_quality_share,
    )
    out = args.out or (
        _BACKEND_ROOT / "data" / "synthetic" / f"{args.market}_synth_{args.profile}.csv"
    )
    write_synthetic_mls(result, out)

    event_share = float(result.latent["event_sold"].mean())
    pending_share = float(result.latent["still_pending"].mean())
    print(f"Wrote {out}")
    print(f"  rows: {len(result.frame)}  profile: {result.truth.profile}")
    print(f"  planted beta_price: {result.truth.beta_price} ({result.truth.hazard_basis})")
    print(f"  view observable: {result.truth.view_observable}")
    print(f"  building share of quality variance: {result.truth.building_quality_share:.2f}")
    print(f"  base hazard: {result.truth.base_hazard_daily:.6f} /day")
    print(f"  event share: {event_share:.3f} (PENDING {pending_share:.3f})")
    print(
        f"  rel_price_premium: mean={result.latent['rel_price_premium'].mean():+.4f} "
        f"sd={result.latent['rel_price_premium'].std():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
