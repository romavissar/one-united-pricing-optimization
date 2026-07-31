"""Derived features for demand estimation.

The one that matters is `rel_price_premium`: its coefficient *is* `beta_price`.
It is computed against the median of its own (submarket, list_month) cell, so
that location and macro regime are held fixed and the surviving variation is
the seller's pricing choice. Never compute that median globally or over the
pooled period.

Nothing here imputes. A feature whose inputs are missing comes out null and,
where the definition has a fallback, carries a `*_source` marker naming which
branch produced it.

Units:
- `list_ppsf`, `cell_median_ppsf`: $/sqft.
- `rel_price_premium`: ratio minus one (0.10 = priced 10% above the cell median).
- `hoa_per_sqft`: $/sqft per month.
- `duration_days`: days (computed upstream in normalize).
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import MarketConfig, load_market_config
from src.data.ingest_mls import ingest_mls
from src.data.normalize import is_missing
from src.data.premium import PremiumFit, fit_price_premium
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

# Phase 3 regresses on exactly these; their null rates decide the usable sample.
DEMAND_COVARIATES = (
    "rel_price_premium",
    "log_floor",
    "living_area_sqft",
    "beds",
    "hoa_per_sqft",
    "is_new_construction",
    "submarket",
    "season",
    "inventory_competition",
)

_FLOOR_BUCKET_EDGES = ((1, 5, "low"), (6, 12, "mid"), (13, 25, "high"))
_FLOOR_BUCKET_TOP = "tower"
_SIZE_BUCKET_COUNT = 4

# Sanity band for monthly HOA in $/sqft, not an estimate. Miami condo dues run
# roughly $0.6-$2.5/sqft/month; above the ceiling the figure is almost always an
# annual fee in a column labelled monthly, which is the units confusion
# AGENTS.md calls the most likely silent numeric bug in this codebase.
_HOA_PSF_MONTHLY_BOUNDS = (0.05, 5.0)

_NEW_CONSTRUCTION_MAX_AGE_YEARS = 2

# PROJECT_BRIEF §3d: below this, sellers priced identically and elasticity is
# not identified at any sample size.
_REL_PREMIUM_IQR_NARROW = 0.03
# Above this, the cell is not holding the unit fixed. Pure pricing behaviour at
# a plausible spread produces an IQR near 0.11; three times that means a
# studio and a penthouse are sharing a cell and the coefficient will pick up
# quality, not elasticity.
_REL_PREMIUM_IQR_WIDE = 0.35

_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


@dataclass
class FeatureReport:
    """Feature-build diagnostics, including the identification smoke signal."""

    rows: int = 0
    median_basis_counts: dict[str, int] = field(default_factory=dict)
    penthouses_without_floor: int = 0
    rel_price_premium: dict[str, float] = field(default_factory=dict)
    corr_rel_price_premium_event_sold: float | None = None
    null_rates: dict[str, float] = field(default_factory=dict)
    hoa_units_suspect: int = 0
    new_construction_sources: dict[str, int] = field(default_factory=dict)
    usable_rows: int = 0
    hedonic_premium: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FeatureResult:
    """Feature frame plus its report."""

    frame: pd.DataFrame
    report: FeatureReport


def _season(list_date: pd.Series) -> pd.Series:
    """Meteorological season from list_date.

    Deliberately neutral: Miami's snowbird seasonality is real, but encoding it
    as high/shoulder/low would presuppose the effect the model is meant to fit.
    """
    return list_date.map(
        lambda d: None if d is None or pd.isna(d) else _SEASON_BY_MONTH[pd.Timestamp(d).month]
    )


def _floor_bucket(floor: Any) -> str | None:
    if is_missing(floor):
        return None
    try:
        value = int(float(floor))
    except (TypeError, ValueError):
        return None
    if value < 1:
        return None
    for lo, hi, label in _FLOOR_BUCKET_EDGES:
        if lo <= value <= hi:
            return label
    return _FLOOR_BUCKET_TOP


def floor_bucket(floor: Any) -> str | None:
    """Coarse floor band (`low`/`mid`/`high`/`tower`), or None if unparseable.

    Public because the optimizer groups comparable units by it: two units on
    adjacent floors of the same line are the same product, one on floor 3 and
    one on floor 40 are not.
    """
    return _floor_bucket(floor)


def season_of(date: Any) -> str | None:
    """Meteorological season of a single date, or None. Units: none."""
    if date is None or pd.isna(date):
        return None
    return _SEASON_BY_MONTH[pd.Timestamp(date).month]


def _size_bucket(frame: pd.DataFrame) -> pd.Series:
    """Quartile of living_area_sqft *within* submarket. Null where inputs are."""
    out = pd.Series(pd.NA, index=frame.index, dtype="object")
    if "living_area_sqft" not in frame.columns or "submarket" not in frame.columns:
        return out
    for submarket, group in frame.groupby("submarket", dropna=True):
        area = group["living_area_sqft"].dropna()
        if len(area) < _SIZE_BUCKET_COUNT or area.nunique() < _SIZE_BUCKET_COUNT:
            continue
        try:
            buckets = pd.qcut(
                area,
                _SIZE_BUCKET_COUNT,
                labels=[f"q{i + 1}" for i in range(_SIZE_BUCKET_COUNT)],
                duplicates="drop",
            )
        except ValueError:
            logger.info("size_bucket skipped for submarket %s (degenerate areas)", submarket)
            continue
        out.loc[buckets.index] = buckets.astype(str)
    return out


def _relative_price_premium(
    frame: pd.DataFrame, min_cell_listings: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (rel_price_premium, cell_median_ppsf, median_basis).

    Ladder: (submarket, list_month) when the cell holds at least
    `min_cell_listings`, else (submarket, list_quarter), else null. A thin
    quarter cell yields `insufficient` rather than a pooled median — a median
    borrowed from elsewhere is an imputation, and a wrong denominator here
    corrupts the only coefficient this project exists to estimate.
    """
    index = frame.index
    ppsf = frame["list_ppsf"]
    usable = ppsf.notna() & (ppsf > 0) & frame["submarket"].notna() & frame["list_month"].notna()

    month_key = frame["submarket"].astype(str) + "|" + frame["list_month"].astype(str)
    quarter_key = frame["submarket"].astype(str) + "|" + frame["list_quarter"].astype(str)

    median = pd.Series(np.nan, index=index, dtype="float64")
    basis = pd.Series("missing", index=index, dtype="object")

    for key, label in ((month_key, "submarket_month"), (quarter_key, "submarket_quarter")):
        if not (usable & (basis == "missing")).any():
            break
        # Median and count come from every usable listing in the cell, not just
        # the rows still awaiting a basis. Grouping only the leftovers would
        # compute the quarterly median over the subsample that happened to sit
        # in thin months, which is a biased denominator for the one variable
        # whose coefficient is beta_price.
        grouped = ppsf[usable].groupby(key[usable])
        medians, counts = grouped.median(), grouped.size()
        dense = (
            usable
            & (basis == "missing")
            & key.map(counts).ge(min_cell_listings).fillna(False)
        )
        median.loc[dense] = key[dense].map(medians)
        basis.loc[dense] = label

    basis.loc[usable & (basis == "missing")] = "insufficient"
    rel = ppsf / median - 1.0
    return rel, median, basis


def _inventory_competition(frame: pd.DataFrame) -> pd.Series:
    """Other listings *entering* the same (submarket, list_month). Self excluded.

    This counts new supply arriving, not standing inventory, and the distinction
    is the whole point. An earlier version counted every listing whose live
    interval [list_date, list_date + duration_days] overlapped the month. That
    made the covariate a function of other listings' durations, so a slow month
    looked crowded precisely *because* its listings were slow — reverse
    causality inside a regressor. On synthetic data where the generator plants
    no inventory effect at all, it produced a positive coefficient in six of
    eight seeds and a significant one in one of them, which is a hard sign-check
    failure on a model that is otherwise correct.

    Entry counts are fixed before any duration is realized, so they cannot
    inherit the outcome. What they give up is unsold standing inventory, which a
    developer competing against a half-empty tower next door would rightly care
    about. Recovering it without the endogeneity needs an as-of-month-start
    active count built from listing dates alone; see README "Known gaps".

    Counts are sample-relative, so months at the edges of the export understate.
    """
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    if not {"submarket", "list_date"}.issubset(frame.columns):
        return out

    keys: list[str | None] = []
    for submarket, start in zip(frame["submarket"], frame["list_date"], strict=True):
        if submarket is None or pd.isna(submarket) or start is None or pd.isna(start):
            keys.append(None)
            continue
        keys.append(f"{submarket}|{pd.Timestamp(start).to_period('M')}")

    key_series = pd.Series(keys, index=frame.index, dtype="object")
    counts = key_series.value_counts()
    known = key_series.notna()
    out.loc[known] = key_series[known].map(counts).astype("float64") - 1.0
    return out


def _hoa_per_sqft(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Monthly HOA in $/sqft, plus a source marker.

    The band that used to null values outside $0.05-$5.00/sqft/month is gone.
    It existed because the old export carried no frequency column, so a figure
    that looked annual could not be distinguished from a high monthly fee, and
    238 rows were nulled on suspicion. The quarterly export settles the
    question: `Association Fee` overlaps `Maintenance Charge/Month` on 6,943
    rows with a **median ratio of exactly 1.00** (medians $1,130 against
    $1,290), and only 0.3% of the overlap sits near a ratio of 12. The field is
    monthly, so there is nothing to correct and no basis for discarding the
    tails — a $6/sqft/month fee in a full-service oceanfront tower is a real
    fee, and nulling it was discarding the amenity signal it carries.

    What remains is arithmetic: a fee needs a positive area to become a rate,
    and a negative fee is not a fee.
    """
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    source = pd.Series("missing", index=frame.index, dtype="object")
    if "hoa_monthly" not in frame.columns or "living_area_sqft" not in frame.columns:
        return values, source

    area = frame["living_area_sqft"]
    hoa = frame["hoa_monthly"]
    computable = hoa.notna() & hoa.ge(0) & area.notna() & (area > 0)
    values.loc[computable] = (hoa[computable] / area[computable]).astype("float64")
    source.loc[computable] = "reported"
    source.loc[hoa.notna() & hoa.lt(0)] = "negative"
    return values, source


def _is_new_construction(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """New-construction flag, preferring the MLS field over the year_built rule.

    Returned as nullable `boolean` so a missing flag stays missing instead of
    collapsing into False when the design matrix is built.
    """
    values = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    source = pd.Series("missing", index=frame.index, dtype="object")

    reported = (
        frame["new_construction"]
        if "new_construction" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="object")
    )
    has_reported = reported.notna()
    values.loc[has_reported] = reported.loc[has_reported].astype(bool)
    source.loc[has_reported] = "reported"

    if "year_built" in frame.columns and "list_date" in frame.columns:
        list_year = frame["list_date"].map(
            lambda d: None if d is None or pd.isna(d) else pd.Timestamp(d).year
        )
        derivable = ~has_reported & frame["year_built"].notna() & list_year.notna()
        if derivable.any():
            age = list_year[derivable].astype(float) - frame.loc[derivable, "year_built"].astype(
                float
            )
            values.loc[derivable] = (age <= _NEW_CONSTRUCTION_MAX_AGE_YEARS).astype(bool)
            source.loc[derivable] = "derived"
    return values, source


# Multi-valued MLS fields, and the minimum share of listings a token must reach
# before it earns an indicator column. Below that the level is a handful of rows
# and the coefficient on it is noise that costs a degree of freedom.
_TOKEN_FIELDS: dict[str, str] = {
    "view_description": "view",
    "waterfront_description": "wf",
    "parking_description": "park",
    "restrictions": "rest",
}
_TOKEN_MIN_SHARE = 0.01


def tokenize_multivalue(series: pd.Series, prefix: str, *, min_share: float = _TOKEN_MIN_SHARE) -> pd.DataFrame:
    """One indicator column per atomic token in a comma-separated MLS field.

    `Unit View` reads `"Bay, Skyline View, Water View"` — three facts about the
    unit, not one categorical level. Treated as a category it has 1,144 distinct
    values on this export and is useless; split into atoms it has 17, of which
    `Direct Ocean` (10.7% of listings) is a different and far more valuable
    thing from `Ocean View` (29.9%). That distinction is the amenity the old
    export could not see at all, and for a Miami tower it is plausibly the
    largest single premium in the building.

    Missingness is encoded, not dropped: a row with no value at all gets a
    `{prefix}_missing` flag and zeros elsewhere, so the listing keeps its place
    in the sample instead of being deleted by listwise deletion. That matters
    most for the fields that are only partly filled — dropping rows on a 63%-
    filled column would cost more sample than the column is worth.
    """
    present = series.notna() & series.astype(str).str.strip().ne("")
    exploded = (
        series.where(present)
        .astype("string")
        .str.split(",")
        .apply(lambda parts: [p.strip() for p in parts] if isinstance(parts, list) else [])
    )
    counts: dict[str, int] = {}
    for parts in exploded:
        for token in set(parts):
            if token:
                counts[token] = counts.get(token, 0) + 1

    threshold = max(1, int(min_share * len(series)))
    kept = sorted(t for t, n in counts.items() if n >= threshold)

    out = pd.DataFrame(index=series.index)
    for token in kept:
        column = f"{prefix}_{re.sub(r'[^a-z0-9]+', '_', token.lower()).strip('_')}"
        out[column] = exploded.apply(lambda parts, t=token: float(t in parts))
    out[f"{prefix}_missing"] = (~present).astype("float64")
    return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ok = numerator.notna() & denominator.notna() & (denominator != 0)
    out = pd.Series(np.nan, index=numerator.index, dtype="float64")
    out.loc[ok] = numerator[ok].astype(float) / denominator[ok].astype(float)
    return out


def build_features(
    frame: pd.DataFrame,
    config: MarketConfig,
    *,
    min_cell_listings: int | None = None,
) -> FeatureResult:
    """Derive demand-model features from a normalized, cleaned MLS frame.

    Args:
        frame: output of `normalize_mls` + `clean_mls`.
        config: market config; supplies `min_cell_listings`.
        min_cell_listings: overrides the config threshold for cell density.

    Raises:
        SchemaError: when the frame lacks columns normalize is contracted to
            provide.
    """
    missing = [c for c in ("list_ppsf", "submarket", "list_date") if c not in frame.columns]
    if missing:
        raise SchemaError(f"Feature build requires normalized columns, missing: {missing}")

    min_cell = int(
        min_cell_listings
        if min_cell_listings is not None
        else (config.get("defaults") or {}).get("min_cell_listings") or 8
    )
    out = frame.copy()

    out["list_month"] = out["list_date"].map(
        lambda d: None if d is None or pd.isna(d) else str(pd.Timestamp(d).to_period("M"))
    )
    out["list_quarter"] = out["list_date"].map(
        lambda d: None if d is None or pd.isna(d) else str(pd.Timestamp(d).to_period("Q"))
    )
    out["season"] = _season(out["list_date"])

    rel, cell_median, basis = _relative_price_premium(out, min_cell)
    out["cell_median_ppsf"] = cell_median
    out["rel_price_premium"] = rel
    out["median_basis"] = basis

    # A hedonic surface is multiplicative in area, so log area is the correctly
    # specified control; raw sqft leaves curvature in the residual, and that
    # residual is exactly the quality contamination that attenuates beta_price.
    out["log_living_area"] = out["living_area_sqft"].map(
        lambda a: None if a is None or pd.isna(a) or float(a) <= 0 else float(np.log(float(a)))
    )

    floor = out["floor"] if "floor" in out.columns else pd.Series(np.nan, index=out.index)
    out["log_floor"] = floor.map(
        lambda f: None if f is None or pd.isna(f) or float(f) < 0 else float(np.log(float(f) + 1.0))
    )
    out["floor_bucket"] = floor.map(_floor_bucket)
    out["size_bucket"] = _size_bucket(out)

    baths = out["baths_full"].fillna(0) + out.get(
        "baths_half", pd.Series(0.0, index=out.index)
    ).fillna(0) * 0.5
    beds_floor = out["beds"].map(lambda b: np.nan if b is None or pd.isna(b) else max(float(b), 1.0))
    out["bed_bath_ratio"] = _safe_ratio(baths.where(out["baths_full"].notna()), beds_floor)

    out["hoa_per_sqft"], out["hoa_per_sqft_source"] = _hoa_per_sqft(out)
    out["is_new_construction"], out["is_new_construction_source"] = _is_new_construction(out)

    out["sold_to_list_ratio"] = _safe_ratio(
        out["close_price"] if "close_price" in out.columns else pd.Series(np.nan, index=out.index),
        out["original_list_price"],
    )
    out["price_cut_pct"] = (
        _safe_ratio(
            out["last_list_price"]
            if "last_list_price" in out.columns
            else pd.Series(np.nan, index=out.index),
            out["original_list_price"],
        )
        - 1.0
    )
    out["inventory_competition"] = _inventory_competition(out)

    # Multi-valued amenity fields → indicator blocks.
    token_columns: dict[str, list[str]] = {}
    for source_column, prefix in _TOKEN_FIELDS.items():
        if source_column not in out.columns or not out[source_column].notna().any():
            continue
        block = tokenize_multivalue(out[source_column], prefix)
        for column in block.columns:
            out[column] = block[column]
        token_columns[source_column] = list(block.columns)
    out.attrs["token_columns"] = token_columns

    # Scalar extras. Each keeps an explicit missing level rather than dropping
    # the row: `furnished_info` is 37% filled and `special_assessment` 32%, so
    # listwise deletion on either would cost more sample than the field is worth.
    if "min_lease_days" in out.columns:
        lease = pd.to_numeric(out["min_lease_days"], errors="coerce")
        out["min_lease_days_value"] = lease.fillna(0.0)
        out["min_lease_days_missing"] = lease.isna().astype("float64")
    for column in ("furnished_info", "special_assessment", "association_type"):
        if column in out.columns and out[column].notna().any():
            out[f"{column}_level"] = (
                out[column].astype("string").fillna("__missing__").str.strip()
            )

    # Building age, an input the list-price hedonic wants and nothing else built.
    if "year_built" in out.columns and "list_date" in out.columns:
        list_year = out["list_date"].map(
            lambda d: np.nan if is_missing(d) else float(pd.Timestamp(d).year)
        )
        built = pd.to_numeric(out["year_built"], errors="coerce")
        age = list_year - built
        out["building_age_years"] = age.where(age.between(-5, 150))

    # The respecified identification variable: the part of the asking price that
    # the unit, its submarket and its month do not explain. Computed alongside
    # the cell-median version rather than replacing it, so the two can be fitted
    # against each other and the change in beta_price decomposed.
    out["cell_median_premium"] = out["rel_price_premium"]
    premium_fit = None
    try:
        premium = fit_price_premium(out)
        out["hedonic_price_premium"] = premium.premium_log
        out["hedonic_price_premium_ratio"] = premium.premium_ratio
        out["hedonic_reference_ppsf"] = premium.reference_ppsf
        out["hedonic_price_premium_source"] = premium.source
        premium_fit = premium.fit
        out.attrs["premium_model"] = premium.model
        # Ship it. `rel_price_premium` is the name the whole downstream stack
        # reads, so the respecification takes effect by replacing what that name
        # means rather than by threading a new column through six modules. The
        # cell-median version stays beside it under its own name so the two can
        # still be compared, and `cell_median_ppsf` — the denominator the
        # optimizer divides a candidate price by — moves with it, because a
        # premium measured against one reference and a coefficient fitted
        # against another are not the same quantity.
        out["cell_median_ppsf_raw"] = out["cell_median_ppsf"]
        keep = premium.premium_ratio.notna() & premium.reference_ppsf.notna()
        out["rel_price_premium"] = premium.premium_ratio.where(keep)
        out["cell_median_ppsf"] = premium.reference_ppsf.where(keep)
        out["rel_price_premium_spec"] = np.where(keep, "hedonic_residual", "unavailable")
    except (SchemaError, IdentificationError) as exc:
        # Loud, not silent: without this variable the demand model falls back to
        # a regressor known to carry unit quality, and the caller has to know.
        logger.warning(
            "List-price hedonic could not be fitted (%s); rel_price_premium "
            "remains the cell-median version, which carries unit quality as "
            "well as pricing choice.", exc,
        )
        out["hedonic_price_premium"] = np.nan
        out["hedonic_price_premium_ratio"] = np.nan
        out["hedonic_reference_ppsf"] = np.nan
        out["hedonic_price_premium_source"] = "unavailable"

    return FeatureResult(frame=out, report=build_feature_report(out, premium_fit))


def build_feature_report(
    frame: pd.DataFrame, premium_fit: PremiumFit | None = None
) -> FeatureReport:
    """Assemble feature diagnostics, including the identification smoke signal."""
    warnings: list[str] = []
    rows = len(frame)

    basis_counts = {
        str(k): int(v) for k, v in frame["median_basis"].value_counts(dropna=False).items()
    }

    rel = frame["rel_price_premium"].dropna()
    stats: dict[str, float] = {}
    if len(rel):
        stats = {
            "mean": float(rel.mean()),
            "median": float(rel.median()),
            "sd": float(rel.std()),
            "iqr": float(rel.quantile(0.75) - rel.quantile(0.25)),
            "p5": float(rel.quantile(0.05)),
            "p95": float(rel.quantile(0.95)),
        }

    corr: float | None = None
    if "event_sold" in frame.columns:
        pair = frame[["rel_price_premium", "event_sold"]].dropna()
        if len(pair) > 2 and pair["rel_price_premium"].std() > 0 and pair["event_sold"].std() > 0:
            corr = float(np.corrcoef(pair["rel_price_premium"], pair["event_sold"])[0, 1])

    null_rates = {
        col: round(float(frame[col].isna().mean()), 4)
        for col in DEMAND_COVARIATES
        if col in frame.columns
    }

    hoa_suspect = int((frame["hoa_per_sqft_source"] == "implausible").sum())
    nc_sources = {
        str(k): int(v)
        for k, v in frame["is_new_construction_source"].value_counts(dropna=False).items()
    }

    present = [c for c in DEMAND_COVARIATES if c in frame.columns]
    usable = int(frame[present].notna().all(axis=1).sum()) if present and rows else 0

    # A penthouse has no numeric floor even though its floor_source is "parsed",
    # so log_floor is null and the row leaves the fit. These are a developer's
    # highest-value units; the count belongs in the report rather than in a
    # silent sample reduction.
    penthouses_without_floor = 0
    if "is_penthouse" in frame.columns and "log_floor" in frame.columns:
        is_ph = frame["is_penthouse"].fillna(False).astype(bool)
        penthouses_without_floor = int((is_ph & frame["log_floor"].isna()).sum())

    iqr = stats.get("iqr")
    if iqr is not None and iqr < _REL_PREMIUM_IQR_NARROW:
        warnings.append(
            f"PRICING VARIATION TOO NARROW — rel_price_premium IQR is {iqr:.4f}. "
            "Sellers priced near-identically relative to comps; elasticity will be "
            "weakly identified regardless of sample size."
        )
    if iqr is not None and iqr > _REL_PREMIUM_IQR_WIDE:
        warnings.append(
            f"PRICING VARIATION TOO WIDE — rel_price_premium IQR is {iqr:.4f}. "
            "A (submarket, month) cell this dispersed is not holding the unit fixed, "
            "so the variable carries unit quality as well as pricing choice. Its "
            "coefficient is not an elasticity without hedonic controls."
        )

    insufficient = basis_counts.get("insufficient", 0) + basis_counts.get("missing", 0)
    if rows and insufficient / rows > 0.20:
        warnings.append(
            f"THIN CELLS — {insufficient} rows ({insufficient / rows:.1%}) have no usable "
            "(submarket, month) or (submarket, quarter) median and carry a null "
            "rel_price_premium. They cannot enter the demand fit."
        )
    if hoa_suspect:
        warnings.append(
            f"HOA UNITS SUSPECT — {hoa_suspect} rows have an implied HOA outside "
            f"${_HOA_PSF_MONTHLY_BOUNDS[0]}-${_HOA_PSF_MONTHLY_BOUNDS[1]}/sqft/month and were "
            "nulled. Likely annual fees in a monthly column; request hoa_frequency."
        )
    if corr is not None and corr >= 0:
        warnings.append(
            f"SMOKE SIGNAL FAILED — corr(rel_price_premium, event_sold) is {corr:+.4f}, "
            "expected negative. Higher relative price should lower the chance of sale."
        )

    if premium_fit is not None:
        share = premium_fit.explained_share
        warnings.append(
            f"IDENTIFICATION VARIABLE RESPECIFIED — the list-price hedonic explains "
            f"{share:.1%} of the variance in log $/sqft, so that much of the old "
            f"cell-median premium was unit characteristics rather than pricing "
            f"choice. What remains is a residual with sd {premium_fit.residual_sd:.4f} "
            f"log points, orthogonal to every control by construction."
        )

    return FeatureReport(
        rows=rows,
        hedonic_premium=premium_fit.as_dict() if premium_fit else {},
        median_basis_counts=basis_counts,
        penthouses_without_floor=penthouses_without_floor,
        rel_price_premium=stats,
        corr_rel_price_premium_event_sold=corr,
        null_rates=null_rates,
        hoa_units_suspect=hoa_suspect,
        new_construction_sources=nc_sources,
        usable_rows=usable,
        warnings=warnings,
    )


def format_feature_report(report: FeatureReport) -> str:
    """Human-readable feature report (the printed correlation is the deliverable)."""
    lines = [f"FEATURES  rows={report.rows}"]
    lines.append("")
    lines.append("MEDIAN BASIS")
    for k, v in report.median_basis_counts.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("REL_PRICE_PREMIUM")
    if report.rel_price_premium:
        for k, v in report.rel_price_premium.items():
            lines.append(f"  {k}: {v:+.4f}")
    else:
        lines.append("  (no non-null values)")
    corr = report.corr_rel_price_premium_event_sold
    lines.append(
        f"  corr with event_sold: {corr:+.4f}" if corr is not None else "  corr with event_sold: n/a"
    )
    lines.append("")
    lines.append("NULL RATES (Phase 3 covariates)")
    for col, rate in report.null_rates.items():
        lines.append(f"  {col}: {rate}")
    lines.append(f"  usable rows (all covariates non-null): {report.usable_rows}")
    lines.append("")
    lines.append("SOURCES")
    for k, v in report.new_construction_sources.items():
        lines.append(f"  is_new_construction {k}: {v}")
    lines.append(f"  hoa_per_sqft implausible (nulled): {report.hoa_units_suspect}")
    lines.append(
        f"  penthouses with no numeric floor (excluded from fit): "
        f"{report.penthouses_without_floor}"
    )
    if report.warnings:
        lines.append("")
        lines.append("WARNINGS")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build demand features from MLS exports")
    parser.add_argument("--inspect", action="store_true", help="Print the feature report")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--file", type=Path, action="append", default=None)
    parser.add_argument("--market", default="miami")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_market_config(args.market)
    try:
        ingested = ingest_mls(paths=args.file, market=args.market, raw_dir=args.dir, config=config)
        result = build_features(ingested.frame, config)
    except SchemaError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(format_feature_report(result.report))
    return 2 if result.report.warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
