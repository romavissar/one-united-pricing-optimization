"""Cleaning filters for normalized MLS frames.

Drops distressed / wrong property types and bad durations; flags outliers.
Never silently imputes missing fields.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import MarketConfig
from src.data.normalize import is_missing

logger = logging.getLogger(__name__)

# Physical plausibility band for a derived $/sqft, in USD per sqft. A bound on
# what a price *can* be, not an estimate of what it is — the same kind of gate
# `coerce_area` already applies to raw area, moved onto the derived quantity.
#
# The band is deliberately wide. Genuine Miami trophy listings reach five
# figures per sqft: the Brickell penthouse at $10,145/sqft in the current export
# is real and stays. What it rejects is arithmetic that cannot describe a
# dwelling — a $49.5M listing recorded as a 798 sqft one-bedroom, giving
# $62,030/sqft. In the export the sample jumps from $6,090/sqft at the 99.9th
# percentile to $53,866 at the 99.99th; the gap is the boundary between listings
# and typos.
#
# Why it matters beyond tidiness: `list_ppsf` is the numerator of
# `rel_price_premium`, so one corrupted area produces a premium of +109 (priced
# 110x its comps). That single value becomes the upper end of the fitted
# `premium_support`, and the extrapolation guard then cannot flag any price at
# all, because nothing is outside a range that runs to +10,900%.
#
# Per AGENTS.md §2 the row is not dropped: the derived price is nulled and
# marked, so the listing keeps every field that is still trustworthy.
_PPSF_PLAUSIBLE_BOUNDS = (100.0, 20_000.0)


@dataclass
class CleanReport:
    """Counts of rows dropped or flagged during cleaning."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    outlier_flagged: int = 0
    list_ppsf_implausible: int = 0
    close_ppsf_implausible: int = 0


def _norm_token(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_true_flag(value: Any) -> bool | None:
    """Read an MLS boolean column. Blank stays None, never False.

    `Short Sale` is blank on 974 rows of the quarterly export. Treating blank as
    False would silently assert those are not short sales; treating it as True
    would drop them. It stays None and the filter leaves the row alone, which is
    the only reading the data supports.
    """
    if is_missing(value):
        return None
    token = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if token in {"true", "t", "y", "yes", "1"}:
        return True
    if token in {"false", "f", "n", "no", "0"}:
        return False
    return None


def _is_distressed(sale_type: Any, exclude: list[str]) -> bool:
    token = _norm_token(sale_type)
    if not token:
        return False
    for ex in exclude:
        ex_n = _norm_token(ex)
        if ex_n and ex_n in token:
            return True
    # Common distressed phrases even if not in config list.
    for phrase in ("foreclosure", "shortsale", "auction", "reo", "bankowned"):
        if phrase in token:
            return True
    return False


def _is_allowed_property(prop_type: Any, allowed: list[str]) -> bool:
    if not allowed:
        return True
    token = _norm_token(prop_type)
    if not token:
        # Unknown property type: keep (IMPORTANT but not REQUIRED); filter only clear mismatches.
        return True
    for a in allowed:
        a_n = _norm_token(a)
        if a_n and a_n in token:
            return True
    # Broader condo synonyms.
    if any(x in token for x in ("condo", "condominium", "coop", "cooperative")):
        return True
    # Explicit rentals / leases.
    if any(x in token for x in ("rent", "lease", "apartmentrental")):
        return False
    return False


def _is_rental(prop_type: Any, sale_type: Any) -> bool:
    for v in (prop_type, sale_type):
        t = _norm_token(v)
        if any(x in t for x in ("rent", "lease", "forrent")):
            return True
    return False


def clean_mls(df: pd.DataFrame, config: MarketConfig) -> tuple[pd.DataFrame, CleanReport]:
    """Filter and flag a normalized MLS frame.

    Drops: distressed sale types, rentals/leases, wrong property types,
    duplicate mls_number (keep first), duration_days <= 0 or > 1095.
    Flags (keeps): list_ppsf beyond ±3σ within submarket×quarter.

    list_ppsf is $/sqft; duration_days is days.
    """
    report = CleanReport(rows_in=len(df))
    out = df.copy()
    filters = config.get("filters") or {}
    exclude_sale = list(filters.get("exclude_sale_types") or [])
    allowed_props = list(filters.get("property_types") or [])
    min_list = filters.get("min_list_price")

    def _drop(mask: pd.Series, reason: str) -> None:
        nonlocal out
        n = int(mask.sum())
        if n:
            report.dropped_by_reason[reason] = report.dropped_by_reason.get(reason, 0) + n
            out = out.loc[~mask].copy()

    if "sale_type" in out.columns:
        _drop(out["sale_type"].map(lambda v: _is_distressed(v, exclude_sale)), "distressed_sale_type")

    # Explicit distressed markers. The old export carried no `sale_type` at all,
    # so the filter above dropped zero rows and every foreclosure and short sale
    # in the pull sat inside the hedonic surface. The quarterly re-pull carries
    # REO and Short Sale as their own booleans, which makes the filter real.
    # A distressed sale is a different transaction: the seller is a lender or is
    # constrained by one, the price is set by a payoff rather than by the market,
    # and the time to sale is driven by lienholder approval. Leaving them in
    # would put that process inside a coefficient meant to measure buyer demand.
    for column, reason in (("is_reo", "reo"), ("is_short_sale", "short_sale")):
        if column not in out.columns:
            continue
        # `.eq(True)` rather than `.fillna(False).astype(bool)`: the mapped
        # column is object dtype holding True / False / None, and filling it
        # triggers pandas' object-downcasting deprecation. Comparing to True
        # gives the same answer — only an explicit True drops the row, a blank
        # never does — without the future breakage.
        flag = out[column].map(_is_true_flag)
        _drop(flag.eq(True), f"distressed_{reason}")

    if "property_type" in out.columns or "sale_type" in out.columns:
        rental_mask = out.apply(
            lambda r: _is_rental(r.get("property_type"), r.get("sale_type")),
            axis=1,
        )
        _drop(rental_mask, "rental_or_lease")

    if "property_type" in out.columns and allowed_props:
        bad_prop = ~out["property_type"].map(lambda v: _is_allowed_property(v, allowed_props))
        # Only drop when property_type is present and clearly not allowed.
        present = out["property_type"].notna() & (out["property_type"].astype(str).str.strip() != "")
        _drop(bad_prop & present, "wrong_property_type")

    if min_list is not None and "original_list_price" in out.columns:
        low = out["original_list_price"].notna() & (out["original_list_price"] < float(min_list))
        _drop(low, "below_min_list_price")

    if "mls_number" in out.columns:
        dup = out["mls_number"].notna() & out["mls_number"].duplicated(keep="first")
        _drop(dup, "duplicate_mls_number")

    if "duration_days" in out.columns:
        bad_dur = out["duration_days"].notna() & (
            (out["duration_days"] <= 0) | (out["duration_days"] > 1095)
        )
        _drop(bad_dur, "bad_duration")

    # Physical plausibility of the derived $/sqft. Runs before the ±3σ flagging
    # below, because a $62,030/sqft typo inflates its own cell's mean and sd and
    # would otherwise hide inside the band it created.
    low, high = _PPSF_PLAUSIBLE_BOUNDS
    for column, counter in (("list_ppsf", "list_ppsf_implausible"),
                            ("close_ppsf", "close_ppsf_implausible")):
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        bad = values.notna() & ~values.between(low, high)
        source = pd.Series("missing", index=out.index, dtype="object")
        source.loc[values.notna()] = "computed"
        source.loc[bad] = "implausible"
        out[f"{column}_source"] = source
        if bad.any():
            out.loc[bad, column] = np.nan
            setattr(report, counter, int(bad.sum()))
            logger.warning(
                "Nulled %d %s values outside $%.0f-$%.0f/sqft as physically "
                "implausible (marked %s_source='implausible'). These are almost "
                "always a wrong living_area_sqft against a correct price; the "
                "row is kept, only the derived price is dropped.",
                int(bad.sum()), column, low, high, column,
            )

    # Outlier flags: ±3σ of list_ppsf within submarket × list_quarter.
    out["list_ppsf_outlier"] = False
    if "list_ppsf" in out.columns and "list_date" in out.columns:
        quarters = out["list_date"].map(
            lambda d: f"{pd.Timestamp(d).year}Q{pd.Timestamp(d).quarter}"
            if d is not None and not pd.isna(d)
            else None
        )
        out["_list_quarter"] = quarters
        outlier = pd.Series(False, index=out.index)
        for _, grp in out.groupby(["submarket", "_list_quarter"], dropna=False):
            vals = grp["list_ppsf"].dropna()
            if len(vals) < 5:
                continue
            mu = float(vals.mean())
            sigma = float(vals.std(ddof=1))
            if sigma == 0 or pd.isna(sigma):
                continue
            lo, hi = mu - 3 * sigma, mu + 3 * sigma
            idx = grp.index[grp["list_ppsf"].notna() & ((grp["list_ppsf"] < lo) | (grp["list_ppsf"] > hi))]
            outlier.loc[idx] = True
        out["list_ppsf_outlier"] = outlier
        report.outlier_flagged = int(outlier.sum())
        if report.outlier_flagged:
            logger.info(
                "Flagged %d list_ppsf outliers (±3σ within submarket×quarter); kept",
                report.outlier_flagged,
            )
        out = out.drop(columns=["_list_quarter"])

    report.rows_out = len(out)
    return out.reset_index(drop=True), report
