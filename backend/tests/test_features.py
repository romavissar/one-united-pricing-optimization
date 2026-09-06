"""Phase 2 — feature engineering tests.

The acceptance criterion is the smoke signal: on synthetic data
`rel_price_premium` centres on zero and correlates negatively with `event_sold`.
Everything else here guards the two ways this module could quietly produce a
wrong `beta_price`: a median computed over the wrong grouping, or a missing
input silently filled in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config
from src.data.features import (
    DEMAND_COVARIATES,
    build_features,
    format_feature_report,
)
from src.data.ingest_mls import ingest_mls
from src.data.normalize import normalize_mls
from src.data.synth import generate_synthetic_mls
from src.exceptions import SchemaError

_REAL_EXPORT_DIR = "data/raw/mls"


@pytest.fixture(scope="module")
def config():
    return load_market_config("miami")


@pytest.fixture(scope="module")
def rich_features(config):
    frame = normalize_mls(generate_synthetic_mls().frame, config=config).frame
    return build_features(frame, config)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Minimal normalized-shaped frame; overrides are applied per row."""
    base = {
        "list_ppsf": 1000.0,
        "submarket": "brickell",
        "list_date": "2025-01-15",
        "original_list_price": 1_000_000.0,
        "last_list_price": 1_000_000.0,
        "close_price": np.nan,
        "living_area_sqft": 1000.0,
        "beds": 2,
        "baths_full": 2.0,
        "baths_half": 0.0,
        "floor": 10.0,
        "floor_source": "reported",
        "hoa_monthly": 1200.0,
        "year_built": 2020,
        "new_construction": None,
        "duration_days": 30.0,
        "event_sold": 0,
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["list_date"] = pd.to_datetime(frame["list_date"])
    return frame


def test_smoke_signal_on_synthetic_data(rich_features, capsys) -> None:
    """PROJECT_BRIEF Phase 2 acceptance: mean ~ 0 and a negative correlation."""
    report = rich_features.report
    stats = report.rel_price_premium

    # The premium is now a hedonic residual, so it is centred by least squares
    # rather than by construction: its mean is zero to machine precision and its
    # median is near zero but not identically so. The old variable — a ratio to
    # a cell median — had an exactly-zero median and a mean pulled positive by
    # the skew of prices, which is the opposite pattern.
    assert abs(stats["mean"]) < 0.05, stats["mean"]
    assert abs(stats["median"]) < 0.02, stats["median"]

    corr = report.corr_rel_price_premium_event_sold
    assert corr is not None and corr < 0, corr

    print(f"corr(rel_price_premium, event_sold) = {corr:+.4f}")
    assert "corr(rel_price_premium" in capsys.readouterr().out
    assert "PRICING VARIATION TOO NARROW" not in format_feature_report(report)


def test_median_is_computed_within_cell_not_globally(config) -> None:
    """A submarket priced 2x higher must not read as priced 2x above comps."""
    rows = []
    for i in range(10):
        rows.append({"submarket": "brickell", "list_ppsf": 1000.0 + i})
        rows.append({"submarket": "aventura", "list_ppsf": 2000.0 + 2 * i})
    result = build_features(_frame(rows), config, min_cell_listings=8)
    rel = result.frame["rel_price_premium"]

    assert (result.frame["median_basis"] == "submarket_month").all()
    # Both submarkets straddle zero; a pooled median would push one far off it.
    for submarket in ("brickell", "aventura"):
        cell = rel[result.frame["submarket"] == submarket]
        assert abs(float(cell.median())) < 1e-9
        assert cell.abs().max() < 0.02


def test_median_basis_ladder_falls_back_then_gives_up(config) -> None:
    rows = []
    # Dense month: 8 listings in one month clears the threshold outright.
    rows += [{"submarket": "brickell", "list_date": "2025-01-10"} for _ in range(8)]
    # Thin months that together make a dense quarter.
    rows += [{"submarket": "edgewater", "list_date": d} for d in ("2025-04-05",) * 3]
    rows += [{"submarket": "edgewater", "list_date": d} for d in ("2025-05-05",) * 3]
    rows += [{"submarket": "edgewater", "list_date": d} for d in ("2025-06-05",) * 3]
    # Thin everywhere: one lonely listing in its own quarter.
    rows += [{"submarket": "downtown", "list_date": "2025-09-09"}]

    result = build_features(_frame(rows), config, min_cell_listings=8)
    frame = result.frame
    basis = frame.set_index(frame.index)["median_basis"]

    assert (basis[frame["submarket"] == "brickell"] == "submarket_month").all()
    assert (basis[frame["submarket"] == "edgewater"] == "submarket_quarter").all()
    assert (basis[frame["submarket"] == "downtown"] == "insufficient").all()
    # Giving up means a null, never a pooled median borrowed from elsewhere.
    assert frame.loc[frame["submarket"] == "downtown", "rel_price_premium"].isna().all()


def test_quarterly_median_uses_the_whole_quarter(config) -> None:
    """Regression: the fallback median must not be built from the leftovers.

    Grouping only the rows that failed the monthly test computes the quarterly
    median over whichever listings happened to sit in thin months — a biased
    denominator for the one variable whose coefficient is beta_price.
    """
    rows = [{"list_date": "2025-01-10", "list_ppsf": 1000.0 + i} for i in range(10)]
    for i in range(4):
        rows.append({"list_date": "2025-02-10", "list_ppsf": 1000.0 + i})
        rows.append({"list_date": "2025-03-10", "list_ppsf": 1000.0 + i})

    frame = _frame(rows)
    result = build_features(frame, config, min_cell_listings=8)
    out = result.frame

    fallback = out.loc[out["median_basis"] == "submarket_quarter"]
    assert len(fallback) == 8
    true_quarter_median = float(frame["list_ppsf"].median())
    assert float(fallback["cell_median_ppsf"].iloc[0]) == pytest.approx(true_quarter_median)

    # The dense January rows keep their own monthly median, not the quarterly one.
    monthly = out.loc[out["median_basis"] == "submarket_month"]
    assert len(monthly) == 10
    assert float(monthly["cell_median_ppsf"].iloc[0]) == pytest.approx(1004.5)


def test_demand_covariates_are_numeric_or_nullable(rich_features) -> None:
    """Object-dtype covariates break design-matrix construction in Phase 3."""
    frame = rich_features.frame
    assert frame["inventory_competition"].dtype == "float64"
    assert frame["is_new_construction"].dtype == "boolean"
    for col in ("rel_price_premium", "log_floor", "hoa_per_sqft"):
        assert pd.api.types.is_float_dtype(frame[col])


def test_missing_inputs_produce_nulls_not_substitutes(config) -> None:
    rows = [
        {"floor": np.nan, "floor_source": "missing", "hoa_monthly": np.nan},
        {"year_built": np.nan, "new_construction": None},
        {"living_area_sqft": np.nan},
        {"submarket": None},
        {"list_date": None},
    ] + [{} for _ in range(8)]
    frame = build_features(_frame(rows), config, min_cell_listings=8).frame

    assert pd.isna(frame.loc[0, "log_floor"])
    assert pd.isna(frame.loc[0, "floor_bucket"])
    assert pd.isna(frame.loc[0, "hoa_per_sqft"])
    assert frame.loc[0, "hoa_per_sqft_source"] == "missing"

    assert pd.isna(frame.loc[1, "is_new_construction"])
    assert frame.loc[1, "is_new_construction_source"] == "missing"

    assert pd.isna(frame.loc[2, "hoa_per_sqft"])
    assert frame.loc[3, "median_basis"] == "missing"
    assert pd.isna(frame.loc[3, "rel_price_premium"])
    assert pd.isna(frame.loc[4, "season"])


def test_hoa_is_taken_at_face_value_now_the_field_is_confirmed_monthly(config) -> None:
    """The old plausibility band nulled real fees and is gone.

    It existed because the previous export carried no frequency column, so a
    high monthly fee could not be told from an annual one and 238 rows were
    discarded on suspicion. The quarterly export settles it: Association Fee
    overlaps Maintenance Charge/Month on 6,943 rows with a median ratio of
    exactly 1.00. A $6/sqft/month fee in a full-service oceanfront tower is a
    real fee carrying real amenity signal, and nulling it was throwing that
    signal away. What remains is arithmetic — a rate needs a positive area, and
    a negative fee is not a fee.
    """
    rows = [
        {"hoa_monthly": 1500.0, "living_area_sqft": 1000.0},    # $1.50/sqft/mo
        {"hoa_monthly": 18_692.0, "living_area_sqft": 1000.0},  # $18.69, kept
        {"hoa_monthly": 0.0, "living_area_sqft": 1000.0},       # a genuine $0
        {"hoa_monthly": -50.0, "living_area_sqft": 1000.0},     # not a fee
    ] + [{} for _ in range(8)]
    frame = build_features(_frame(rows), config, min_cell_listings=8).frame

    assert frame.loc[0, "hoa_per_sqft"] == pytest.approx(1.5)
    assert frame.loc[0, "hoa_per_sqft_source"] == "reported"
    assert frame.loc[1, "hoa_per_sqft"] == pytest.approx(18.692)
    assert frame.loc[1, "hoa_per_sqft_source"] == "reported"
    assert frame.loc[2, "hoa_per_sqft"] == pytest.approx(0.0)
    assert pd.isna(frame.loc[3, "hoa_per_sqft"])
    assert frame.loc[3, "hoa_per_sqft_source"] == "negative"


def test_inventory_competition_counts_entries_and_excludes_self(config) -> None:
    rows = [
        {"list_date": "2025-01-15", "duration_days": 60.0},
        {"list_date": "2025-01-20", "duration_days": 10.0},
        {"list_date": "2025-03-01", "duration_days": 5.0},
        {"list_date": "2025-01-22", "duration_days": 3.0, "submarket": "aventura"},
    ]
    frame = build_features(_frame(rows), config, min_cell_listings=1).frame

    # Two brickell listings enter in January, so each sees one competitor.
    assert frame.loc[0, "inventory_competition"] == 1
    assert frame.loc[1, "inventory_competition"] == 1
    # March has one brickell entry; a different submarket never competes.
    assert frame.loc[2, "inventory_competition"] == 0
    assert frame.loc[3, "inventory_competition"] == 0


def test_inventory_competition_does_not_depend_on_durations(config) -> None:
    """The count must be fixed before any outcome is realized.

    A covariate built from other listings' durations makes a slow month look
    crowded because its listings were slow, which puts the outcome on both
    sides of the regression.
    """
    rows = [
        {"list_date": "2025-01-15", "duration_days": 5.0},
        {"list_date": "2025-01-20", "duration_days": 5.0},
        {"list_date": "2025-02-05", "duration_days": 5.0},
    ]
    short = build_features(_frame(rows), config, min_cell_listings=1).frame

    for row in rows:
        row["duration_days"] = 400.0
    long = build_features(_frame(rows), config, min_cell_listings=1).frame

    pd.testing.assert_series_equal(
        short["inventory_competition"], long["inventory_competition"]
    )


def test_derived_features_and_buckets(config) -> None:
    rows = [
        {"floor": 3.0},
        {"floor": 9.0},
        {"floor": 20.0},
        {"floor": 45.0},
        {"last_list_price": 900_000.0},
        {"close_price": 950_000.0, "event_sold": 1},
        {"list_date": "2025-07-15"},
    ] + [{} for _ in range(8)]
    frame = build_features(_frame(rows), config, min_cell_listings=8).frame

    assert list(frame.loc[0:3, "floor_bucket"]) == ["low", "mid", "high", "tower"]
    assert frame.loc[0, "log_floor"] == pytest.approx(np.log(4.0))
    assert frame.loc[4, "price_cut_pct"] == pytest.approx(-0.10)
    assert frame.loc[5, "sold_to_list_ratio"] == pytest.approx(0.95)
    assert pd.isna(frame.loc[4, "sold_to_list_ratio"])
    assert frame.loc[6, "season"] == "summer"
    assert frame.loc[0, "season"] == "winter"
    assert frame.loc[0, "list_month"] == "2025-01"
    assert frame.loc[0, "list_quarter"] == "2025Q1"


def test_export_like_profile_builds_without_its_absent_columns(config) -> None:
    """No new_construction and no view column must not break the build."""
    synth = generate_synthetic_mls(profile="like_export")
    frame = normalize_mls(synth.frame, config=config).frame
    result = build_features(frame, config)

    for col in DEMAND_COVARIATES:
        assert col in result.frame.columns

    sources = result.report.new_construction_sources
    assert sources.get("reported", 0) == 0
    assert sources.get("derived", 0) > 0

    corr = result.report.corr_rel_price_premium_event_sold
    assert corr is not None and corr < 0, corr
    # The wide-IQR guard is for the cell-median variable, where dispersion is
    # evidence the cell pools studios with penthouses. A hedonic residual is
    # orthogonal to unit characteristics by construction, so its spread is
    # price dispersion and firing here would attach a true number to a false
    # explanation. What must be reported instead is the respecification itself.
    assert not any("TOO NARROW" in w for w in result.report.warnings)
    assert not any("TOO WIDE" in w for w in result.report.warnings)
    assert any("IDENTIFICATION VARIABLE RESPECIFIED" in w for w in result.report.warnings)
    assert result.report.hedonic_premium["r_squared"] > 0.0


def test_narrow_variation_trips_the_narrow_guard(config) -> None:
    """The brief's degenerate case: everyone prices identically."""
    rows = [{"list_ppsf": 1000.0 + 0.01 * i} for i in range(20)]
    result = build_features(_frame(rows), config, min_cell_listings=8)
    assert any("TOO NARROW" in w for w in result.report.warnings)


def test_missing_normalized_columns_raise(config) -> None:
    with pytest.raises(SchemaError, match="normalized columns"):
        build_features(pd.DataFrame({"mls_number": ["A1"]}), config)


@pytest.mark.skipif(
    not (__import__("pathlib").Path(_REAL_EXPORT_DIR).is_dir()),
    reason="real MLS export not present",
)
def test_real_export_features_report_honestly(config) -> None:
    """Not a calibration run — only checks the report tells the truth about it."""
    frame = ingest_mls(market="miami", config=config).frame
    result = build_features(frame, config)
    report = result.report

    assert report.rel_price_premium["iqr"] > 0.03
    # See above: the wide guard does not apply to a residual. The report must
    # instead say what the first stage explained, which is the honest measure of
    # how much unit quality was removed from the identifying variable.
    assert not any("TOO WIDE" in w for w in report.warnings)
    assert any("IDENTIFICATION VARIABLE RESPECIFIED" in w for w in report.warnings)
    assert 0.0 < report.hedonic_premium["r_squared"] < 1.0
    assert report.hedonic_premium["residual_sd_log"] > 0.0
    # The fallback ladder is genuinely exercised by this export.
    assert report.median_basis_counts.get("submarket_quarter", 0) > 0
    assert report.median_basis_counts.get("insufficient", 0) > 0
    assert report.usable_rows < report.rows
