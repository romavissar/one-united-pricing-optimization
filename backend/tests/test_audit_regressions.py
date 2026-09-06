"""Regression tests for the findings in `audit/AUDIT_REPORT.md`.

Each test names the finding it guards and fails on the behaviour that was
shipped before the audit. Several are property-based or metamorphic rather than
example-based: those guard the *class* of defect, not the one instance found.

Reproduction scripts for the underlying measurements live under `audit/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config, validate_market_id
from src.data.clean import _PPSF_PLAUSIBLE_BOUNDS, clean_mls
from src.data.features import build_features
from src.data.ingest_mls import build_sampling_report
from src.data.normalize import (
    build_unit_key,
    is_missing,
    normalize_hoa_frequency,
    normalize_mls,
    reconstruct_list_date,
    slug,
)
from src.demand.registry import model_dir
from src.demand.survival import CoxDemandModel
from src.exceptions import SchemaError
from src.optimizer.solve import _CEILING_SHARE_ALARM
from src.simulation.monte_carlo import summarize


@pytest.fixture(scope="module")
def config():
    return load_market_config("miami")


# --- F01 / F02  the calibration flag ----------------------------------------


def test_f01_market_config_is_not_marked_calibrated(config):
    """F01 — config claimed real-data calibration while nothing was calibrated.

    `is_calibrated_on_real_data: true` in the market config makes
    /api/config/{market} report the system as calibrated, which suppressed the
    illustrative banner over numbers fitted on synthetic data. Flipping it is a
    human decision taken after inspecting a real fit (CALIBRATION_READY.md §6),
    never a checked-in default.
    """
    assert config["defaults"]["is_calibrated_on_real_data"] is False


# --- F03  list_date reconstruction ------------------------------------------


def test_f03_list_date_is_recovered_from_terminal_minus_dom():
    """F03 — a blank List Date dropped whole statuses out of identification.

    In the Miami export `List Date` is blank for every PENDING and every
    WITHDRAWN row and present for every other status, so leaving it blank
    removed 100% of one event class and 100% of one censored class from the
    panel — selection on the outcome. The value is recoverable as
    `terminal date - days_on_market`.
    """
    frame = pd.DataFrame(
        {
            "list_date": [pd.Timestamp("2025-01-01"), pd.NaT, pd.NaT, pd.NaT],
            "pending_date": [pd.NaT, pd.Timestamp("2025-04-11"), pd.NaT, pd.NaT],
            "off_market_date": [pd.NaT, pd.NaT, pd.Timestamp("2025-06-10"), pd.NaT],
            "days_on_market": [50, 100, 40, 30],
        }
    )
    listed, source, recovered, _rejected = reconstruct_list_date(frame)

    assert recovered == 2
    assert source.tolist() == ["reported", "derived_from_dom", "derived_from_dom", "missing"]
    # An event terminates at buyer commitment, a termination at going off market.
    assert listed.iloc[1] == pd.Timestamp("2025-01-01")
    assert listed.iloc[2] == pd.Timestamp("2025-05-01")
    # A reported date is never overwritten.
    assert listed.iloc[0] == pd.Timestamp("2025-01-01")
    # No terminal date means no derivation — nothing is invented.
    assert pd.isna(listed.iloc[3])


def test_f03_recovered_rows_carry_a_source_marker_not_a_silent_value():
    """F03 — AGENTS.md §2: a derived value must be distinguishable from a reported one."""
    frame = pd.DataFrame(
        {
            "list_date": [pd.NaT],
            "pending_date": [pd.Timestamp("2025-03-01")],
            "off_market_date": [pd.NaT],
            "days_on_market": [10],
        }
    )
    _, source, _, _ = reconstruct_list_date(frame)
    assert source.iloc[0] == "derived_from_dom"


# --- F04 / F05  sampling diagnostics ----------------------------------------


def test_f05_terminal_date_window_selection_is_detected():
    """F05 — a sample of terminations in a window is not a sample of listings.

    Every spell complete, no ACTIVE listings, and terminal dates spanning far
    less than list dates means slow and unsold listings are structurally absent,
    which attenuates beta_price. Nothing detected this before.
    """
    frame = pd.DataFrame(
        {
            "list_date": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2025-01-01", "2025-06-01"]
            ),
            "off_market_date": pd.to_datetime(
                ["2025-10-20", "2025-11-15", "2026-01-10", "2026-02-01"]
            ),
            "close_date": [pd.NaT] * 4,
            "pending_date": [pd.NaT] * 4,
            "status": ["CANCELED", "EXPIRED", "SOLD", "SOLD"],
        }
    )
    sampling, warnings = build_sampling_report(frame, rows_in=len(frame))
    assert sampling["active_listings"] == 0
    assert sampling["terminal_date_span_days"] < sampling["list_date_span_days"]
    assert any("SELECTED ON THE TERMINAL DATE" in w for w in warnings)


def test_f04_status_sorted_export_is_detected():
    """F04 — a row-capped export sorted by status truncates on the outcome."""
    status = pd.Series(["Cancelled"] * 5 + ["Closed"] * 5 + ["Expired"] * 3)
    frame = pd.DataFrame({"status": ["CANCELED"] * 13})
    sampling, warnings = build_sampling_report(frame, rows_in=13, raw_status=status)
    assert sampling["export_sorted_by_status"] is True
    assert sampling["last_status_block"] == {"status": "Expired", "n": 3}
    assert any("SORTED BY STATUS" in w for w in warnings)


def test_f04_interleaved_status_is_not_flagged_as_sorted():
    """The status warning must not fire on an export in natural order."""
    status = pd.Series(["Closed", "Expired", "Closed", "Cancelled", "Closed"])
    frame = pd.DataFrame({"status": ["SOLD"] * 5})
    sampling, warnings = build_sampling_report(frame, rows_in=5, raw_status=status)
    assert sampling.get("export_sorted_by_status") is False
    assert not any("SORTED BY STATUS" in w for w in warnings)


# --- F06  implausible $/sqft ------------------------------------------------


def test_f06_implausible_ppsf_is_nulled_and_marked_not_dropped(config):
    """F06 — one bad area set the premium support and disabled the guard.

    A $49.5M listing recorded as 798 sqft gives $62,030/sqft, a
    rel_price_premium of +109, and a fitted support running to +10,900% — inside
    which no recommended price can ever be flagged as extrapolation.
    """
    low, high = _PPSF_PLAUSIBLE_BOUNDS
    frame = pd.DataFrame(
        {
            "list_ppsf": [700.0, high * 3, low / 3, np.nan],
            "close_ppsf": [650.0, np.nan, np.nan, np.nan],
            "original_list_price": [700_000.0] * 4,
            "mls_number": list("abcd"),
            "duration_days": [30.0] * 4,
            "submarket": ["brickell"] * 4,
            "list_date": pd.to_datetime(["2025-01-01"] * 4),
        }
    )
    cleaned, report = clean_mls(frame, config)

    assert report.list_ppsf_implausible == 2
    # The rows survive; only the untrustworthy derived value is removed.
    assert len(cleaned) == 4
    assert cleaned["list_ppsf"].notna().sum() == 1
    assert cleaned.loc[cleaned["list_ppsf_source"] == "implausible"].shape[0] == 2
    # A genuine trophy price stays: real Miami listings reach five figures/sqft.
    genuine = pd.DataFrame(
        {
            "list_ppsf": [10_145.0],
            "close_ppsf": [np.nan],
            "original_list_price": [34_350_900.0],
            "mls_number": ["e"],
            "duration_days": [30.0],
            "submarket": ["brickell"],
            "list_date": pd.to_datetime(["2025-01-01"]),
        }
    )
    kept, kept_report = clean_mls(genuine, config)
    assert kept_report.list_ppsf_implausible == 0
    assert kept["list_ppsf"].notna().all()


# --- F08  pd.NA leaking through null guards ---------------------------------


@pytest.mark.parametrize("null", [None, np.nan, pd.NA, pd.NaT, float("nan")])
def test_f08_every_null_flavour_is_recognised(null):
    """F08 — `isinstance(x, float) and pd.isna(x)` misses pd.NA and pd.NaT.

    `map_columns` materialises every canonical field the export omitted as
    `pd.NA`, so whole columns took the fall-through path.
    """
    assert is_missing(null) is True


@pytest.mark.parametrize("present", [0, 0.0, "", "x", False])
def test_f08_present_values_are_not_treated_as_missing(present):
    assert is_missing(present) is False


def test_f08_missing_building_name_falls_back_to_street_address():
    """F08 — slug(pd.NA) returned "na", so the documented fallback never fired.

    Worse than a missing fallback: every unit whose building name was absent
    landed in one `na|<unit>` namespace and collided with every other such unit.
    """
    assert slug(pd.NA) == ""
    assert build_unit_key(pd.NA, "123 Ocean Dr", "1204") == "123oceandr|1204"
    assert build_unit_key(np.nan, "123 Ocean Dr", "1204") == "123oceandr|1204"
    # Two different buildings with no name must not collide.
    a = build_unit_key(pd.NA, "1 Alpha Way", "500")
    b = build_unit_key(pd.NA, "2 Beta Rd", "500")
    assert a != b


def test_f08_absent_hoa_frequency_stays_null_not_the_string_NA():
    """F08 — hoa_frequency came out as the literal string "<NA>" on every row."""
    monthly, label = normalize_hoa_frequency(pd.NA, 500.0)
    assert monthly == 500.0
    assert label is None


# --- F10 / F09  what the optimizer does not model ---------------------------


def test_f10_unmodelled_probability_mass_is_reported(config):
    """F10 — expected revenue is a partial sum and nothing said so.

    A unit at D=0.7 contributes 0.7*p*A; the other 30% is not carried into a
    later phase and simply vanishes.
    """
    from src.optimizer.solve import OptimizeResult, PlanRow, SolveStatus

    rows = [
        PlanRow("u1", 0, "p1", 5, 1000.0, 1_000_000.0, 0.60, 600_000.0, 600_000.0),
        PlanRow("u2", 0, "p1", 5, 1000.0, 2_000_000.0, 0.80, 1_600_000.0, 1_600_000.0),
    ]
    result = OptimizeResult(
        status=SolveStatus.OPTIMAL, plan=rows, objective_usd=2_200_000.0,
        expected_revenue_usd=2_200_000.0, per_phase=[], unreleased_unit_ids=[],
        excluded_units=[], provenance={}, constraints={}, caveats=[], solve_seconds=0.0,
    )
    assert result.gross_ask_usd == pytest.approx(3_000_000.0)
    assert result.unmodelled_probability_mass == pytest.approx(0.6)
    assert result.unmodelled_revenue_usd == pytest.approx(800_000.0)
    payload = result.as_dict()
    for key in ("gross_ask_usd", "unmodelled_probability_mass", "unmodelled_revenue_usd"):
        assert key in payload


def test_f21_ceiling_alarm_triggers_below_four_fifths():
    """F21 — two thirds of the stack at its ceiling passed unremarked at 0.80."""
    assert _CEILING_SHARE_ALARM <= 0.67


# --- F11  horizon beyond the fitted follow-up -------------------------------


def test_f11_cox_reports_the_follow_up_it_was_fitted_over(config):
    """F11 — past the last event the baseline hazard is flat and says nothing.

    `horizon_days` is caller-settable through the API, so a horizon beyond the
    data returns the last observed probability wearing a longer label.
    """
    from src.data.features import DEMAND_COVARIATES
    from src.data.synth import generate_synthetic_mls

    synth = generate_synthetic_mls(n=1200, seed=5, market="miami", profile="rich")
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    model = CoxDemandModel(covariates=DEMAND_COVARIATES)
    model.fit(frame)

    followup = model.max_observed_duration_days
    assert followup is not None and followup > 0

    score = frame.head(50)
    at_edge = model.predict_sale_probability(score, score["list_ppsf"].to_numpy(), int(followup))
    far_past = model.predict_sale_probability(
        score, score["list_ppsf"].to_numpy(), int(followup * 10)
    )
    # Identical is the point: the model has no information out there, and the
    # tensor builder must therefore warn rather than let this pass as a forecast.
    assert np.allclose(np.nan_to_num(at_edge), np.nan_to_num(far_past))


# --- F07  coefficient covariance --------------------------------------------


def test_f07_coefficient_covariance_is_available_and_in_raw_units(config):
    """F07 — Phase 4 could not propagate joint uncertainty because Phase 2 dropped it."""
    from src.data.features import DEMAND_COVARIATES
    from src.data.synth import generate_synthetic_mls

    synth = generate_synthetic_mls(n=1500, seed=6, market="miami", profile="rich")
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    model = CoxDemandModel(covariates=DEMAND_COVARIATES)
    result = model.fit(frame)

    cov = model.coefficient_covariance
    assert cov is not None
    assert "rel_price_premium" in cov.columns
    # Diagonal must reproduce the reported (raw-unit) standard errors.
    sd = float(np.sqrt(cov.loc["rel_price_premium", "rel_price_premium"]))
    assert sd == pytest.approx(result.beta_price.std_error, rel=1e-6)
    # It must not be diagonal — that is precisely why marginal SEs are not enough.
    off = cov.to_numpy() - np.diag(np.diag(cov.to_numpy()))
    assert np.abs(off).max() > 0


# --- F16  path containment ---------------------------------------------------


@pytest.mark.parametrize("bad", ["../../etc", "miami/../..", "..", "MIAMI", "mia mi", ""])
def test_f16_market_ids_that_are_paths_are_rejected(bad):
    """F16 — `market` reached a filesystem join straight from the URL."""
    with pytest.raises(SchemaError):
        validate_market_id(bad)


@pytest.mark.parametrize("bad", ["../secrets", "a/b", "", "x" * 65])
def test_f16_bundle_names_that_are_paths_are_rejected(bad):
    """F16 — the bundle directory holds a pickle that load_bundle executes."""
    with pytest.raises(SchemaError):
        model_dir("miami", bad)


def test_f16_valid_identifiers_still_resolve():
    assert validate_market_id("miami") == "miami"
    assert model_dir("miami", "current").name == "current"


# --- Metamorphic / property guards (brief §1.3) ------------------------------


def test_metamorphic_scale_invariance_of_the_identification_variable():
    """Multiplying prices and the comp median together must leave the premium fixed.

    This is the relation that would catch a currency or sqft/sqm conversion
    applied to one side of the ratio and not the other.
    """
    from src.demand.base import relative_premium

    price = np.array([500.0, 1000.0, 2500.0])
    median = np.array([800.0, 800.0, 800.0])
    base = relative_premium(price, median)
    for factor in (0.25, 2.0, 10.7639, 137.0):
        assert np.allclose(relative_premium(price * factor, median * factor), base)


def test_metamorphic_non_positive_median_yields_nan_not_zero():
    """A missing comp median must not read as "priced exactly at comps"."""
    from src.demand.base import relative_premium

    assert np.isnan(relative_premium(np.array([900.0]), np.array([0.0])))[0]
    assert np.isnan(relative_premium(np.array([900.0]), np.array([np.nan])))[0]


@pytest.mark.parametrize(
    "sample",
    [
        np.array([1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 5.0, 10.0, 1e9]),
        np.linspace(-1e6, 1e6, 501),
    ],
)
def test_property_percentiles_ordered_and_cvar_below_p5(sample):
    """P5 <= P25 <= P50 <= P75 <= P95 and CVaR5 <= P5, on every shape of sample."""
    summary = summarize(sample)
    assert summary.is_ordered
    assert summary.cvar5 <= summary.p5 + 1e-9


def test_property_summarize_rejects_an_empty_sample():
    with pytest.raises(SchemaError):
        summarize(np.array([]))


# --- Batch B: encoding and silent-fallback guards ---------------------------


def test_no_single_field_dominates_the_design_matrix(config):
    """`view_description` once took 420 of 443 design columns as a raw category.

    A comma-separated MLS field is not one categorical level — `Unit View` reads
    "Bay, Skyline View, Water View", three facts about the unit. Treated as a
    category it had 1,139 distinct values on the quarterly pull and consumed 95%
    of the design while pooling 694 levels into `__rare__`; the 16 atomic
    indicators that carry the same signal went unused.

    The assertion is on the *class* of defect, not the instance: no source field
    may dominate the design, whichever field a future export introduces.
    """
    from collections import Counter

    from src.data.ingest_mls import ingest_mls
    from src.data.features import DEMAND_COVARIATES
    from src.demand.survival import (
        CONTROLLED_CATEGORICALS, CoxDemandModel, available_covariates,
    )

    built = build_features(ingest_mls(market="miami", config=config).frame, config)
    frame = built.frame
    result = CoxDemandModel(
        covariates=available_covariates(frame),
        categoricals=CONTROLLED_CATEGORICALS,
        premium_model=built.premium_model,
    ).fit(frame)

    columns = result.design.X.columns
    by_source = Counter(c.split("=")[0] for c in columns)
    worst, count = by_source.most_common(1)[0]
    assert count <= 20, (
        f"{worst} takes {count} of {len(columns)} design columns — a multi-valued "
        "field is being encoded as a raw categorical again"
    )
    # The view signal must still be present, as indicators.
    assert any(c.startswith("view_") for c in columns)


def test_losing_the_premium_model_raises_instead_of_falling_back(config):
    """`frame.attrs` does not survive `pd.concat`, and the fallback is wrong.

    `rel_price_premium` is a residual against the list-price hedonic, so scoring
    a price against the submarket median instead measures the coefficient
    against a quantity it was never fitted on. That produces entirely plausible
    numbers with no warning, which is the failure this guard exists to prevent.
    """
    from src.data.ingest_mls import ingest_mls
    from src.data.features import DEMAND_COVARIATES
    from src.demand.survival import CoxDemandModel
    from src.exceptions import IdentificationError

    frame = build_features(
        ingest_mls(market="miami", config=config).frame, config
    ).frame
    assert "premium_model" in frame.attrs

    # concat is the realistic way it gets lost.
    stripped = pd.concat([frame.head(4000), frame.tail(4000)])
    assert "premium_model" not in stripped.attrs

    with pytest.raises(IdentificationError, match="premium model"):
        CoxDemandModel(covariates=DEMAND_COVARIATES).fit(stripped)


def test_absent_binary_indicator_scores_as_zero_not_an_error(config):
    """An inventory exercising fewer view tokens than the fit must still score.

    A fit that saw nine view tokens can price an inventory that carries three:
    the six it lacks are zero, because the unit does not have that view. That is
    replaying the fit's vocabulary, not imputing — which is why it is restricted
    to columns that were strictly binary at fit time. A missing `log_floor` is a
    gap and must still raise.
    """
    from src.data.ingest_mls import ingest_mls
    from src.data.features import DEMAND_COVARIATES
    from src.demand.base import transform_to_design
    from src.demand.survival import CoxDemandModel, available_covariates
    from src.exceptions import SchemaError

    built = build_features(ingest_mls(market="miami", config=config).frame, config)
    frame = built.frame
    model = CoxDemandModel(
        covariates=available_covariates(frame), premium_model=built.premium_model
    )
    model.fit(frame)

    scoring = frame.head(200).copy()
    view_columns = [c for c in scoring.columns if c.startswith("view_")]
    assert view_columns, "fixture must exercise the token path"
    scoring = scoring.drop(columns=view_columns[:3])
    encoded, usable = transform_to_design(scoring, model._design)
    assert usable.any()
    for column in view_columns[:3]:
        if column in encoded.columns:
            assert (encoded[column] == 0.0).all()

    # A non-indicator covariate is still a hard error.
    with pytest.raises(SchemaError, match="missing fitted covariates"):
        transform_to_design(frame.head(50).drop(columns=["log_floor"]), model._design)
