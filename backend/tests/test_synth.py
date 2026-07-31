"""Phase 1.5 — synthetic MLS generator tests.

The point of these is that the synthetic file must be a *valid stand-in for a
real export* (same ingest path, no schema errors) and must carry recoverable
identification, so Phase 3 can assert that an estimator recovers the planted
`true_beta_price` rather than merely producing a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config
from src.data.clean import clean_mls
from src.data.ingest_mls import ingest_mls
from src.data.normalize import CANONICAL_ALL, normalize_mls
from src.data.synth import SyntheticMLS, generate_synthetic_mls, write_synthetic_mls
from src.exceptions import SchemaError

# rel_price_premium is a ratio of a right-skewed price to a cell *median*, so its
# mean sits slightly above zero even under a symmetric pricing rule. The median
# is zero by construction.
_REL_PREMIUM_MEAN_TOL = 0.05
# PROJECT_BRIEF §2: IQR below this means elasticity is weakly identified.
_MIN_IDENTIFYING_IQR = 0.03
# The generator's default calendar span, in months. Kept next to the assertions
# that depend on it so the coupling is visible.
SYNTH_MONTHS = 24


@pytest.fixture(scope="module")
def synth() -> SyntheticMLS:
    return generate_synthetic_mls()


@pytest.fixture(scope="module")
def export_like() -> SyntheticMLS:
    return generate_synthetic_mls(profile="like_export")


def test_generated_frame_passes_the_real_ingest_path(synth: SyntheticMLS) -> None:
    cfg = load_market_config("miami")
    result = normalize_mls(synth.frame, config=cfg)

    assert result.mapping.required_missing == []
    assert result.mapping.important_missing == []
    assert not result.mapping.unmapped_headers
    for col in CANONICAL_ALL:
        assert col in result.frame.columns

    assert pd.api.types.is_datetime64_any_dtype(result.frame["list_date"])
    assert pd.api.types.is_float_dtype(result.frame["original_list_price"])
    assert result.frame["zip_code"].map(lambda z: isinstance(z, str) and len(z) == 5).all()
    assert set(result.frame["status"].dropna().unique()) <= {
        "SOLD",
        "EXPIRED",
        "WITHDRAWN",
        "CANCELED",
        "ACTIVE",
        "PENDING",
    }

    # The rich profile plants distressed and rental rows so the cleaning filters
    # have coverage beyond the 20-row fixture.
    cleaned, report = clean_mls(result.frame, cfg)
    assert report.dropped_by_reason.get("distressed_sale_type", 0) > 0
    assert report.dropped_by_reason.get("rental_or_lease", 0) > 0
    assert 0 < len(cleaned) < len(synth.frame)


def test_export_like_profile_mirrors_the_real_field_availability(
    export_like: SyntheticMLS,
) -> None:
    """The withheld columns are exactly the ones the Miami export omits."""
    for absent in ("view_description", "property_type", "sale_type", "unit_number"):
        assert absent not in export_like.frame.columns

    result = normalize_mls(export_like.frame, config=load_market_config("miami"))
    assert result.mapping.required_missing == []
    # Matches the real export's report verbatim.
    assert sorted(result.mapping.important_missing) == [
        "property_type",
        "sale_type",
        "unit_number",
    ]
    assert not export_like.truth.view_observable

    # No unit column, so the unit has to be recovered from the address.
    recovered = result.frame["unit_number"].notna().mean()
    assert recovered > 0.95, recovered
    assert result.frame["unit_key"].str.contains(r"\|.", regex=True).mean() > 0.95


def test_export_like_profile_matches_observed_market_behaviour(
    export_like: SyntheticMLS,
) -> None:
    frame = normalize_mls(export_like.frame, config=load_market_config("miami")).frame
    shares = frame["status"].value_counts(normalize=True)

    # Real export: 38.0% SOLD, 5.8% PENDING, longest listing 722 days.
    assert 0.30 <= shares.get("SOLD", 0.0) <= 0.45
    assert 0.02 <= shares.get("PENDING", 0.0) <= 0.10
    assert frame["duration_days"].max() > 180
    assert frame["duration_days"].max() <= 1095

    # PENDING rows are events with a pending_date but no close.
    pending = frame.loc[frame["status"] == "PENDING"]
    assert len(pending) > 0
    assert pending["close_price"].isna().all()
    assert bool(pending["pending_treated_as_sold"].all())
    assert (pending["event_sold"] == 1).all()


def test_export_like_price_level_tracks_the_real_market(
    export_like: SyntheticMLS,
) -> None:
    """Absolute levels matter to the Phase 4 price ladder, not just to beta."""
    frame = normalize_mls(export_like.frame, config=load_market_config("miami")).frame
    median_ppsf = float(frame["list_ppsf"].median())
    # Real export median list $/sqft is 683.
    assert 500 <= median_ppsf <= 900, median_ppsf


def test_status_mix_lands_in_the_accepted_band(synth: SyntheticMLS) -> None:
    sold_share = float(synth.latent["event_sold"].mean())
    assert 0.55 <= sold_share <= 0.75, sold_share


def test_generation_is_deterministic_for_a_seed() -> None:
    first = generate_synthetic_mls(n=400, seed=99)
    second = generate_synthetic_mls(n=400, seed=99)
    assert first.frame.equals(second.frame)
    assert first.truth.base_hazard_daily == second.truth.base_hazard_daily
    assert not generate_synthetic_mls(n=400, seed=100).frame.equals(first.frame)


def test_aggressiveness_is_independent_of_fair_value_noise(synth: SyntheticMLS) -> None:
    """The identifying variation must not be correlated with unobserved quality."""
    corr = float(
        np.corrcoef(synth.latent["aggressiveness"], synth.latent["fair_value_noise"])[0, 1]
    )
    assert abs(corr) < 0.05, corr


def test_confound_knob_breaks_identification_on_demand() -> None:
    confounded = generate_synthetic_mls(n=2000, seed=7, confound_strength=0.8)
    corr = float(
        np.corrcoef(
            confounded.latent["aggressiveness"], confounded.latent["fair_value_noise"]
        )[0, 1]
    )
    assert corr > 0.7, corr


def test_rel_price_premium_smoke_signal(synth: SyntheticMLS) -> None:
    """Phase 2's acceptance check, verified here before features exist."""
    rpp = synth.latent["rel_price_premium"]
    assert abs(float(rpp.median())) < 1e-9
    assert abs(float(rpp.mean())) < _REL_PREMIUM_MEAN_TOL

    iqr = float(rpp.quantile(0.75) - rpp.quantile(0.25))
    assert iqr > _MIN_IDENTIFYING_IQR, iqr

    corr = float(np.corrcoef(rpp, synth.latent["event_sold"])[0, 1])
    assert corr < 0, corr


def test_identification_strengthens_as_beta_grows() -> None:
    """A larger planted |beta| must show up as a stronger price/sale correlation.

    Sold *share* is a poor probe here: the baseline hazard is calibrated at
    rel_price_premium = 0, so beta barely moves it. The correlation is the
    quantity that carries the signal.
    """
    corrs = []
    for beta in (-0.5, -1.6, -3.0):
        result = generate_synthetic_mls(n=3000, seed=7, true_beta_price=beta)
        corrs.append(
            float(
                np.corrcoef(
                    result.latent["rel_price_premium"], result.latent["event_sold"]
                )[0, 1]
            )
        )
    assert corrs[0] > corrs[1] > corrs[2], corrs
    assert all(c < 0 for c in corrs), corrs


def test_censored_rows_carry_no_close_price(synth: SyntheticMLS) -> None:
    frame = normalize_mls(synth.frame, config=load_market_config("miami")).frame
    censored = frame.loc[frame["status"] != "SOLD"]
    assert len(censored) > 0
    assert censored["close_price"].isna().all()
    assert censored["close_date"].isna().all()
    assert censored["off_market_date"].notna().all()

    sold = frame.loc[frame["status"] == "SOLD"]
    assert sold["close_price"].notna().all()
    assert (sold["close_price"] <= sold["original_list_price"]).all()


def test_planted_duration_survives_normalization(synth: SyntheticMLS) -> None:
    """Durations must round-trip, or Phase 3 would fit against corrupted times."""
    frame = normalize_mls(synth.frame, config=load_market_config("miami")).frame
    merged = frame[["mls_number", "duration_days", "event_sold"]].merge(
        synth.latent[["mls_number", "duration_days", "event_sold"]],
        on="mls_number",
        suffixes=("_observed", "_planted"),
    )
    assert len(merged) == len(synth.frame)
    assert (merged["duration_days_observed"] == merged["duration_days_planted"]).all()
    assert (merged["event_sold_observed"] == merged["event_sold_planted"]).all()


def test_missingness_is_injected_without_imputation(synth: SyntheticMLS) -> None:
    frame = normalize_mls(synth.frame, config=load_market_config("miami")).frame
    missing_floor = frame["floor_source"].eq("missing")
    assert 0.01 < float(missing_floor.mean()) < 0.10
    assert frame.loc[missing_floor, "floor"].isna().all()
    assert frame["hoa_monthly"].isna().any()


def test_written_csv_ingests_with_full_coverage(tmp_path, synth: SyntheticMLS) -> None:
    path = write_synthetic_mls(synth, tmp_path / "miami_synth.csv")
    result = ingest_mls(paths=path, market="miami")
    report = result.report

    assert report.rows_in == len(synth.frame)
    assert report.mapping["required_missing"] == []
    assert report.submarket_coverage["rows_without_submarket"] == 0
    assert not any("SUBMARKET COVERAGE GAP" in w for w in report.warnings)
    assert not any("ELASTICITY NOT IDENTIFIABLE" in w for w in report.warnings)
    assert report.floor_rejected_by_stories == 0

    # Every (submarket, month) cell the generator populates should clear
    # min_cell_listings. The expected count is derived from the market config
    # rather than written down: the generator builds towers for whatever
    # submarkets the config declares, so a hardcoded number silently becomes a
    # different assertion the moment a submarket is added. It was written as 192
    # for the original eight submarkets, and adding six for the real export's
    # ZIP coverage turned it into a failure that said nothing about the code.
    n_submarkets = len(load_market_config("miami")["submarkets"])
    expected_cells = n_submarkets * SYNTH_MONTHS
    assert report.identification["submarket_month_cells_ge_min"] == pytest.approx(
        expected_cells, abs=n_submarkets
    ), (
        f"{report.identification['submarket_month_cells_ge_min']} dense cells against "
        f"{n_submarkets} submarkets x {SYNTH_MONTHS} months; a shortfall means the "
        "generator is spreading listings too thinly to populate its own cells"
    )
    assert report.identification["event_sold_count"] == report.identification["sold_count"]


def test_quality_dominates_the_identification_variable(
    synth: SyntheticMLS, export_like: SyntheticMLS
) -> None:
    """rel_price_premium is mostly unit quality, not seller pricing choice.

    The real export's rel_price_premium has IQR 0.617 within (submarket, month)
    cells — far too wide to be pricing behaviour alone. `like_export`
    reproduces that, which is the point: Phase 2 has to residualize quality
    hedonically before the coefficient can be read as an elasticity.
    """
    for result, floor, ceiling in ((synth, 0.20, 0.40), (export_like, 0.50, 0.80)):
        rpp = result.latent["rel_price_premium"]
        iqr = float(rpp.quantile(0.75) - rpp.quantile(0.25))
        assert floor <= iqr <= ceiling, (result.truth.profile, iqr)

    choice_share = float(
        np.var(export_like.latent["aggressiveness"])
        / np.var(export_like.latent["rel_price_premium"])
    )
    assert choice_share < 0.10, choice_share


def test_latent_hazard_basis_attenuates_a_naive_fit() -> None:
    """Quality contamination must be able to hide the planted elasticity.

    Under hazard_basis='latent' the sale hazard responds to the seller's actual
    pricing choice, while the estimator only sees rel_price_premium — which is
    mostly unit quality. A naive fit collapses toward zero even though beta is
    still -1.6, so Phase 3 has to earn the coefficient with hedonic controls
    rather than inherit it from the generator.
    """

    def naive_slope(result: SyntheticMLS) -> float:
        return float(
            np.polyfit(
                result.latent["rel_price_premium"], result.latent["event_sold"], 1
            )[0]
        )

    realized = generate_synthetic_mls(
        profile="like_export", hazard_basis="realized", seed=11
    )
    latent = generate_synthetic_mls(profile="like_export", hazard_basis="latent", seed=11)

    assert naive_slope(realized) < -0.10
    assert abs(naive_slope(latent)) < 0.05

    # The causal channel is intact in both; only the observable proxy degrades.
    for result in (realized, latent):
        corr = float(
            np.corrcoef(result.latent["aggressiveness"], result.latent["event_sold"])[0, 1]
        )
        assert corr < 0, (result.truth.hazard_basis, corr)

    assert latent.truth.hazard_basis == "latent"
    assert (latent.latent["hazard_price_signal"] == latent.latent["aggressiveness"]).all()


def test_invalid_parameters_raise_domain_errors() -> None:
    with pytest.raises(SchemaError):
        generate_synthetic_mls(n=0)
    with pytest.raises(SchemaError):
        generate_synthetic_mls(n=10, target_sold_share=1.0)
    with pytest.raises(SchemaError):
        generate_synthetic_mls(n=10, confound_strength=1.0)
    with pytest.raises(SchemaError):
        generate_synthetic_mls(n=10, profile="nope")
    with pytest.raises(SchemaError):
        generate_synthetic_mls(n=10, hazard_basis="nope")
