"""Phase 3 — the test that everything downstream rests on.

PROJECT_BRIEF §3d: generate synthetic data with a known `true_beta_price`, fit
the Cox model, and assert the recovered coefficient lands near the planted value
with a 95% CI that excludes zero. Repeat at -0.4 and -3.0 to confirm the
pipeline tracks the parameter rather than always returning the same number. If
this does not pass, nothing downstream can be trusted.

There are two bars here, and they measure different things.

Under `hazard_basis="realized"` the hazard is driven by `rel_price_premium`
itself, so the estimator regresses on exactly the variable that generated the
data. Recovery is then a test of *estimator code* — the design matrix, the
scaling round-trip, the censoring handling — and it should be close to exact.
That is the brief's acceptance criterion.

Under `hazard_basis="latent"` the hazard is driven by the seller's own
aggressiveness, and `rel_price_premium` is a quality-contaminated proxy for it.
This is what the real export looks like. Recovery is then a test of
*specification*, and classical errors-in-variables guarantees attenuation
toward zero that no amount of sample size fixes. The tests below pin the shape
of that attenuation: it shrinks monotonically as controls are added, it largely
disappears when the contaminating quality is building-level and absorbed by
fixed effects, and when it is present the diagnostics say so instead of
reporting the shrunken number as if it were the elasticity.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config
from src.data.features import DEMAND_COVARIATES, build_features
from src.data.normalize import normalize_mls
from src.data.synth import _BUILDINGS_PER_SUBMARKET, generate_synthetic_mls
from src.demand.base import build_design_matrix, relative_premium
from src.demand.diagnostics import (
    format_diagnostics_report,
    require_pass,
    run_diagnostics,
)
from src.demand.hedonic import fit_hedonic
from src.demand.logistic import LogisticDemandModel, build_horizon_label
from src.demand.registry import (
    REAL_MLS,
    ModelBundle,
    Provenance,
    load_bundle,
    load_metadata,
    save_bundle,
)
from src.demand.survival import (
    CONTROLLED_CATEGORICALS,
    CONTROLLED_COVARIATES,
    CoxDemandModel,
    available_covariates,
)
from src.exceptions import IdentificationError, SchemaError

N_LISTINGS = 6000
# Submarket count pinned for the building-fixed-effects test, so the number of
# fixed effects (and therefore the incidental-parameters bias) is a constant of
# the test rather than a function of how many submarkets the market config
# happens to declare. Eight is what the documented recovery was measured on.
_FE_TEST_SUBMARKETS = 8
SEED = 13
HORIZON_DAYS = 180

# The brief's acceptance tolerance, stated in the units the estimator's
# precision is actually measured in. See the docstring of
# `test_cox_recovers_the_planted_beta_price` for why a fixed relative tolerance
# cannot serve here: at n=6,000 the standard error barely moves with the size of
# the planted coefficient, so 20% is six standard errors at -3.0 and less than
# one at -0.4.
RECOVERY_TOLERANCE_SES = 3.0
# A backstop on precision, loose enough to be non-binding at every planted value
# the suite uses but tight enough to catch an estimator that passed the SE bar
# only by reporting a uselessly wide interval.
RECOVERY_TOLERANCE_RELATIVE = 0.75

_CACHE: dict[tuple, pd.DataFrame] = {}


@pytest.fixture(scope="module")
def config():
    return load_market_config("miami")


def _features(
    config, *, n: int = N_LISTINGS, config_for_generator=None, **kwargs
) -> pd.DataFrame:
    """Synthetic export pushed through the real ingest and feature path, cached.

    `config_for_generator` pins the submarket layout the generator draws towers
    from, for tests whose claim depends on how many fixed effects exist.
    """
    generator_config = config_for_generator if config_for_generator is not None else config
    key = (n, id(generator_config) if config_for_generator is not None else 0,
           *sorted(kwargs.items()))
    if key not in _CACHE:
        synthetic = generate_synthetic_mls(
            n=n, seed=SEED, config=generator_config, **kwargs
        )
        normalized = normalize_mls(synthetic.frame, config=config).frame
        _CACHE[key] = build_features(normalized, config).frame
    return _CACHE[key]


def _fit_cox(frame: pd.DataFrame, *, controlled: bool = False, fixed_effects: bool = False):
    model = CoxDemandModel(
        covariates=CONTROLLED_COVARIATES if controlled else DEMAND_COVARIATES,
        categoricals=CONTROLLED_CATEGORICALS if controlled else ("submarket", "season"),
        building_fixed_effects=fixed_effects,
    )
    return model, model.fit(frame)


# --- The acceptance criterion ------------------------------------------------


@pytest.mark.parametrize("true_beta", [-0.4, -1.6, -3.0])
def test_cox_recovers_the_planted_beta_price(config, true_beta):
    """Fit recovers the planted elasticity and its CI excludes zero.

    Recovery is asserted in standard errors, not as a fixed percentage of the
    planted value, and the reason is arithmetic rather than taste. At n=6,000
    the fitted SE is about 0.084 whatever beta is planted, because it is set by
    the spread of `rel_price_premium` and the event count, not by the size of
    the coefficient. A flat 20% tolerance is therefore 6.1 SE at a planted -3.0
    and **0.95 SE** at -0.4 — so at the smallest parameter the old assertion
    demanded the estimate land inside one standard error, which a correct,
    unbiased estimator does only about two thirds of the time.

    Measured over 24 seeds (`audit/a04_recovery_seed_sweep.py`): mean recovered
    -0.3957 against a planted -0.4000, a bias of +0.0043 with a t-statistic of
    +0.21 — no detectable bias — while the shipped 20% rule failed 12 of those
    24 seeds. A test that a correct estimator fails half the time is not
    measuring recovery; it is measuring the seed.

    Three SEs is a genuine bar: an unbiased estimator clears it with probability
    ~0.997, and a pipeline that had lost the coefficient entirely (attenuated to
    zero, or wired to the wrong variable) would miss it by 5-35 SE.
    """
    frame = _features(config, true_beta_price=true_beta, hazard_basis="realized")
    _, result = _fit_cox(frame)
    beta = result.beta_price

    error_in_ses = abs(beta.value - true_beta) / beta.std_error
    assert error_in_ses <= RECOVERY_TOLERANCE_SES, (
        f"recovered {beta.value:+.4f} against a planted {true_beta:+.4f} — "
        f"{error_in_ses:.2f} standard errors away (SE {beta.std_error:.4f}), "
        f"beyond the {RECOVERY_TOLERANCE_SES} SE bar. This is a bias, not noise."
    )
    # A second, absolute guard so an estimator that lost precision as well as
    # accuracy cannot pass by reporting a huge standard error.
    relative_error = abs(beta.value - true_beta) / abs(true_beta)
    assert relative_error <= RECOVERY_TOLERANCE_RELATIVE, (
        f"recovered {beta.value:+.4f} against a planted {true_beta:+.4f} "
        f"({relative_error:.1%} off)"
    )
    assert beta.excludes_zero, (
        f"95% CI [{beta.ci_low:+.4f}, {beta.ci_high:+.4f}] covers zero, so the fit "
        "cannot distinguish this market from one with no price response"
    )
    assert beta.value < 0


def test_recovery_interval_covers_the_planted_value(config):
    """The headline case is not merely close — the interval contains the truth."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, result = _fit_cox(frame)
    beta = result.beta_price
    assert beta.ci_low <= -1.6 <= beta.ci_high


def test_pipeline_tracks_the_parameter_rather_than_returning_a_constant(config):
    """Three planted values produce three ordered estimates, not one number."""
    recovered = [
        _fit_cox(_features(config, true_beta_price=b, hazard_basis="realized"))[1].beta_price.value
        for b in (-0.4, -1.6, -3.0)
    ]
    assert recovered[0] > recovered[1] > recovered[2], recovered
    # A pipeline that ignored its input would return three near-identical numbers.
    assert abs(recovered[2] - recovered[0]) > 2.0


# --- What happens when the price signal is latent ----------------------------


def test_naive_fit_is_severely_attenuated_when_quality_contaminates_the_premium(config):
    """The brief's covariate list alone recovers a small fraction of the truth.

    This is the honest depiction of the real export. It is not a failing
    estimator; it is a regressor that mostly measures which unit it is rather
    than how that unit was priced.
    """
    frame = _features(config, true_beta_price=-1.6, hazard_basis="latent")
    _, result = _fit_cox(frame)
    beta = result.beta_price

    assert beta.value < 0
    assert abs(beta.value) < 0.5 * 1.6, (
        f"expected heavy attenuation under the latent basis, got {beta.value:+.4f}"
    )

    report = run_diagnostics(frame, result)
    assert any("lower bound" in note for note in report.identification.notes), (
        "diagnostics must say the fitted coefficient understates the elasticity, "
        "not present it as the elasticity"
    )

    # The same frame fitted with the hedonic controls: now most of the
    # identifying variable is visibly explained by unit characteristics, which
    # is the direct measurement of the contamination.
    _, controlled = _fit_cox(frame, controlled=True)
    share = run_diagnostics(frame, controlled).identification.quality_explained_share
    assert share is not None and share > 0.35


def test_a_non_numeric_covariate_raises_instead_of_silently_vanishing(config):
    """A text column not declared categorical must not coerce to a column of NaN."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    with pytest.raises(SchemaError, match="is not numeric"):
        build_design_matrix(
            frame,
            [*DEMAND_COVARIATES, "view_description"],
            categoricals=("submarket", "season"),
        )


def test_controls_and_fixed_effects_monotonically_reduce_attenuation(config):
    """Each rung of the control ladder moves the estimate toward the truth."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="latent")
    naive = _fit_cox(frame)[1].beta_price.value
    controlled = _fit_cox(frame, controlled=True)[1].beta_price.value
    with_fe = _fit_cox(frame, controlled=True, fixed_effects=True)[1].beta_price.value

    assert abs(naive) < abs(controlled) < abs(with_fe) < 1.6, (
        f"naive={naive:+.4f} controlled={controlled:+.4f} fe={with_fe:+.4f}"
    )


def test_building_fixed_effects_recover_beta_when_quality_is_building_level(config):
    """With quality fully at the building level, fixed effects absorb it.

    This is the configuration where identification is genuinely achievable:
    everything contaminating `rel_price_premium` is constant within a tower, so
    the fixed effects remove it and what survives is the seller's pricing
    choice. The residual gap is the part of the specification that stays
    misspecified — the hazard is linear in the aggressiveness that generated it
    but the model regresses on a premium that is a ratio of prices, and those
    are not the same functional form.

    **The tower layout is pinned deliberately.** The generator builds
    `_BUILDINGS_PER_SUBMARKET` towers for every submarket the market config
    declares, so both the number of fixed effects and the listings per tower are
    functions of `config/miami.yaml`. When the config grew from 8 submarkets to
    14 to cover the real export's ZIPs, towers rose from 96 to 168, density fell
    from ~60 listings per tower to ~34, and recovery fell from 74% of the
    planted value to 44% — with no change to any estimator, purely from
    incidental-parameters bias in a partial likelihood carrying one dummy per
    level. `audit/a11_building_fe_coupling.py` measures that curve.

    Pinning to the eight submarkets this claim was originally measured on is
    what keeps it a test of whether fixed effects absorb building-level quality,
    rather than a test of how many submarkets someone last added to a config
    file. The bias itself is not a defect and is not suppressed here: the fitted
    model reports it (`THIN FIXED EFFECTS`) whenever levels are this sparse,
    which on the real export they overwhelmingly are.
    """
    pinned = dict(config)
    pinned["submarkets"] = {
        name: meta
        for name, meta in list(config["submarkets"].items())[:_FE_TEST_SUBMARKETS]
    }
    towers = len(pinned["submarkets"]) * _BUILDINGS_PER_SUBMARKET

    frame = _features(
        pinned,
        config_for_generator=pinned,
        true_beta_price=-1.6,
        hazard_basis="latent",
        building_quality_share=1.0,
    )
    n_listings = N_LISTINGS
    naive = _fit_cox(frame)[1].beta_price.value
    _, result = _fit_cox(frame, controlled=True, fixed_effects=True)
    beta = result.beta_price

    assert abs(beta.value) >= 0.60 * 1.6, (
        f"fixed effects recovered only {beta.value:+.4f} of a planted -1.6 at "
        f"{n_listings} listings across {towers} towers "
        f"({n_listings / towers:.0f} per tower)"
    )
    assert abs(beta.value) > 3.0 * abs(naive)
    assert beta.excludes_zero
    assert "submarket" not in result.covariates, (
        "building fixed effects nest submarket; keeping both makes the design singular"
    )


# --- Failing loudly ----------------------------------------------------------


def test_a_positive_price_coefficient_is_a_hard_failure(config):
    """A wrong-signed beta_price blocks, regardless of how tight its interval is."""
    frame = _features(config, true_beta_price=1.5, hazard_basis="realized")
    _, result = _fit_cox(frame)
    assert result.beta_price.value > 0

    report = run_diagnostics(frame, result)
    assert report.status == "FAILED"
    assert any(check.name == "cox_beta_price" for check in report.failed)
    with pytest.raises(IdentificationError, match="blocking diagnostic"):
        require_pass(report)


def test_export_like_data_cannot_identify_beta_and_says_so(config):
    """The honest, uncomfortable result, pinned so nobody quietly tunes it away.

    The `like_export` profile mirrors the real Miami pull: no view column, and
    a fair-value spread roughly six times the spread of sellers' pricing
    choices. Under the latent hazard basis — the one the real market obeys —
    under 2% of the variance in `rel_price_premium` is the pricing decision, so
    the coefficient is indistinguishable from zero and lands on either side of
    it by chance.

    The required behaviour is not a recovered elasticity. It is a `FAILED`
    diagnostic saying this data cannot answer the question. If this test ever
    starts passing with a confident negative beta, check what changed in the
    specification before believing it.
    """
    frame = _features(
        config,
        true_beta_price=-1.6,
        hazard_basis="latent",
        profile="like_export",
    )
    covariates = available_covariates(frame)
    assert "view_description" not in covariates, (
        "an all-null column must count as unavailable, or listwise deletion "
        "empties the sample"
    )

    model = CoxDemandModel(covariates=covariates, categoricals=CONTROLLED_CATEGORICALS)
    result = model.fit(frame)
    beta = result.beta_price
    assert not beta.excludes_zero, (
        f"expected an interval covering zero, got [{beta.ci_low:+.4f}, {beta.ci_high:+.4f}]"
    )

    report = run_diagnostics(frame, result)
    if beta.value >= 0:
        assert report.status == "FAILED"
        with pytest.raises(IdentificationError):
            require_pass(report)
    else:
        assert any("covers zero" in warning for warning in report.warnings)


def test_a_linearly_dependent_column_is_dropped_with_an_explanation(config):
    """A rank-deficient design must not reach the fitter as a singular matrix.

    Sparse exports produce exact dependencies after listwise deletion. Both
    lifelines and statsmodels answer with an opaque "singular matrix" that names
    nothing, so the redundant column is removed here and recorded instead.
    """
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized").copy()
    frame["log_floor_twin"] = frame["log_floor"]

    design = build_design_matrix(frame, [*DEMAND_COVARIATES, "log_floor_twin"])
    assert len(design.dropped_dependent) == 1
    assert design.dropped_dependent[0] in {"log_floor", "log_floor_twin"}
    assert any("linearly dependent" in note for note in design.notes)
    assert np.linalg.matrix_rank(design.X.to_numpy()) == design.X.shape[1]


def test_an_unfittable_cross_check_shows_up_in_the_report(config):
    """A missing logistic must not read as a passing one."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, result = _fit_cox(frame)
    report = run_diagnostics(frame, result, logistic_error="singular matrix")

    check = next(c for c in report.checks if c.name == "logistic_cross_check")
    assert check.status == "WARN"
    assert "singular matrix" in check.message
    assert "nothing corroborating it" in check.message


def test_the_cli_refuses_to_fit_on_the_real_export_unquestioned():
    """PROJECT_BRIEF §5 is a mechanism here, not a convention.

    Running the demand CLI with no source defaults to `data/raw/mls`, which is
    the easiest way to calibrate on real data by accident.
    """
    from src.demand.diagnostics import _targets_real_data, main

    assert _targets_real_data(None, None) is True
    assert _targets_real_data(None, ["data/raw/mls/export.csv"]) is True
    assert _targets_real_data("data/synthetic", None) is False

    assert main(["--inspect"]) == 1


def test_a_healthy_fit_passes_the_gate(config):
    """The gate lets a correctly signed fit through, warnings and all."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, result = _fit_cox(frame)
    report = run_diagnostics(frame, result)
    assert report.status in {"PASS", "WARN"}
    require_pass(report)


def test_diagnostics_report_is_serializable_and_printable(config):
    """The report has to survive the trip to the API and to a terminal."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, result = _fit_cox(frame)
    report = run_diagnostics(frame, result)

    encoded = json.dumps(report.as_dict(), default=str)
    assert "rel_price_premium" in encoded
    text = format_diagnostics_report(report)
    assert "beta_price" in text
    assert "IDENTIFICATION" in text


# --- The logistic cross-check ------------------------------------------------


def test_horizon_label_excludes_outcomes_unknown_at_the_horizon():
    """A listing withdrawn before H has no outcome at H and must not become a zero."""
    frame = pd.DataFrame(
        {
            "duration_days": [30.0, 200.0, 30.0, 200.0, 180.0],
            "event_sold": [1, 1, 0, 0, 0],
        }
    )
    label = build_horizon_label(frame, 180)

    assert label.y.tolist()[:1] == [1.0]
    # Sold, but on day 200: it did not sell within the horizon.
    assert label.y.iloc[1] == 0.0
    # Withdrawn on day 30: unknowable at 180 days, so excluded rather than scored 0.
    assert pd.isna(label.y.iloc[2])
    assert not label.usable.iloc[2]
    assert label.y.iloc[3] == 0.0
    assert label.y.iloc[4] == 0.0
    assert label.n_sold_by_horizon == 1
    assert label.n_survived_horizon == 3
    assert label.n_unknown_at_horizon == 1


def test_logistic_agrees_with_the_cox_on_the_sign_of_beta_price(config):
    """The cross-check exists to disagree; when it does, the specification is wrong.

    Magnitudes are not comparable — one is a log-hazard, the other a log-odds —
    so only the sign is asserted.
    """
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, cox_result = _fit_cox(frame)
    model = LogisticDemandModel(HORIZON_DAYS)
    logit_result = model.fit(frame)

    assert logit_result.beta_price.value < 0
    assert cox_result.beta_price.value < 0
    assert logit_result.beta_price.excludes_zero
    assert logit_result.n_observations < len(frame), (
        "listings censored before the horizon must be excluded from the logit"
    )
    assert any("censored before" in note for note in logit_result.notes)


def test_logistic_refuses_a_horizon_it_was_not_fitted_at(config):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model = LogisticDemandModel(HORIZON_DAYS)
    model.fit(frame)
    with pytest.raises(SchemaError, match="fitted at 180 days"):
        model.predict_sale_probability(frame.head(5), 900.0, horizon_days=90)


def test_diagnostics_include_auc_and_calibration_when_the_logit_is_supplied(config):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, cox_result = _fit_cox(frame)
    model = LogisticDemandModel(HORIZON_DAYS)
    model.fit(frame)

    report = run_diagnostics(frame, cox_result, logistic_model=model)
    names = {check.name for check in report.checks}
    assert {"logistic_auc_holdout", "calibration_max_decile_gap"} <= names
    assert len(report.calibration_deciles) == 10
    assert all(0.0 <= row["observed_rate"] <= 1.0 for row in report.calibration_deciles)


# --- Prediction --------------------------------------------------------------


def test_sale_probability_falls_as_the_asking_price_rises(config):
    """The whole point: a higher price must lower the chance of selling."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model, _ = _fit_cox(frame)
    units = frame.dropna(subset=list(DEMAND_COVARIATES)).head(20)

    probabilities = [
        np.nanmean(model.predict_sale_probability(units, price, HORIZON_DAYS))
        for price in (600.0, 900.0, 1200.0, 1600.0)
    ]
    assert probabilities == sorted(probabilities, reverse=True), probabilities
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_prediction_needs_the_comps_median_to_interpret_a_price(config):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model, _ = _fit_cox(frame)
    units = frame.dropna(subset=list(DEMAND_COVARIATES)).head(5).drop(columns=["cell_median_ppsf"])
    with pytest.raises(SchemaError, match="cell_median_ppsf"):
        model.predict_sale_probability(units, 900.0, HORIZON_DAYS)


def test_unseen_categorical_levels_raise_rather_than_defaulting(config):
    """Scoring an unknown submarket as the reference level is a wrong answer."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model, _ = _fit_cox(frame)
    units = frame.dropna(subset=list(DEMAND_COVARIATES)).head(5).copy()
    units["submarket"] = "atlantis"
    with pytest.raises(SchemaError, match="never seen at fit time"):
        model.predict_sale_probability(units, 900.0, HORIZON_DAYS)


def test_rows_with_missing_covariates_score_null_not_a_guess(config):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model, _ = _fit_cox(frame)
    units = frame.dropna(subset=list(DEMAND_COVARIATES)).head(5).copy()
    units.loc[units.index[0], "hoa_per_sqft"] = np.nan

    predicted = model.predict_sale_probability(units, 900.0, HORIZON_DAYS)
    assert np.isnan(predicted[0])
    assert np.isfinite(predicted[1:]).all()


def test_relative_premium_is_dollars_per_sqft_on_both_sides():
    """Guards the units confusion AGENTS.md calls the likeliest silent numeric bug."""
    assert relative_premium(1100.0, 1000.0) == pytest.approx(0.10)
    assert relative_premium(1000.0, 1000.0) == pytest.approx(0.0)
    assert np.isnan(relative_premium(1000.0, 0.0))
    assert np.isnan(relative_premium(1000.0, np.nan))


# --- Hedonic surface ---------------------------------------------------------


def test_hedonic_recovers_the_planted_floor_premium(config):
    """alpha lands near the planted floor elasticity and is measured, not assumed.

    Exact recovery is not expected: the generator multiplies fair value by
    `1 + alpha*ln(floor+1)` while the surface fits `exp(alpha*ln(floor+1))`, so
    the fitted alpha sits slightly below the planted one by the curvature of the
    log. The point of the test is that the number comes from the data and lands
    in the right place, not that two different functional forms coincide.
    """
    synthetic = generate_synthetic_mls(n=N_LISTINGS, seed=SEED, true_beta_price=-1.6)
    frame = build_features(normalize_mls(synthetic.frame, config=config).frame, config).frame
    surface = fit_hedonic(frame, config)
    premium = surface.floor_premium

    planted = synthetic.truth.floor_premium_alpha
    assert premium.alpha > 0
    assert abs(premium.alpha - planted) / planted < 0.25, (
        f"fitted alpha {premium.alpha:.4f} against planted {planted}"
    )
    assert premium.alpha_std_error > 0
    # gamma is the normalization that zeroes the premium at the reference floor.
    assert premium.reference_floor == 1.0
    assert premium.premium(1.0) == pytest.approx(0.0, abs=1e-12)
    assert premium.premium(40.0) > premium.premium(5.0) > 0.0
    assert surface.fit.fit_stats["r_squared"] > 0.5


def test_hedonic_price_bounds_bracket_realized_sale_prices(config):
    """The comps band is wide enough to contain most closings, and no wider."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    surface = fit_hedonic(frame, config)
    sold = frame[frame["close_ppsf"].notna()]
    bounds = surface.price_bounds(sold)

    ordered = bounds.dropna()
    assert (ordered["p_floor_ppsf"] < ordered["predicted_ppsf"]).all()
    assert (ordered["predicted_ppsf"] < ordered["p_ceiling_ppsf"]).all()

    inside = (sold["close_ppsf"] >= bounds["p_floor_ppsf"]) & (
        sold["close_ppsf"] <= bounds["p_ceiling_ppsf"]
    )
    assert 0.70 <= inside.mean() <= 0.95, f"coverage {inside.mean():.3f}"


def test_hedonic_refuses_to_fit_on_too_few_sales(config):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    with pytest.raises(SchemaError, match="at least"):
        fit_hedonic(frame.head(20), config)


def test_hedonic_skips_all_null_view_on_like_export(config):
    """Miami-shaped exports omit view; the surface must still fit without it."""
    frame = _features(
        config,
        true_beta_price=-1.6,
        hazard_basis="realized",
        profile="like_export",
    )
    assert "view_description" in frame.columns
    assert frame["view_description"].isna().all()

    surface = fit_hedonic(frame, config)
    assert "view_description" not in surface.fit.covariates
    assert any("view_description" in note for note in surface.fit.notes)
    assert surface.floor_premium.alpha_std_error > 0


# --- Never impute ------------------------------------------------------------


def test_design_matrix_drops_nulls_and_reports_the_cost(config):
    """Listwise deletion is visible per covariate, and nothing is filled in."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    design = build_design_matrix(frame, DEMAND_COVARIATES)

    assert design.rows_used < design.rows_in
    assert design.dropped_by_null["log_floor"] > 0
    assert design.X.notna().all().all()
    assert 0.0 < design.retention <= 1.0

    surviving = frame.loc[design.index]
    for covariate in DEMAND_COVARIATES:
        assert surviving[covariate].notna().all(), (
            f"{covariate} has nulls inside the fitted sample, so something imputed"
        )


def test_scaling_round_trip_leaves_coefficients_in_raw_units(config):
    """A coefficient on living_area_sqft must mean 'per square foot'."""
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    _, result = _fit_cox(frame)
    area = result.coefficients["living_area_sqft"]
    # Raw-unit coefficients on a variable spanning thousands are necessarily tiny;
    # an unscaled one would come back three orders of magnitude larger.
    assert abs(area.value) < 0.01
    assert result.design.scale["living_area_sqft"] > 100.0


# --- Registry and provenance -------------------------------------------------


def test_saved_bundle_round_trips_and_defaults_to_uncalibrated(config, tmp_path):
    frame = _features(config, true_beta_price=-1.6, hazard_basis="realized")
    model, result = _fit_cox(frame)
    report = run_diagnostics(frame, result)
    bundle = ModelBundle(
        provenance=Provenance(market="miami", planted_truth={"beta_price": -1.6}),
        cox=model,
        cox_result=result,
        diagnostics=report,
    )

    save_bundle(bundle, root=tmp_path)
    restored = load_bundle("miami", root=tmp_path)
    assert restored.beta_price == pytest.approx(bundle.beta_price)
    assert restored.provenance.is_calibrated_on_real_data is False
    assert restored.provenance.data_source == "synthetic"

    metadata = load_metadata("miami", root=tmp_path)
    assert metadata["provenance"]["is_calibrated_on_real_data"] is False
    assert metadata["beta_price"]["value"] == pytest.approx(bundle.beta_price)
    assert metadata["diagnostics"]["status"] in {"PASS", "WARN", "FAILED"}


def test_provenance_cannot_claim_real_calibration_on_synthetic_data():
    """The banner flag and the data source are not allowed to disagree."""
    with pytest.raises(SchemaError, match="cannot disagree"):
        Provenance(market="miami", data_source="synthetic", is_calibrated_on_real_data=True)
    honest = Provenance(market="miami", data_source=REAL_MLS, is_calibrated_on_real_data=True)
    assert honest.is_calibrated_on_real_data is True
    assert honest.fitted_at
    assert honest.library_versions["lifelines"]


def test_loading_a_missing_bundle_says_so(tmp_path):
    with pytest.raises(SchemaError, match="No fitted model"):
        load_bundle("miami", root=tmp_path)
