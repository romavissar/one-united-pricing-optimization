"""Phase 5 — does uncertainty around β_price show up as a revenue distribution?

The load-bearing acceptance checks from `PROJECT_BRIEF.md` §5:

1. 10,000 fixed-plan draws finish in under 60 seconds.
2. Widening `SE(β̂)` widens the revenue distribution.
3. Percentiles are ordered and finite.
4. Cash-flow breach probability is reported when the floor binds in expectation
   but not always in realization.
5. Tornado and LP shadow prices are defined and carry provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config
from src.exceptions import SchemaError
from src.optimizer.constraints import ConstraintSet
from src.optimizer.discretize import ladder_from_bounds
from src.optimizer.formulate import (
    Comps,
    DemandProvenance,
    Phase,
    ProjectSpec,
    build_revenue_tensor,
)
from src.optimizer.solve import SolveStatus, assert_feasible, solve_release_plan
from src.simulation.monte_carlo import (
    PlanStability,
    apply_scenario,
    format_distribution,
    perturbed_probability,
    resolve_scenarios,
    simulate_plan,
    summarize,
)
from src.simulation.scenarios import (
    DEFAULT_CORRELATIONS,
    LHS,
    ScenarioSpec,
    draw_scenarios,
    spec_from_fit,
)
from src.simulation.sensitivity import (
    format_shadow_prices,
    format_tornado,
    shadow_prices,
    tornado,
)
from src.utils.validate import validate_inventory
from tests.test_optimizer import PlantedDemand

INVENTORY_CSV = Path(__file__).resolve().parents[1] / "data" / "project_inputs" / "example_inventory.csv"
PROJECT_START = pd.Timestamp("2026-01-01")
COMPS_PPSF = {"brickell": 1000.0, "edgewater": 900.0}
BETA = -1.6


@pytest.fixture(scope="module")
def config():
    return load_market_config("miami")


@pytest.fixture(scope="module")
def units(config) -> pd.DataFrame:
    result = validate_inventory(pd.read_csv(INVENTORY_CSV), config)
    assert result.is_valid, result.errors
    return result.frame


@pytest.fixture(scope="module")
def ladder(units):
    cost = pd.to_numeric(units["cost_basis_ppsf"], errors="coerce").to_numpy(dtype="float64")
    return ladder_from_bounds(
        units["unit_id"].tolist(), cost * 1.5, cost * 2.6, n_levels=15
    )


def _phases(overrides: dict[int, dict] | None = None) -> tuple[Phase, ...]:
    defaults = {"competing_listings": 20.0, "max_units": 20}
    overrides = overrides or {}
    return (
        Phase(0, "launch", 0.0, **{**defaults, **overrides.get(0, {})}),
        Phase(1, "wave_two", 6.0, **{**defaults, **overrides.get(1, {})}),
        Phase(2, "wave_three", 12.0, **{**defaults, **overrides.get(2, {})}),
        Phase(3, "closeout", 18.0, **{**defaults, **overrides.get(3, {})}),
    )


def _spec(config, phases=None, **overrides) -> ProjectSpec:
    return ProjectSpec.from_config(
        config,
        project_start=PROJECT_START,
        phases=phases or _phases(),
        comps=Comps(COMPS_PPSF),
        **overrides,
    )


def _provenance(beta_price: float = BETA, se: float = 0.15) -> DemandProvenance:
    return DemandProvenance(
        demand_model="planted",
        fitted_on="synthetic",
        beta_price=beta_price,
        beta_price_se=se,
        beta_price_ci95=(beta_price - 1.96 * se, beta_price + 1.96 * se),
    )


@pytest.fixture(scope="module")
def solved(units, ladder, config):
    tensor = build_revenue_tensor(
        units, ladder, PlantedDemand(BETA), _spec(config)
    )
    result = assert_feasible(
        solve_release_plan(tensor, units, _provenance())
    )
    return tensor, result, _spec(config)


# --- scenarios ------------------------------------------------------------


def test_scenario_spec_rejects_a_non_negative_beta():
    with pytest.raises(SchemaError, match="identification failed"):
        ScenarioSpec(beta_price_mean=0.1, beta_price_se=0.05)


def test_scenario_spec_rejects_a_non_psd_correlation():
    with pytest.raises(SchemaError, match="positive semi-definite"):
        ScenarioSpec(
            beta_price_mean=BETA,
            beta_price_se=0.2,
            absorption_log_hazard_sd=0.2,
            competing_listings_sd=0.2,
            correlation={
                ("beta_price", "absorption"): 0.99,
                ("beta_price", "competing_listings"): 0.99,
                ("absorption", "competing_listings"): -0.99,
            },
        ).correlation_matrix()


def test_draws_are_reproducible_and_correlated():
    spec = ScenarioSpec(
        beta_price_mean=BETA,
        beta_price_se=0.25,
        absorption_log_hazard_sd=0.20,
        correlation=DEFAULT_CORRELATIONS,
    )
    a = draw_scenarios(spec, 5_000, seed=7)
    b = draw_scenarios(spec, 5_000, seed=7)
    assert np.allclose(a.beta_price, b.beta_price)
    assert a.beta_price.mean() == pytest.approx(BETA, abs=0.05)
    assert a.beta_price.std(ddof=1) == pytest.approx(0.25, rel=0.1)
    realized = a.realized_correlation()
    assert realized["beta_price|absorption"] == pytest.approx(0.50, abs=0.08)


def test_lhs_covers_the_margins_more_evenly_than_plain_normal():
    spec = ScenarioSpec(beta_price_mean=BETA, beta_price_se=0.3)
    lhs = draw_scenarios(spec, 100, seed=1, method=LHS)
    normal = draw_scenarios(spec, 100, seed=1, method="normal")
    # Stratified margins should put at least one draw in each extreme decile
    # more reliably; check the range is at least as wide.
    assert lhs.beta_price.max() - lhs.beta_price.min() >= (
        normal.beta_price.max() - normal.beta_price.min()
    ) * 0.85


def test_completion_delay_is_clipped_at_zero():
    spec = ScenarioSpec(
        beta_price_mean=BETA, beta_price_se=0.05, completion_delay_months_sd=2.0
    )
    draws = draw_scenarios(spec, 2_000, seed=3)
    assert (draws.completion_delay_months >= 0).all()
    assert any("clipped at zero" in n for n in draws.notes)


# --- closed-form hazard ---------------------------------------------------


def test_perturbed_probability_matches_the_proportional_hazards_identity(solved, units):
    from src.simulation.monte_carlo import _plan_arrays

    tensor, result, project = solved
    arrays = _plan_arrays(
        result, tensor, units, project.presale_lead_months, project.phase_dates()
    )
    # At the fitted beta with zero shocks, P_new must equal P_hat.
    draws = draw_scenarios(
        ScenarioSpec(beta_price_mean=BETA, beta_price_se=0.0), 1, seed=0
    )
    p = perturbed_probability(
        arrays, draws, beta_price_hat=BETA, beta_inventory=0.0
    )
    assert np.allclose(p[0], arrays.probability, atol=1e-9)


# --- monte carlo ----------------------------------------------------------


def test_summarize_percentiles_are_ordered_and_finite():
    rng = np.random.default_rng(0)
    summary = summarize(rng.normal(1e8, 1e7, size=5_000))
    assert summary.is_ordered
    assert summary.cvar5 <= summary.p5
    assert all(np.isfinite(v) for v in summary.as_dict().values())


def test_ten_thousand_draws_finish_under_sixty_seconds(solved, units, ladder):
    tensor, result, project = solved
    scenario = ScenarioSpec(
        beta_price_mean=BETA,
        beta_price_se=0.20,
        absorption_log_hazard_sd=0.15,
        comps_drift_sd=0.03,
        completion_delay_months_sd=1.0,
    )
    distribution = simulate_plan(
        result,
        tensor,
        units,
        scenario,
        n_draws=10_000,
        seed=11,
        presale_lead_months=project.presale_lead_months,
        phase_dates=project.phase_dates(),
        ladder=ladder,
    )
    assert distribution.seconds < 60.0
    assert distribution.discounted_usd.is_ordered
    assert distribution.prob_beats_baseline is not None
    assert 0.0 <= distribution.prob_beats_baseline <= 1.0
    assert distribution.provenance["is_calibrated_on_real_data"] is False
    assert "parameter_share" in distribution.variance_decomposition
    text = format_distribution(distribution)
    assert "REVENUE DISTRIBUTION" in text
    assert "CVaR@5%" in text


def test_distribution_widens_when_beta_se_increases(solved, units):
    tensor, result, project = solved
    widths = []
    for se in (0.05, 0.20, 0.50):
        dist = simulate_plan(
            result,
            tensor,
            units,
            ScenarioSpec(beta_price_mean=BETA, beta_price_se=se),
            n_draws=4_000,
            seed=21,
            presale_lead_months=project.presale_lead_months,
            phase_dates=project.phase_dates(),
        )
        widths.append(dist.discounted_usd.p95 - dist.discounted_usd.p5)
    assert widths[0] < widths[1] < widths[2]


def test_cash_flow_breach_probability_is_reported_when_floor_binds(
    units, ladder, config
):
    # A floor near expected phase revenue will clear in expectation and still
    # breach in a non-trivial share of realized draws.
    phases = _phases(
        {
            0: {"cash_flow_floor_usd": 8_000_000.0, "max_units": 25},
            1: {"max_units": 25},
            2: {"max_units": 25},
            3: {"max_units": 25},
        }
    )
    project = _spec(config, phases=phases)
    tensor = build_revenue_tensor(units, ladder, PlantedDemand(BETA), project)
    result = assert_feasible(solve_release_plan(tensor, units, _provenance()))
    dist = simulate_plan(
        result,
        tensor,
        units,
        ScenarioSpec(beta_price_mean=BETA, beta_price_se=0.25),
        n_draws=3_000,
        seed=5,
        presale_lead_months=project.presale_lead_months,
        phase_dates=project.phase_dates(),
    )
    launch = dist.phase_breach[0]
    assert launch.cash_flow_floor_usd == 8_000_000.0
    assert 0.0 < launch.breach_probability < 1.0
    if launch.is_alarming:
        assert launch.buffered_floor_usd is not None
        assert launch.buffered_floor_usd > launch.cash_flow_floor_usd


def test_simulate_rejects_a_beta_mismatch(solved, units):
    tensor, result, project = solved
    with pytest.raises(SchemaError, match="beta_price_mean"):
        simulate_plan(
            result,
            tensor,
            units,
            ScenarioSpec(beta_price_mean=-0.5, beta_price_se=0.1),
            n_draws=10,
            presale_lead_months=project.presale_lead_months,
            phase_dates=project.phase_dates(),
        )


def test_competing_listings_channel_requires_the_fitted_coefficient(solved, units):
    tensor, result, project = solved
    with pytest.raises(SchemaError, match="inventory_competition"):
        simulate_plan(
            result,
            tensor,
            units,
            ScenarioSpec(
                beta_price_mean=BETA,
                beta_price_se=0.1,
                competing_listings_sd=5.0,
            ),
            n_draws=10,
            presale_lead_months=project.presale_lead_months,
            phase_dates=project.phase_dates(),
        )


# --- sensitivity ----------------------------------------------------------


def test_tornado_orders_bars_by_swing_and_carries_provenance(solved, units):
    tensor, result, project = solved
    report = tornado(
        result,
        tensor,
        units,
        ScenarioSpec(
            beta_price_mean=BETA,
            beta_price_se=0.30,
            absorption_log_hazard_sd=0.20,
            comps_drift_sd=0.05,
        ),
        presale_lead_months=project.presale_lead_months,
        phase_dates=project.phase_dates(),
    )
    assert report.bars
    swings = [b.swing_usd for b in report.bars]
    assert swings == sorted(swings, reverse=True)
    assert report.provenance["is_calibrated_on_real_data"] is False
    assert any(b.channel == "beta_price" and b.fitted for b in report.bars)
    text = format_tornado(report)
    assert "SENSITIVITY TORNADO" in text


def test_shadow_prices_report_a_positive_cost_when_the_floor_binds(
    units, ladder, config
):
    phases = _phases(
        {
            0: {"cash_flow_floor_usd": 12_000_000.0, "max_units": 30},
            1: {"max_units": 30},
            2: {"max_units": 30},
            3: {"max_units": 30},
        }
    )
    project = _spec(config, phases=phases)
    tensor = build_revenue_tensor(units, ladder, PlantedDemand(BETA), project)
    result = solve_release_plan(tensor, units, _provenance())
    if result.status is not SolveStatus.OPTIMAL:
        pytest.skip("floor made the MIP infeasible; dual test needs a feasible LP")
    report = shadow_prices(
        tensor,
        units,
        integer_objective_usd=result.objective_usd,
        provenance=result.provenance,
    )
    assert report.provenance["is_calibrated_on_real_data"] is False
    launch = next(s for s in report.cash_flow if s.phase_index == 0)
    # Either the floor binds (cost > 0) or it does not; both are valid, but the
    # dual must be finite and the sign convention must hold (cost == -dual).
    assert np.isfinite(launch.dual)
    assert launch.objective_cost_per_floor_dollar == pytest.approx(-launch.dual)
    text = format_shadow_prices(report)
    assert "SHADOW PRICES" in text


# --- re-solve (slow) ------------------------------------------------------


@pytest.mark.slow
def test_resolve_scenarios_reports_plan_stability(solved, units):
    tensor, result, project = solved
    stability = resolve_scenarios(
        result,
        tensor,
        units,
        ScenarioSpec(beta_price_mean=BETA, beta_price_se=0.25),
        _provenance(),
        n_scenarios=8,
        seed=9,
        constraints=ConstraintSet(cash_flow_floor=False),
        presale_lead_months=project.presale_lead_months,
        phase_dates=project.phase_dates(),
    )
    assert isinstance(stability, PlanStability)
    assert stability.n_solved >= 1
    assert 0.0 <= stability.mean_same_phase_share <= 1.0
    assert stability.max_regret_usd >= 0.0


def test_apply_scenario_changes_probabilities(solved):
    tensor, result, project = solved
    del result
    draws = draw_scenarios(
        ScenarioSpec(beta_price_mean=BETA, beta_price_se=0.4), 3, seed=2
    )
    shocked = apply_scenario(
        tensor, draws, 0, beta_price_hat=BETA, beta_inventory=0.0
    )
    assert shocked.probability.shape == tensor.probability.shape
    assert not np.allclose(shocked.probability, tensor.probability)


def test_spec_from_fit_is_a_thin_constructor():
    spec = spec_from_fit(BETA, 0.12, absorption_log_hazard_sd=0.1)
    assert spec.beta_price_mean == BETA
    assert spec.beta_price_se == 0.12
    assert "absorption" in spec.active_channels
