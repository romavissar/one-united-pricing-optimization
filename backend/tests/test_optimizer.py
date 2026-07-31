"""Phase 4 — does the demand model actually reach the recommended price?

`PROJECT_BRIEF.md` §4 names the failure mode these tests exist to catch: an
optimizer that produces confident, well-formatted prices which are really just
"charge the maximum" with a dashboard around them. Two tests do the load-bearing
work.

The **elasticity-response test** solves the same inventory at `beta_price = -0.4`
and at `-3.0`. If the recommended prices come back identical, the demand
function is not entering the objective and every other test in this file is
passing on a wiring bug.

The **degenerate-input test** solves at `beta_price = 0` and asserts every unit
is pushed to its ceiling. That is not a bug being tested for — it is the correct
answer when demand does not respond to price, and pinning it here documents
exactly what a broken elasticity estimate would look like coming out of the
optimizer.

Most tests use `PlantedDemand`, a closed-form hazard with a known `beta_price`.
That is deliberate: it isolates the optimizer from estimation noise, so a
failure here is an optimizer failure. One end-to-end test runs a real Cox model
fitted on synthetic data through the whole path to confirm the pieces connect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_market_config
from src.data.features import build_features
from src.data.normalize import normalize_mls
from src.data.synth import generate_synthetic_mls
from src.demand.base import PremiumSupport, premium_support, relative_premium
from src.demand.hedonic import fit_hedonic
from src.demand.survival import (
    CONTROLLED_CATEGORICALS,
    CoxDemandModel,
    available_covariates,
)
from src.exceptions import InfeasibleModelError, SchemaError
from src.optimizer.constraints import ConstraintSet, comparable_groups
from src.optimizer.crowding import apply_crowding, crowded_result, crowding_factor
from src.optimizer.discretize import (
    FLOOR_FROM_COMPS,
    FLOOR_FROM_COST,
    build_price_ladder,
    ladder_from_bounds,
)
from src.optimizer.formulate import (
    Comps,
    DemandProvenance,
    Phase,
    ProjectSpec,
    build_phase_features,
    build_revenue_tensor,
)
from src.optimizer.solve import (
    SolveStatus,
    assert_feasible,
    check_monotone,
    format_plan,
    solve_release_plan,
)
from src.utils.npv import discount_factor, discount_factors, npv
from src.utils.validate import inventory_scoring_frame, validate_inventory

INVENTORY_CSV = Path(__file__).resolve().parents[1] / "data" / "project_inputs" / "example_inventory.csv"
PROJECT_START = pd.Timestamp("2026-01-01")
COMPS_PPSF = {"brickell": 1000.0, "edgewater": 900.0}


class PlantedDemand:
    """A hazard with a known `beta_price`, and nothing else going on.

    `P(sale within H) = 1 - exp(-base * exp(beta * rel_price_premium))`, where
    `rel_price_premium` is computed the same way the Cox model computes it. No
    quality terms, no noise. When the optimizer's prices move as `beta` changes,
    that movement is the objective responding to elasticity and not to anything
    else.
    """

    kind = "planted"

    def __init__(
        self,
        beta_price: float,
        base_hazard: float = 0.7,
        support: PremiumSupport | None = None,
    ) -> None:
        self.beta_price = float(beta_price)
        self.base_hazard = float(base_hazard)
        self.premium_support = support

    def predict_sale_probability(
        self, features: pd.DataFrame, price_ppsf: float | np.ndarray, horizon_days: int
    ) -> np.ndarray:
        del horizon_days
        median = features["cell_median_ppsf"].to_numpy(dtype="float64")
        rel = relative_premium(price_ppsf, median)
        return 1.0 - np.exp(-self.base_hazard * np.exp(self.beta_price * rel))


@pytest.fixture(scope="module")
def config():
    return load_market_config("miami")


@pytest.fixture(scope="module")
def inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV)


@pytest.fixture(scope="module")
def units(inventory, config) -> pd.DataFrame:
    result = validate_inventory(inventory, config)
    assert result.is_valid, result.errors
    return result.frame


@pytest.fixture(scope="module")
def ladder(units):
    """A comps band built from cost basis and a fixed spread, not from a fit.

    The optimizer tests want a band whose endpoints they control. The
    surface-derived path is exercised separately in the end-to-end test.
    """
    cost = pd.to_numeric(units["cost_basis_ppsf"], errors="coerce").to_numpy(dtype="float64")
    return ladder_from_bounds(
        units["unit_id"].tolist(), cost * 1.5, cost * 2.6, n_levels=15
    )


def _phases(**overrides) -> tuple[Phase, ...]:
    defaults = {"competing_listings": 20.0}
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


def _tensor(units, ladder, beta_price, config, phases=None, **overrides):
    return build_revenue_tensor(
        units, ladder, PlantedDemand(beta_price), _spec(config, phases, **overrides)
    )


def _provenance(beta_price: float) -> DemandProvenance:
    return DemandProvenance(
        demand_model="planted", fitted_on="synthetic", beta_price=beta_price
    )


def _solve(units, ladder, beta_price, config, *, constraints=None, phases=None, **overrides):
    tensor = _tensor(units, ladder, beta_price, config, phases, **overrides)
    return tensor, solve_release_plan(
        tensor, units, _provenance(beta_price), constraints=constraints
    )


# --- discounting ---------------------------------------------------------


def test_discounting_uses_an_annual_decimal_rate():
    assert discount_factor(0, 0.12) == pytest.approx(1.0)
    assert discount_factor(12, 0.12) == pytest.approx(1 / 1.12)
    assert discount_factor(18, 0.12) == pytest.approx(1 / 1.12**1.5)
    assert npv([100.0, 100.0], [0.0, 12.0], 0.12) == pytest.approx(100 + 100 / 1.12)


def test_a_percentage_discount_rate_is_rejected_rather_than_silently_used():
    # 12 instead of 0.12 makes every future phase worthless and pushes the
    # optimizer to release everything at once, which looks like an answer.
    with pytest.raises(SchemaError, match="percentage"):
        discount_factor(12, 12.0)
    with pytest.raises(SchemaError):
        discount_factors([0.0, 6.0], 100.0)


# --- inventory validation ------------------------------------------------


def test_the_example_inventory_is_sixty_clean_units(units):
    assert len(units) == 60
    assert units["unit_id"].is_unique
    assert set(units["unit_type"]) == {"one_bed", "two_bed", "three_bed", "penthouse"}


def test_bad_rows_come_back_named_rather_than_raising(units):
    broken = units.head(5).copy().reset_index(drop=True)
    broken["completion_date"] = broken["completion_date"].astype("object")
    broken.loc[0, "living_area_sqft"] = 12.0
    broken.loc[1, "cost_basis_ppsf"] = None
    broken.loc[2, "unit_id"] = broken.loc[3, "unit_id"]
    broken.loc[4, "completion_date"] = "not a date"

    result = validate_inventory(broken, load_market_config("miami"))

    assert not result.is_valid
    assert result.rows_valid == 1
    fields = {(e.row_number, e.field) for e in result.errors}
    assert (1, "living_area_sqft") in fields
    assert (2, "cost_basis_ppsf") in fields
    assert (5, "completion_date") in fields
    assert any(e.field == "unit_id" and "duplicate" in e.message for e in result.errors)


def test_a_missing_column_is_a_malformed_file_not_a_row_error(units):
    with pytest.raises(SchemaError, match="cost_basis_ppsf"):
        validate_inventory(units.drop(columns=["cost_basis_ppsf"]))


def test_an_empty_upload_does_not_report_as_clean(units, config):
    """Zero rows and zero errors is a parse failure, not a valid inventory."""
    result = validate_inventory(units.iloc[:0], config)

    assert result.rows_valid == 0
    assert result.errors == []
    assert result.is_valid is False
    assert any("no rows" in w for w in result.warnings)


def test_an_unknown_submarket_is_rejected_against_market_config(units, config):
    stray = units.head(2).copy().reset_index(drop=True)
    stray.loc[0, "submarket"] = "manhattan"
    result = validate_inventory(stray, config)
    assert result.rows_valid == 1
    assert any(e.field == "submarket" for e in result.errors)


def test_the_scoring_shim_renames_without_inventing(units):
    scored = inventory_scoring_frame(units)
    assert scored["baths_full"].notna().all()
    assert scored["hoa_per_sqft"].between(0.5, 2.5).all()
    assert scored["view_description"].equals(units["view"])
    # No 'year_built' anywhere in a pre-construction inventory, and the flag is
    # a definition rather than an inference.
    assert bool(scored["is_new_construction"].all())


# --- price ladder --------------------------------------------------------


def test_the_ladder_is_ascending_and_inside_its_band(ladder):
    steps = np.diff(ladder.levels_ppsf, axis=1)
    assert (steps > 0).all()
    assert np.allclose(ladder.levels_ppsf[:, 0], ladder.p_floor_ppsf)
    assert np.allclose(ladder.levels_ppsf[:, -1], ladder.p_ceiling_ppsf)


def test_a_unit_whose_cost_exceeds_comps_is_excluded_and_named(units, config):
    class FlatSurface:
        """A surface that prices everything at $600/sqft with a tight band."""

        residual_sd = 0.05
        bound_sd_multiple = 1.0
        design = type("D", (), {"reference_levels": {}, "categorical_levels": {}})()

        def price_bounds(self, frame):
            predicted = np.full(len(frame), 600.0)
            return pd.DataFrame(
                {
                    "predicted_ppsf": predicted,
                    "p_floor_ppsf": predicted * 0.95,
                    "p_ceiling_ppsf": predicted * 1.05,
                },
                index=frame.index,
            )

    # Bases run 470-540 $/sqft, so a 15% margin lands between $540 and $621
    # against a $570-$630 comps band: the expensive units are squeezed out, the
    # cheap ones keep a comps floor, and the ones in between are cost-bound.
    ladder = build_price_ladder(units, FlatSurface(), config, min_margin_over_cost=0.15)

    assert ladder.excluded
    assert all(e.reason == "cost_exceeds_comps" for e in ladder.excluded)
    assert all("does not clear at market" in e.detail for e in ladder.excluded)
    assert ladder.n_units + len(ladder.excluded) == len(units)
    assert ladder.cost_bound_units
    assert set(ladder.floor_source) == {FLOOR_FROM_COST, FLOOR_FROM_COMPS}
    assert any("basis" in note for note in ladder.notes)


def test_a_ladder_with_fewer_than_two_levels_is_refused(units, config):
    with pytest.raises(SchemaError, match="at least 2"):
        build_price_ladder(units, _null_surface(), config, n_levels=1)


def _null_surface():
    class Surface:
        design = type("D", (), {"reference_levels": {}, "categorical_levels": {}})()

        def price_bounds(self, frame):
            return pd.DataFrame(
                {
                    "predicted_ppsf": np.full(len(frame), 1000.0),
                    "p_floor_ppsf": np.full(len(frame), 800.0),
                    "p_ceiling_ppsf": np.full(len(frame), 1200.0),
                },
                index=frame.index,
            )

    return Surface()


# --- the revenue tensor --------------------------------------------------


def test_revenue_is_price_times_area_times_probability_times_discount(units, ladder, config):
    tensor = _tensor(units, ladder, -1.5, config)
    i, j, k = 3, 2, 7
    expected = (
        tensor.price_ppsf[i, k] * tensor.area_sqft[i] * tensor.probability[i, j, k]
    )
    assert tensor.expected_revenue_usd[i, j, k] == pytest.approx(expected)
    assert tensor.discounted_usd[i, j, k] == pytest.approx(
        expected * discount_factor(tensor.phases[j].start_month, 0.12)
    )


def test_the_construction_gate_blocks_phases_before_presale_opens(units, ladder, config):
    tensor = _tensor(units, ladder, -1.5, config)
    # Bayline North completes 2028-03-31; with a 24-month lead it cannot be
    # presold until 2026-03-31, which is after the launch phase opens.
    late = [i for i, u in enumerate(tensor.unit_ids) if u.startswith("BN-")]
    early = [i for i, u in enumerate(tensor.unit_ids) if u.startswith("MT-")]
    assert late and early
    assert not tensor.releasable[late, 0].any()
    assert tensor.releasable[late, 1:].all()
    assert tensor.releasable[early].all()


def test_a_unit_the_model_cannot_score_is_excluded_not_guessed(units, ladder, config):
    class PartlyBlind(PlantedDemand):
        def predict_sale_probability(self, features, price_ppsf, horizon_days):
            out = super().predict_sale_probability(features, price_ppsf, horizon_days)
            out[0] = np.nan
            return out

    tensor = build_revenue_tensor(units, ladder, PartlyBlind(-1.5), _spec(config))

    assert tensor.n_units == ladder.n_units - 1
    assert [e.reason for e in tensor.excluded] == ["unscorable"]
    assert np.isfinite(tensor.discounted_usd).all()


def test_a_submarket_with_no_comp_median_is_named_not_silently_dropped(units, ladder, config):
    """The bug this pins: one absent key used to look like unpriceable units.

    A missing comp median propagates as a NaN `rel_price_premium`, then a NaN
    sale probability, and the unit leaves the plan reported as "the demand model
    could not score it" — which sends the user through their inventory columns
    when the fault is one key in the comps mapping.
    """
    spec = ProjectSpec.from_config(
        config,
        project_start=PROJECT_START,
        phases=_phases(),
        comps=Comps({"brickell": 1000.0}),
    )
    with pytest.raises(SchemaError, match="edgewater"):
        build_revenue_tensor(units, ladder, PlantedDemand(-1.6), spec)


def test_comps_drift_is_zero_unless_the_caller_asks_for_it(units, config):
    flat = Comps(COMPS_PPSF)
    assert flat.median_at("brickell", 24.0) == pytest.approx(1000.0)
    rising = Comps(COMPS_PPSF, monthly_drift=0.005)
    assert rising.median_at("brickell", 24.0) == pytest.approx(1000.0 * 1.005**24)

    features = build_phase_features(units, Phase(1, "p", 12.0), _spec(config))
    assert (features["cell_median_ppsf"] > 0).all()
    assert set(features["season"]) == {"winter"}


# --- the two tests the phase exists for ----------------------------------


def test_elastic_demand_produces_materially_lower_prices(units, ladder, config):
    """PROJECT_BRIEF §4 accept: -0.4 versus -3.0 must move the recommendation."""
    _, inelastic = _solve(units, ladder, -0.4, config)
    _, elastic = _solve(units, ladder, -3.0, config)

    assert inelastic.status is SolveStatus.OPTIMAL
    assert elastic.status is SolveStatus.OPTIMAL

    soft = {r.unit_id: r.price_ppsf for r in inelastic.plan}
    hard = {r.unit_id: r.price_ppsf for r in elastic.plan}
    shared = sorted(set(soft) & set(hard))
    assert len(shared) > 30, "the two solves should release largely the same units"

    ratios = np.array([hard[u] / soft[u] for u in shared])
    assert (ratios <= 1.0).all(), "no unit should be priced higher when demand is elastic"
    assert ratios.mean() < 0.95, (
        f"mean price ratio {ratios.mean():.3f} — elasticity is barely reaching the "
        "objective. If it were exactly 1.0 the demand function is not wired in at all."
    )


def test_zero_elasticity_pushes_every_unit_to_its_ceiling(units, ladder, config):
    """The failure mode the whole project is built to avoid, made explicit.

    With no price response, revenue is monotone increasing in price and the
    right answer *is* the ceiling. A plan that looks like this in production
    means `beta_price` is not identified, not that prices should be raised.
    """
    _, result = _solve(units, ladder, 0.0, config, constraints=ConstraintSet())

    assert result.status is SolveStatus.OPTIMAL
    assert result.plan
    top = ladder.n_levels - 1
    assert {row.level_index for row in result.plan} == {top}
    assert any("price ceiling" in c for c in result.caveats)


def test_the_charge_the_maximum_alarm_stays_quiet_when_demand_responds(
    units, ladder, config
):
    """The alarm has to discriminate, or it is noise nobody reads."""
    _, elastic = _solve(units, ladder, -1.6, config)
    at_ceiling = sum(1 for r in elastic.plan if r.level_index == ladder.n_levels - 1)

    assert at_ceiling / len(elastic.plan) < 0.8
    assert not any("price ceiling" in c for c in elastic.caveats)


# --- pricing outside the evidence ----------------------------------------


def _solve_with_support(units, ladder, config, support):
    tensor = build_revenue_tensor(
        units, ladder, PlantedDemand(-1.6, support=support), _spec(config)
    )
    return tensor, solve_release_plan(tensor, units, _provenance(-1.6))


def test_prices_beyond_the_fitted_range_are_flagged_as_extrapolation(
    units, ladder, config
):
    """A Cox predictor answers anywhere. Only this check separates answer from evidence.

    The ladder here reaches roughly +0.5 above comps; a model fitted only on
    listings priced within a few points of theirs has no evidence at all up
    there, and the sale probabilities behind those prices are extrapolations
    that carry the same decimal places as predictions.
    """
    narrow = PremiumSupport(low=-0.05, high=0.05, p1=-0.04, p99=0.04, n_observations=2000)
    _, result = _solve_with_support(units, ladder, config, narrow)

    assert result.status is SolveStatus.OPTIMAL
    report = result.extrapolation
    assert report is not None
    assert report.checked is True
    assert report.is_clean is False
    assert report.units_outside
    assert "extrapolations, not predictions" in report.message
    # It leads the caveat list rather than sitting at the bottom of it.
    assert result.caveats[0] == report.message


def test_prices_inside_the_fitted_range_pass_the_check_quietly(units, ladder, config):
    wide = PremiumSupport(low=-0.9, high=2.0, p1=-0.6, p99=1.5, n_observations=2000)
    _, result = _solve_with_support(units, ladder, config, wide)

    report = result.extrapolation
    assert report.is_clean
    assert report.units_outside == []
    assert "well inside the fitted range" in report.message
    assert report.message not in result.caveats


def test_a_model_that_reports_no_support_says_so_instead_of_assuming(
    units, ladder, config
):
    """Absence of the check has to be visible, or it reads as a passed check."""
    tensor, result = _solve_with_support(units, ladder, config, None)

    assert tensor.premium_support is None
    assert any("cannot be checked for extrapolation" in n for n in tensor.notes)
    assert result.extrapolation.checked is False
    assert result.caveats[0] == result.extrapolation.message


def test_a_cox_fit_records_the_range_it_is_evidence_about(config):
    frame = build_features(
        normalize_mls(generate_synthetic_mls(n=1500, seed=7).frame, config=config).frame,
        config,
    ).frame
    model = CoxDemandModel()
    model.fit(frame)

    support = model.premium_support
    assert support is not None
    assert support.low < 0 < support.high
    assert support.low <= support.p1 <= support.p99 <= support.high
    # Measured over the rows that survived listwise deletion, not the raw frame.
    assert support.n_observations == model.result.n_observations

    observed = frame["rel_price_premium"].dropna()
    assert support.outside(np.array([observed.max() + 1.0])).all()
    assert not support.outside(np.array([0.0])).any()


def test_an_empty_fitting_sample_reports_no_support_rather_than_a_fake_one():
    assert premium_support(pd.Series([], dtype="float64")) is None
    assert premium_support(pd.Series([np.nan, np.nan])) is None


# --- constraints hold in the returned plan -------------------------------


def test_the_sixty_unit_inventory_solves_to_optimal_quickly(units, ladder, config):
    phases = (
        Phase(0, "launch", 0.0, cash_flow_floor_usd=8_000_000, max_units=18,
              min_type_counts={"one_bed": 2, "two_bed": 2}, competing_listings=20.0),
        Phase(1, "wave_two", 6.0, cash_flow_floor_usd=8_000_000, max_units=18,
              min_type_counts={"one_bed": 2}, competing_listings=20.0),
        Phase(2, "wave_three", 12.0, cash_flow_floor_usd=6_000_000, max_units=18,
              competing_listings=20.0),
        Phase(3, "closeout", 18.0, max_units=18, competing_listings=20.0),
    )
    tensor, result = _solve(units, ladder, -1.6, config, phases=phases)
    assert_feasible(result)

    assert result.status is SolveStatus.OPTIMAL
    assert result.solve_seconds < 10.0, f"took {result.solve_seconds:.1f}s"

    released = [row.unit_id for row in result.plan]
    assert len(released) == len(set(released)), "a unit was released twice"
    assert result.units_released > 0

    for phase in result.per_phase:
        assert phase.floor_met, f"{phase.name} breached its cash-flow floor"
        assert phase.units_released <= 18

    for row in result.plan:
        i = tensor.unit_ids.index(row.unit_id)
        assert tensor.releasable[i, row.phase_index], f"{row.unit_id} released before its gate"

    assert check_monotone(result, units) == []

    types = units.set_index("unit_id")["unit_type"]
    launch = [r.unit_id for r in result.plan if r.phase_index == 0]
    assert sum(1 for u in launch if types[u] == "one_bed") >= 2
    assert sum(1 for u in launch if types[u] == "two_bed") >= 2


def test_monotonicity_binds_across_a_skipped_phase(units, ladder, config):
    """A group that sits out a phase must not reset its price ladder.

    Consecutive-phase constraints alone would let a group price at the top of
    its band in phase 0, release nothing in phase 1, and come back at the bottom
    in phase 2. All-pairs ordering is what prevents that, and this is the test
    that would catch a regression to the cheaper formulation.
    """
    phases = (
        Phase(0, "launch", 0.0, max_units=4, competing_listings=20.0),
        Phase(1, "wave_two", 6.0, max_units=0, competing_listings=20.0),
        Phase(2, "wave_three", 12.0, max_units=6, competing_listings=20.0),
    )
    _, result = _solve(units, ladder, -1.6, config, phases=phases)

    assert result.status is SolveStatus.OPTIMAL
    assert all(row.phase_index != 1 for row in result.plan)
    assert check_monotone(result, units) == []


def test_monotonicity_is_a_real_constraint_and_costs_real_money(units, ladder, config):
    """Removing it must produce violations, or the formulation is a no-op.

    A constraint that never binds is indistinguishable from one that is wired up
    wrong. Solving the same problem without it should both raise the objective
    and produce plans that violate the rule the constraint encodes.
    """
    phases = tuple(
        Phase(j, f"p{j}", month, max_units=15, competing_listings=20.0)
        for j, month in enumerate((0.0, 6.0, 12.0, 18.0))
    )
    _, constrained = _solve(units, ladder, -1.6, config, phases=phases)
    _, free = _solve(
        units,
        ladder,
        -1.6,
        config,
        phases=phases,
        constraints=ConstraintSet(monotone_price_path=False),
    )

    assert check_monotone(constrained, units) == []
    assert check_monotone(free, units), "the unconstrained plan should cut prices somewhere"
    assert free.objective_usd > constrained.objective_usd


def test_comparable_groups_keep_products_apart(units):
    groups = comparable_groups(units, tuple(units["unit_id"]))
    for name, members in groups.items():
        types = {units.iloc[m]["unit_type"] for m in members}
        assert len(types) == 1, f"{name} mixes unit types: {types}"
    assert any(name.startswith("penthouse|") for name in groups)


# --- infeasibility -------------------------------------------------------


def test_an_impossible_cash_flow_floor_is_named_rather_than_returned_empty(
    units, ladder, config
):
    phases = (
        Phase(0, "launch", 0.0, cash_flow_floor_usd=5_000_000_000, competing_listings=20.0),
        Phase(1, "wave_two", 6.0, competing_listings=20.0),
    )
    _, result = _solve(units, ladder, -1.6, config, phases=phases)

    assert result.status is SolveStatus.INFEASIBLE
    assert result.plan == []
    assert result.infeasibility is not None
    assert result.infeasibility.binding_family == "cash_flow_floor"
    assert "cash_flow_floor" in result.infeasibility.message
    # The empty plan must never read as "release nothing".
    assert any("not a recommendation" in c for c in result.caveats)
    with pytest.raises(InfeasibleModelError, match="cash_flow_floor"):
        assert_feasible(result)


def test_a_diversity_requirement_no_unit_can_satisfy_fails_at_build_time(
    units, ladder, config
):
    phases = (Phase(0, "launch", 0.0, min_type_counts={"townhouse": 1}, competing_listings=20.0),)
    with pytest.raises(SchemaError, match="never be satisfied"):
        _solve(units, ladder, -1.6, config, phases=phases)


def test_dropping_a_structural_constraint_is_refused():
    with pytest.raises(SchemaError, match="structural"):
        ConstraintSet().without("at_most_once")


# --- provenance ----------------------------------------------------------


def test_every_result_carries_provenance_with_the_calibration_flag(units, ladder, config):
    _, result = _solve(units, ladder, -1.6, config)
    block = result.provenance

    assert block["is_calibrated_on_real_data"] is False
    assert block["independence_assumption"] is True
    assert block["crowding_correction_applied"] is False
    assert block["beta_price"] == -1.6
    assert any("illustrative" in w for w in block["warnings"])


def test_the_documented_simplifications_ride_along_in_every_plan(units, ladder, config):
    phases = (
        Phase(0, "launch", 0.0, cash_flow_floor_usd=1_000_000, competing_listings=20.0),
        Phase(1, "wave_two", 6.0, competing_listings=20.0),
    )
    _, result = _solve(units, ladder, -1.6, config, phases=phases)
    text = " ".join(result.caveats)

    assert "independent" in text
    assert "expected revenue" in text
    assert "Competitor pricing is exogenous" in text
    assert "held fixed across every phase" in text
    assert "RELEASE PLAN" in format_plan(result)


# --- crowding ------------------------------------------------------------


def test_the_crowding_factor_is_the_formula_in_the_brief():
    assert crowding_factor(0, 50, 0.3) == pytest.approx(1.0)
    assert crowding_factor(10, 50, 0.3) == pytest.approx(1 - 0.3 * 10 / 50)
    # Floored rather than driven to zero: a swamped cluster sells slower, not never.
    assert crowding_factor(500, 10, 1.0) == pytest.approx(0.10)
    with pytest.raises(SchemaError, match="buyer_pool"):
        crowding_factor(5, 0, 0.3)


def test_crowding_lowers_revenue_and_says_so_in_the_provenance(units, ladder, config):
    _, result = _solve(units, ladder, -1.6, config)
    correction = apply_crowding(result, units, buyer_pool=25.0, config=config)

    assert correction.corrected_revenue_usd < correction.uncorrected_revenue_usd
    assert 0.0 < correction.share_removed < 1.0
    assert correction.worst_cluster is not None
    assert correction.worst_cluster.factor < 1.0

    corrected = crowded_result(result, correction)
    assert corrected.provenance["crowding_correction_applied"] is True
    assert corrected.provenance["independence_assumption"] is False
    assert corrected.expected_revenue_usd == pytest.approx(correction.corrected_revenue_usd)
    assert any("crowding-corrected" in w for w in corrected.provenance["warnings"])
    # The plan itself is unchanged; only the revenue attached to it moves.
    assert [r.unit_id for r in corrected.plan] == [r.unit_id for r in result.plan]
    assert [r.price_ppsf for r in corrected.plan] == [r.price_ppsf for r in result.plan]


def test_a_lambda_of_zero_leaves_revenue_untouched(units, ladder, config):
    _, result = _solve(units, ladder, -1.6, config)
    correction = apply_crowding(result, units, buyer_pool=25.0, lambda_=0.0)
    assert correction.share_removed == pytest.approx(0.0)


def test_crowding_can_reveal_a_cash_flow_floor_that_no_longer_clears(units, ladder, config):
    phases = (
        Phase(0, "launch", 0.0, cash_flow_floor_usd=8_000_000, competing_listings=20.0),
        Phase(1, "wave_two", 6.0, competing_listings=20.0),
    )
    _, result = _solve(units, ladder, -1.6, config, phases=phases)
    assert all(p.floor_met for p in result.per_phase)

    corrected = crowded_result(
        result, apply_crowding(result, units, buyer_pool=6.0, lambda_=0.9)
    )
    breached = [p.name for p in corrected.per_phase if not p.floor_met]
    assert breached, "a severe correction should push the phase below its floor"
    assert any("no longer clear" in w for w in corrected.provenance["warnings"])


# --- end to end with a real fitted model ---------------------------------


@pytest.mark.slow
def test_a_cox_model_fitted_on_synthetic_data_drives_a_real_plan(units, config):
    """The whole path: synthetic export -> features -> Cox + hedonic -> plan."""
    synthetic = generate_synthetic_mls(n=4000, seed=13, profile="rich")
    frame = build_features(normalize_mls(synthetic.frame, config=config).frame, config).frame

    surface = fit_hedonic(frame, config)
    ladder = build_price_ladder(units, surface, config)
    assert ladder.n_units > 0
    assert any("comps priced as of" in note for note in ladder.notes)

    covariates = available_covariates(frame)
    model = CoxDemandModel(covariates=covariates, categoricals=CONTROLLED_CATEGORICALS)
    fit = model.fit(frame)
    assert fit.beta_price.value < 0, "the fixture data must identify a negative beta"

    comps = Comps(
        {
            submarket: float(
                frame.loc[frame["submarket"] == submarket, "cell_median_ppsf"].median()
            )
            for submarket in units["submarket"].unique()
        }
    )
    spec = ProjectSpec.from_config(
        config,
        project_start=PROJECT_START,
        phases=(
            Phase(0, "launch", 0.0, max_units=20, competing_listings=30.0),
            Phase(1, "wave_two", 9.0, max_units=20, competing_listings=30.0),
            Phase(2, "closeout", 18.0, max_units=20, competing_listings=30.0),
        ),
        comps=comps,
    )
    tensor = build_revenue_tensor(units, ladder, model, spec)
    result = solve_release_plan(
        tensor,
        units,
        DemandProvenance(
            demand_model=fit.model_kind,
            beta_price=fit.beta_price.value,
            beta_price_se=fit.beta_price.std_error,
            beta_price_ci95=(fit.beta_price.ci_low, fit.beta_price.ci_high),
        ),
    )

    assert result.status is SolveStatus.OPTIMAL
    assert result.units_released > 0
    assert check_monotone(result, units) == []
    assert result.provenance["is_calibrated_on_real_data"] is False
    # A negative elasticity should keep at least some units off the ceiling.
    assert min(row.level_index for row in result.plan) < tensor.n_levels - 1
    for row in result.plan:
        i = tensor.unit_ids.index(row.unit_id)
        assert ladder.p_floor_ppsf[ladder.index_of(row.unit_id)] <= row.price_ppsf
        assert row.price_ppsf <= ladder.p_ceiling_ppsf[ladder.index_of(row.unit_id)]
        assert 0.0 <= row.sale_probability <= 1.0
        assert tensor.releasable[i, row.phase_index]
