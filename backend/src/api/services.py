"""Orchestration behind the routes — no FastAPI types here.

Side effects (fitting, saving bundles) stay in `demand/fit` and `registry`.
This module builds the objects those layers already know how to consume.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

from src.api.schemas import (
    ConstraintInput,
    DemandCurveRequest,
    FitBody,
    OptimizeRequest,
    PhaseInput,
    PlanRowInput,
    ScenarioInput,
    SensitivityRequest,
    SimulateRequest,
)
from src.config import MarketConfig, load_market_config
from src.demand.fit import FitRequest, fit_demand
from src.demand.registry import ModelBundle, load_bundle, load_metadata
from src.exceptions import InfeasibleModelError, SchemaError
from src.optimizer.constraints import ConstraintSet
from src.optimizer.crowding import apply_crowding, crowded_result
from src.optimizer.discretize import build_price_ladder
from src.optimizer.formulate import (
    Comps,
    DemandProvenance,
    Phase,
    ProjectSpec,
    build_revenue_tensor,
)
from src.optimizer.solve import (
    OptimizeResult,
    PlanRow,
    PhaseSummary,
    SolveStatus,
    solve_release_plan,
)
from src.simulation.monte_carlo import RevenueDistribution, simulate_plan
from src.simulation.scenarios import ScenarioSpec
from src.simulation.sensitivity import ShadowPriceReport, TornadoReport, shadow_prices, tornado
from src.utils.validate import InventoryValidation, validate_inventory

logger = logging.getLogger(__name__)


def inventory_frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def inventory_frame_from_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def inventory_frame_from_xlsx_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw))


def validate_rows(
    rows: list[dict[str, Any]] | pd.DataFrame, market: str
) -> InventoryValidation:
    config = load_market_config(market)
    frame = rows if isinstance(rows, pd.DataFrame) else inventory_frame_from_rows(rows)
    return validate_inventory(frame, config)


def require_bundle(market: str, *, name: str = "current") -> ModelBundle:
    """Load the active bundle or explain that a fit is required first."""
    try:
        return load_bundle(market, name=name)
    except SchemaError as exc:
        raise SchemaError(
            f"{exc} Call POST /api/demand/fit with dataset=synthetic first."
        ) from exc


def current_demand_metadata(market: str, *, name: str = "current") -> dict[str, Any]:
    try:
        return load_metadata(market, name=name)
    except SchemaError as exc:
        raise SchemaError(
            f"{exc} Call POST /api/demand/fit with dataset=synthetic first."
        ) from exc


def run_fit(body: FitBody) -> dict[str, Any]:
    outcome = fit_demand(
        FitRequest(
            dataset=body.dataset,
            market=body.market,
            calibration_gate=body.calibration_gate,
            assert_calibrated=body.assert_calibrated,
            n_listings=body.n_listings,
            seed=body.seed,
            controls=body.controls,
            building_fe=body.building_fe,
            horizon_days=body.horizon_days,
            name=body.name,
        )
    )
    return outcome.as_dict()


def _phases(inputs: list[PhaseInput]) -> tuple[Phase, ...]:
    if not inputs:
        raise SchemaError("At least one phase is required")
    return tuple(
        Phase(
            index=j,
            name=item.name,
            start_month=float(item.start_month),
            cash_flow_floor_usd=float(item.cash_flow_floor_usd),
            max_units=item.max_units,
            min_type_counts=item.min_type_counts,
            competing_listings=item.competing_listings,
        )
        for j, item in enumerate(inputs)
    )


def _constraint_set(inp: ConstraintInput) -> ConstraintSet:
    return ConstraintSet(
        at_most_once=inp.at_most_once,
        monotone_price_path=inp.monotone_price_path,
        cash_flow_floor=inp.cash_flow_floor,
        max_units_per_phase=inp.max_units_per_phase,
        construction_gate=inp.construction_gate,
        type_diversity=inp.type_diversity,
        cash_flow_basis=inp.cash_flow_basis,
    )


def _project_spec(
    config: MarketConfig,
    *,
    project_start: str,
    phases: list[PhaseInput],
    comps: Comps,
    discount_rate_annual: float | None,
    presale_lead_months: int | None,
    horizon_days: int | None,
) -> ProjectSpec:
    overrides: dict[str, Any] = {}
    if discount_rate_annual is not None:
        overrides["discount_rate_annual"] = discount_rate_annual
    if presale_lead_months is not None:
        overrides["presale_lead_months"] = presale_lead_months
    if horizon_days is not None:
        overrides["horizon_days"] = horizon_days
    return ProjectSpec.from_config(
        config,
        project_start=pd.Timestamp(project_start),
        phases=_phases(phases),
        comps=comps,
        **overrides,
    )


def _provenance_from_bundle(bundle: ModelBundle) -> DemandProvenance:
    return DemandProvenance.from_bundle(bundle)


def _build_context(
    market: str,
    inventory: list[dict[str, Any]],
    phases: list[PhaseInput],
    *,
    project_start: str,
    comps: dict[str, float] | None,
    discount_rate_annual: float | None,
    presale_lead_months: int | None,
    horizon_days: int | None,
) -> tuple[ModelBundle, pd.DataFrame, Any, ProjectSpec, Any]:
    config = load_market_config(market)
    bundle = require_bundle(market)
    if bundle.hedonic is None:
        raise SchemaError(
            "The active demand bundle has no hedonic surface. Re-fit with "
            "POST /api/demand/fit so price bounds can be built."
        )
    validation = validate_rows(inventory, market)
    if not validation.is_valid:
        raise SchemaError(
            "Inventory is not valid. Call POST /api/inventory/validate and fix "
            f"the {len(validation.errors)} row error(s) first."
        )
    units = validation.frame
    resolved = dict(comps or {})
    if not resolved:
        planted = bundle.provenance.planted_truth or {}
        hint = planted.get("comps_ppsf_by_submarket") or {}
        if not hint:
            raise SchemaError(
                "comps (submarket → median $/sqft) is required. Pass the map, "
                "or re-fit so the bundle stores fit-time submarket medians."
            )
        resolved = {str(k): float(v) for k, v in hint.items()}
    missing = sorted(set(units["submarket"].astype(str)) - set(resolved))
    if missing:
        raise SchemaError(
            f"comps is missing submarket(s) {missing}. Every inventory "
            "submarket needs a median $/sqft."
        )
    comps = resolved
    spec = _project_spec(
        config,
        project_start=project_start,
        phases=phases,
        comps=Comps(comps),
        discount_rate_annual=discount_rate_annual,
        presale_lead_months=presale_lead_months,
        horizon_days=horizon_days,
    )
    ladder = build_price_ladder(units, bundle.hedonic, config)
    tensor = build_revenue_tensor(units, ladder, bundle.cox, spec)
    return bundle, units, ladder, spec, tensor


def run_optimize(body: OptimizeRequest) -> dict[str, Any]:
    bundle, units, ladder, spec, tensor = _build_context(
        body.market,
        body.inventory,
        body.phases,
        project_start=body.project_start,
        comps=body.comps,
        discount_rate_annual=body.discount_rate_annual,
        presale_lead_months=body.presale_lead_months,
        horizon_days=body.horizon_days,
    )
    config = load_market_config(body.market)
    provenance = _provenance_from_bundle(bundle)
    result = solve_release_plan(
        tensor, units, provenance, constraints=_constraint_set(body.constraints)
    )
    if body.crowding_buyer_pool is not None and result.status is SolveStatus.OPTIMAL:
        result = crowded_result(
            result,
            apply_crowding(
                result, units, buyer_pool=body.crowding_buyer_pool, config=config
            ),
        )
    if result.status is SolveStatus.INFEASIBLE:
        raise InfeasibleModelError(
            result.infeasibility.message
            if result.infeasibility
            else "No feasible release plan under the given constraints."
        )
    payload = result.as_dict()
    # Echo the comps the solve actually used, not the ones the caller sent. When
    # the request omits them they are resolved from the bundle's fit-time
    # submarket medians, and echoing the empty request body would hand the client
    # back nothing to pass to /api/simulate — which would then re-resolve and
    # could disagree with what this plan was priced against.
    payload["comps"] = dict(spec.comps.median_ppsf_by_submarket)
    payload["project_start"] = body.project_start
    payload["phases"] = [p.as_dict() for p in spec.phases]
    payload["ladder_summary"] = ladder.summary()
    return payload


def _plan_rows(plan: list[PlanRowInput]) -> list[PlanRow]:
    return [
        PlanRow(
            unit_id=r.unit_id,
            phase_index=r.phase_index,
            phase_name=r.phase_name,
            level_index=r.level_index,
            price_ppsf=r.price_ppsf,
            total_price_usd=r.total_price_usd,
            sale_probability=r.sale_probability,
            expected_revenue_usd=r.expected_revenue_usd,
            discounted_expected_revenue_usd=r.discounted_expected_revenue_usd,
        )
        for r in plan
    ]


def _synthetic_result(
    plan: list[PlanRowInput],
    phases: tuple[Phase, ...],
    provenance: dict[str, Any],
    cash_flow_basis: str = "nominal",
) -> OptimizeResult:
    """Rebuild enough of an OptimizeResult for simulate/sensitivity."""
    rows = _plan_rows(plan)
    per_phase: list[PhaseSummary] = []
    for phase in phases:
        members = [r for r in rows if r.phase_index == phase.index]
        revenue = sum(r.expected_revenue_usd for r in members)
        discounted = sum(r.discounted_expected_revenue_usd for r in members)
        # Match the basis the plan was solved under; measuring a discounted
        # floor against nominal revenue reports a breach that is not one.
        measured = discounted if cash_flow_basis == "discounted" else revenue
        per_phase.append(
            PhaseSummary(
                index=phase.index,
                name=phase.name,
                start_month=phase.start_month,
                units_released=len(members),
                expected_revenue_usd=revenue,
                discounted_expected_revenue_usd=discounted,
                cash_flow_floor_usd=phase.cash_flow_floor_usd,
                floor_met=measured >= phase.cash_flow_floor_usd - 0.01,
                mean_price_ppsf=(
                    float(sum(r.price_ppsf for r in members) / len(members))
                    if members
                    else None
                ),
            )
        )
    return OptimizeResult(
        status=SolveStatus.OPTIMAL,
        plan=rows,
        objective_usd=sum(r.discounted_expected_revenue_usd for r in rows),
        expected_revenue_usd=sum(r.expected_revenue_usd for r in rows),
        per_phase=per_phase,
        unreleased_unit_ids=[],
        excluded_units=[],
        provenance=provenance,
        constraints={},
        caveats=[],
        solve_seconds=0.0,
    )


def _scenario_spec(bundle: ModelBundle, scenario: ScenarioInput) -> ScenarioSpec:
    beta = bundle.cox_result.beta_price
    se = (
        float(scenario.beta_price_se)
        if scenario.beta_price_se is not None
        else float(beta.std_error)
    )
    return ScenarioSpec(
        beta_price_mean=float(beta.value),
        beta_price_se=se,
        absorption_log_hazard_sd=scenario.absorption_log_hazard_sd,
        competing_listings_sd=scenario.competing_listings_sd,
        comps_drift_sd=scenario.comps_drift_sd,
        completion_delay_months_sd=scenario.completion_delay_months_sd,
    )


def run_simulate(body: SimulateRequest) -> RevenueDistribution:
    bundle, units, ladder, spec, tensor = _build_context(
        body.market,
        body.inventory,
        body.phases,
        project_start=body.project_start,
        comps=body.comps,
        discount_rate_annual=body.discount_rate_annual,
        presale_lead_months=body.presale_lead_months,
        horizon_days=body.horizon_days,
    )
    if not body.plan:
        raise SchemaError("simulate requires a non-empty plan from /api/optimize")
    provenance = _provenance_from_bundle(bundle).as_response_block()
    result = _synthetic_result(
        body.plan, spec.phases, provenance, cash_flow_basis=body.cash_flow_basis
    )
    defaults = load_market_config(body.market).get("defaults") or {}
    n_draws = int(body.n_draws or defaults.get("monte_carlo_draws") or 10_000)
    hazard = {name: float(c.value) for name, c in bundle.cox_result.coefficients.items()}
    return simulate_plan(
        result,
        tensor,
        units,
        _scenario_spec(bundle, body.scenario),
        n_draws=n_draws,
        seed=body.seed,
        presale_lead_months=spec.presale_lead_months,
        phase_dates=spec.phase_dates(),
        hazard_coefficients=hazard,
        ladder=ladder,
    )


def run_sensitivity(
    body: SensitivityRequest,
) -> tuple[TornadoReport, ShadowPriceReport | None]:
    bundle, units, ladder, spec, tensor = _build_context(
        body.market,
        body.inventory,
        body.phases,
        project_start=body.project_start,
        comps=body.comps,
        discount_rate_annual=body.discount_rate_annual,
        presale_lead_months=body.presale_lead_months,
        horizon_days=body.horizon_days,
    )
    del ladder
    if not body.plan:
        raise SchemaError("sensitivity requires a non-empty plan from /api/optimize")
    provenance = _provenance_from_bundle(bundle).as_response_block()
    result = _synthetic_result(
        body.plan, spec.phases, provenance, cash_flow_basis=body.cash_flow_basis
    )
    hazard = {name: float(c.value) for name, c in bundle.cox_result.coefficients.items()}
    tornado_report = tornado(
        result,
        tensor,
        units,
        _scenario_spec(bundle, body.scenario),
        presale_lead_months=spec.presale_lead_months,
        phase_dates=spec.phase_dates(),
        hazard_coefficients=hazard,
    )
    shadows = None
    if body.include_shadow_prices:
        shadows = shadow_prices(
            tensor,
            units,
            integer_objective_usd=result.objective_usd,
            provenance=provenance,
        )
    return tornado_report, shadows


def run_demand_curve(body: DemandCurveRequest) -> dict[str, Any]:
    """Sale probability at each ladder price for one unit in one phase."""
    bundle, units, ladder, spec, tensor = _build_context(
        body.market,
        body.inventory,
        body.phases,
        project_start=body.project_start,
        comps=body.comps,
        discount_rate_annual=body.discount_rate_annual,
        presale_lead_months=body.presale_lead_months,
        horizon_days=body.horizon_days,
    )
    del bundle
    if body.unit_id not in tensor.unit_ids:
        raise SchemaError(
            f"unit_id {body.unit_id!r} is not in the scored inventory "
            f"(excluded or missing). Check /api/optimize excluded_units."
        )
    if not (0 <= body.phase_index < tensor.n_phases):
        raise SchemaError(
            f"phase_index {body.phase_index} out of range for "
            f"{tensor.n_phases} phases"
        )
    i = tensor.unit_ids.index(body.unit_id)
    j = body.phase_index
    prices = [float(p) for p in tensor.price_ppsf[i]]
    probabilities = [float(p) for p in tensor.probability[i, j, :]]
    row = units.loc[units["unit_id"].astype(str) == body.unit_id].iloc[0]
    return {
        "unit_id": body.unit_id,
        "phase_index": j,
        "phase_name": tensor.phases[j].name,
        "floor": int(row["floor"]) if pd.notna(row.get("floor")) else None,
        "living_area_sqft": float(row["living_area_sqft"]),
        "unit_type": str(row.get("unit_type", "")),
        "building": str(row.get("building", "")),
        "prices_ppsf": prices,
        "probabilities": probabilities,
        "p_floor_ppsf": float(ladder.p_floor_ppsf[ladder.index_of(body.unit_id)]),
        "p_ceiling_ppsf": float(ladder.p_ceiling_ppsf[ladder.index_of(body.unit_id)]),
        "horizon_days": spec.horizon_days,
    }


def market_config_payload(market: str) -> dict[str, Any]:
    config = load_market_config(market)
    defaults = dict(config.get("defaults") or {})
    return {
        "market": config["market"],
        "currency": config["currency"],
        "area_unit": config["area_unit"],
        "price_unit": config["price_unit"],
        "submarkets": config["submarkets"],
        "view_categories": config["view_categories"],
        "defaults": defaults,
        "filters": config.get("filters") or {},
        "is_calibrated_on_real_data": bool(
            defaults.get("is_calibrated_on_real_data", False)
        ),
    }
