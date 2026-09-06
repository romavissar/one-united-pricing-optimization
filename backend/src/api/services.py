"""Orchestration behind the routes — no FastAPI types here.

Side effects (fitting, saving bundles) stay in `demand/fit` and `registry`.
This module builds the objects those layers already know how to consume.
"""

from __future__ import annotations

import io
import logging
from dataclasses import replace
from typing import Any

import numpy as np
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
from src.data.macro import MacroSnapshot, build_macro_snapshot, fallback_correlation_pairs
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


_MACRO_CHANNEL_FIELDS = (
    "absorption_log_hazard_sd",
    "comps_drift_sd",
    "competing_listings_sd",
)
_INVENTORY_TERM = "inventory_competition"


def _competing_listings_baseline(phases: list[PhaseInput]) -> float:
    """Representative competing-listings level the relative macro swing scales.

    The median of the phases' assumed competing-listings counts. A relative
    listing-count swing measured from the metro series is meaningless as an
    absolute count until it is applied to the level the plan assumes.
    """
    values = [float(p.competing_listings) for p in phases if p.competing_listings is not None]
    if not values:
        return 30.0
    return float(np.median(values))


def _scenario_spec(
    bundle: ModelBundle,
    scenario: ScenarioInput,
    macro: MacroSnapshot,
    *,
    competing_baseline: float,
) -> tuple[ScenarioSpec, dict[str, Any]]:
    """Build the scenario spec, data-driven by default with per-channel override.

    Each macro channel takes its σ from `macro` unless the request supplied an
    explicit number ("input custom"), which wins for that channel only. The
    cross-channel correlations come from `macro` too — estimated among the
    observable channels, documented priors for `beta_price`'s pairs.

    Returns the spec and a resolution block naming, per channel, whether the σ
    was `data` or `custom` — surfaced in provenance so a reader always knows
    which numbers were measured.
    """
    beta = bundle.cox_result.beta_price
    se = (
        float(scenario.beta_price_se)
        if scenario.beta_price_se is not None
        else float(beta.std_error)
    )

    derived = macro.scenario_dispersions(competing_baseline)
    has_inventory_coef = _INVENTORY_TERM in bundle.cox_result.coefficients

    resolution: dict[str, Any] = {}
    values: dict[str, float] = {}
    for field_name in _MACRO_CHANNEL_FIELDS:
        override = getattr(scenario, field_name)
        if override is not None:
            values[field_name] = float(override)
            resolution[field_name] = {"source": "custom", "value": float(override)}
        else:
            values[field_name] = float(derived.get(field_name, 0.0))
            resolution[field_name] = {
                "source": macro.source,
                "value": values[field_name],
            }

    # competing_listings needs a fitted inventory coefficient to reach the
    # hazard. If the model was not fit with it, a data-driven activation would
    # trip the monte_carlo guard; hold the channel at zero and say so rather
    # than derive a σ the simulator cannot use. An explicit user override is
    # left to trip the guard with its own clear message.
    if (
        values["competing_listings_sd"] > 0
        and not has_inventory_coef
        and scenario.competing_listings_sd is None
    ):
        values["competing_listings_sd"] = 0.0
        resolution["competing_listings_sd"] = {
            "source": "disabled",
            "value": 0.0,
            "note": (
                "competing_listings held at zero: the fitted demand model has no "
                f"{_INVENTORY_TERM} coefficient for the shift to act through."
            ),
        }

    spec = ScenarioSpec(
        beta_price_mean=float(beta.value),
        beta_price_se=se,
        absorption_log_hazard_sd=values["absorption_log_hazard_sd"],
        competing_listings_sd=values["competing_listings_sd"],
        comps_drift_sd=values["comps_drift_sd"],
        completion_delay_months_sd=scenario.completion_delay_months_sd,
        correlation=macro.correlation_pairs(),
    )
    # A sample correlation among the macro channels is PSD, but adding
    # beta_price's fixed priors can, for some active-channel sets, tip the joint
    # matrix out of PSD — a world with no joint distribution. Probe it over the
    # channels that are actually active; on failure fall back to the documented
    # PSD priors rather than let the draw raise at simulate time.
    try:
        spec.correlation_matrix()
    except SchemaError:
        logger.warning(
            "estimated macro correlations are not PSD over the active channels; "
            "falling back to documented sign priors."
        )
        spec = replace(spec, correlation=fallback_correlation_pairs())
        resolution["correlation"] = {"source": "fallback_priors_non_psd"}
    else:
        resolution["correlation"] = {"source": macro.source}
    resolution["completion_delay_months_sd"] = {
        "source": "user",
        "value": float(scenario.completion_delay_months_sd),
    }
    resolution["beta_price_se"] = {
        "source": "custom" if scenario.beta_price_se is not None else "fitted",
        "value": se,
    }
    return spec, resolution


def _macro_snapshot_for(market: str, horizon_days: int) -> MacroSnapshot:
    """Build the market's macro snapshot, scaled to the sale horizon.

    Never raises for a network/key problem — `build_macro_snapshot` degrades to
    cache then documented fallback, both labelled in `source`.
    """
    try:
        config = load_market_config(market)
    except SchemaError:
        config = None
    return build_macro_snapshot(market, horizon_days, config=config)


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
    macro = _macro_snapshot_for(body.market, spec.horizon_days)
    scenario_spec, resolution = _scenario_spec(
        bundle,
        body.scenario,
        macro,
        competing_baseline=_competing_listings_baseline(body.phases),
    )
    provenance = _provenance_from_bundle(bundle).as_response_block()
    provenance["macro"] = macro.as_dict()
    provenance["scenario_resolution"] = resolution
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
        scenario_spec,
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
    macro = _macro_snapshot_for(body.market, spec.horizon_days)
    scenario_spec, resolution = _scenario_spec(
        bundle,
        body.scenario,
        macro,
        competing_baseline=_competing_listings_baseline(body.phases),
    )
    provenance = _provenance_from_bundle(bundle).as_response_block()
    provenance["macro"] = macro.as_dict()
    provenance["scenario_resolution"] = resolution
    result = _synthetic_result(
        body.plan, spec.phases, provenance, cash_flow_basis=body.cash_flow_basis
    )
    hazard = {name: float(c.value) for name, c in bundle.cox_result.coefficients.items()}
    tornado_report = tornado(
        result,
        tensor,
        units,
        scenario_spec,
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


def macro_payload(market: str, horizon_days: int | None = None) -> dict[str, Any]:
    """Data-derived macro assumptions for a market — the frontend's default view.

    This is what the "Macro assumptions" panel shows before any user override:
    the σ each channel gets from FRED/BLS, the estimated correlations, the
    context figures, and where they came from. `horizon_days` defaults to the
    market's configured sale horizon so the σ match what optimize/simulate use.
    """
    config = load_market_config(market)
    defaults = config.get("defaults") or {}
    horizon = int(horizon_days or defaults.get("horizon_days") or 180)
    snapshot = build_macro_snapshot(market, horizon, config=config)
    return snapshot.as_dict()


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
