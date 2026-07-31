"""Pydantic request models for the API.

Responses stay as plain dicts from the domain `as_dict()` methods so the
provenance block is exactly what `DemandProvenance.as_response_block` emits —
no second schema that could drift from `AGENTS.md` §4.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InventoryValidateRequest(BaseModel):
    """JSON inventory — what Phase 7 sends after client-side CSV/XLSX parse."""

    market: str = "miami"
    rows: list[dict[str, Any]]


class FitBody(BaseModel):
    dataset: Literal["synthetic", "mls"]
    market: str = "miami"
    calibration_gate: bool = False
    assert_calibrated: bool = False
    n_listings: int = Field(default=4000, ge=200, le=50_000)
    seed: int = 42
    controls: bool = True
    building_fe: bool = False
    horizon_days: int | None = None
    name: str = "current"


class PhaseInput(BaseModel):
    name: str
    start_month: float
    cash_flow_floor_usd: float = 0.0
    max_units: int | None = None
    min_type_counts: dict[str, int] = Field(default_factory=dict)
    competing_listings: float | None = 30.0


class ConstraintInput(BaseModel):
    at_most_once: bool = True
    monotone_price_path: bool = True
    cash_flow_floor: bool = True
    max_units_per_phase: bool = True
    construction_gate: bool = True
    type_diversity: bool = True
    cash_flow_basis: Literal["nominal", "discounted"] = "nominal"


class OptimizeRequest(BaseModel):
    market: str = "miami"
    project_start: str = "2026-01-01"
    inventory: list[dict[str, Any]]
    phases: list[PhaseInput]
    comps: dict[str, float] | None = None
    discount_rate_annual: float | None = None
    presale_lead_months: int | None = None
    horizon_days: int | None = None
    crowding_buyer_pool: float | None = None
    constraints: ConstraintInput = Field(default_factory=ConstraintInput)


class PlanRowInput(BaseModel):
    unit_id: str
    phase_index: int
    phase_name: str
    level_index: int
    price_ppsf: float
    total_price_usd: float = 0.0
    sale_probability: float = 0.0
    expected_revenue_usd: float = 0.0
    discounted_expected_revenue_usd: float = 0.0


class ScenarioInput(BaseModel):
    """Uncertainty around a plan. Only beta_price's SE is fitted evidence."""

    beta_price_se: float | None = None
    absorption_log_hazard_sd: float = 0.15
    competing_listings_sd: float = 0.0
    comps_drift_sd: float = 0.03
    completion_delay_months_sd: float = 1.0


class SimulateRequest(BaseModel):
    market: str = "miami"
    project_start: str = "2026-01-01"
    inventory: list[dict[str, Any]]
    phases: list[PhaseInput]
    plan: list[PlanRowInput]
    comps: dict[str, float] | None = None
    discount_rate_annual: float | None = None
    presale_lead_months: int | None = None
    horizon_days: int | None = None
    n_draws: int | None = None
    seed: int = 0
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)
    # Must match the basis the plan was optimized under, or the per-phase
    # cash-flow floors are measured against the wrong quantity.
    cash_flow_basis: Literal["nominal", "discounted"] = "nominal"


class SensitivityRequest(BaseModel):
    market: str = "miami"
    project_start: str = "2026-01-01"
    inventory: list[dict[str, Any]]
    phases: list[PhaseInput]
    plan: list[PlanRowInput]
    comps: dict[str, float] | None = None
    discount_rate_annual: float | None = None
    presale_lead_months: int | None = None
    horizon_days: int | None = None
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)
    include_shadow_prices: bool = True
    cash_flow_basis: Literal["nominal", "discounted"] = "nominal"


class DemandCurveRequest(BaseModel):
    """P(sell) vs price for one unit across its comps ladder."""

    market: str = "miami"
    project_start: str = "2026-01-01"
    unit_id: str
    inventory: list[dict[str, Any]]
    phases: list[PhaseInput]
    comps: dict[str, float] | None = None
    discount_rate_annual: float | None = None
    presale_lead_months: int | None = None
    horizon_days: int | None = None
    phase_index: int = 0
