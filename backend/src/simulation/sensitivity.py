"""Local sensitivity of a plan: tornado bars and cash-flow shadow prices.

Two complementary views of the same question — what moves the answer.

**Tornado.** Hold the plan fixed. Move one channel to `mean ± 1σ`, leave the
rest at their means, and record the change in discounted *expected* revenue.
No Bernoulli draws: the tornado is about parameter sensitivity, not sales
lumpiness. Bars are ordered by absolute swing so the largest threat sits at
the top.

**Shadow prices.** Solve the LP relaxation of the same MILP and read the dual
on each cash-flow floor. That dual is how much discounted expected revenue the
covenant is costing per dollar of floor — exactly the number the brief asks
for. Duals are unavailable on the integer programme; the continuous relaxation
is the standard workaround, and the dual is an upper bound on the MIP's true
marginal cost when the integer solution is fractional on the binding cells.

Units: tornado impacts in USD of discounted expected revenue; shadow prices in
USD of objective per USD of floor.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pulp

from src.exceptions import SchemaError
from src.optimizer.constraints import CASH_FLOW_FLOOR, ConstraintSet
from src.optimizer.formulate import RevenueTensor
from src.optimizer.solve import OptimizeResult, SolveStatus, build_program, _solver
from src.simulation.monte_carlo import (
    _alive_under_delay,
    _plan_arrays,
    perturbed_probability,
)
from src.simulation.scenarios import CHANNELS, ScenarioDraws, ScenarioSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TornadoBar:
    """One channel's ±1σ swing on discounted expected revenue (USD)."""

    channel: str
    low_value: float
    high_value: float
    revenue_at_low_usd: float
    revenue_at_high_usd: float
    baseline_usd: float
    fitted: bool

    @property
    def swing_usd(self) -> float:
        return abs(self.revenue_at_high_usd - self.revenue_at_low_usd)

    @property
    def low_delta_usd(self) -> float:
        return self.revenue_at_low_usd - self.baseline_usd

    @property
    def high_delta_usd(self) -> float:
        return self.revenue_at_high_usd - self.baseline_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "swing_usd": self.swing_usd,
            "low_delta_usd": self.low_delta_usd,
            "high_delta_usd": self.high_delta_usd,
        }


@dataclass
class TornadoReport:
    """Tornado diagram data for a fixed plan under `ScenarioSpec`."""

    baseline_usd: float
    bars: list[TornadoBar]
    provenance: dict[str, Any]
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_usd": self.baseline_usd,
            "bars": [b.as_dict() for b in self.bars],
            "provenance": self.provenance,
            "caveats": self.caveats,
        }


@dataclass(frozen=True)
class ShadowPrice:
    """LP dual on one phase's cash-flow floor.

    `objective_cost_per_floor_dollar` is positive when the floor binds and
    costs revenue: raising the floor by $1 costs that many dollars of
    discounted expected revenue in the LP relaxation.
    """

    phase_index: int
    phase_name: str
    cash_flow_floor_usd: float
    dual: float
    objective_cost_per_floor_dollar: float
    binds: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowPriceReport:
    """Cash-flow duals from the LP relaxation, plus any other named duals."""

    objective_usd: float
    integer_objective_usd: float | None
    relaxation_gap_usd: float | None
    cash_flow: list[ShadowPrice]
    provenance: dict[str, Any]
    caveats: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_usd": self.objective_usd,
            "integer_objective_usd": self.integer_objective_usd,
            "relaxation_gap_usd": self.relaxation_gap_usd,
            "cash_flow": [s.as_dict() for s in self.cash_flow],
            "provenance": self.provenance,
            "caveats": self.caveats,
            "seconds": self.seconds,
        }


def _mean_draws(spec: ScenarioSpec) -> ScenarioDraws:
    """A single draw at every channel's mean — the tornado baseline."""
    return ScenarioDraws(
        beta_price=np.array([spec.beta_price_mean], dtype="float64"),
        absorption=np.zeros(1, dtype="float64"),
        competing_listings=np.zeros(1, dtype="float64"),
        comps_drift=np.zeros(1, dtype="float64"),
        completion_delay_months=np.zeros(1, dtype="float64"),
        spec=spec,
        method="point",
        seed=0,
    )


def _channel_point(spec: ScenarioSpec, channel: str, value: float) -> ScenarioDraws:
    draws = _mean_draws(spec)
    if channel == "beta_price":
        draws.beta_price = np.array([value], dtype="float64")
    elif channel == "absorption":
        draws.absorption = np.array([value], dtype="float64")
    elif channel == "competing_listings":
        draws.competing_listings = np.array([value], dtype="float64")
    elif channel == "comps_drift":
        draws.comps_drift = np.array([value], dtype="float64")
    elif channel == "completion_delay_months":
        draws.completion_delay_months = np.array([max(0.0, value)], dtype="float64")
    else:
        raise SchemaError(f"unknown channel {channel!r}; known: {CHANNELS}")
    return draws


def _expected_discounted(
    arrays: Any,
    draws: ScenarioDraws,
    *,
    beta_price_hat: float,
    beta_inventory: float,
) -> float:
    probability = perturbed_probability(
        arrays, draws, beta_price_hat=beta_price_hat, beta_inventory=beta_inventory
    )
    alive = _alive_under_delay(arrays, draws)
    gross = arrays.price_ppsf * arrays.area_sqft * arrays.discount_factor
    return float((probability * alive * gross[None, :]).sum())


def tornado(
    result: OptimizeResult,
    tensor: RevenueTensor,
    units: pd.DataFrame,
    spec: ScenarioSpec,
    *,
    presale_lead_months: float = 24.0,
    phase_dates: list[pd.Timestamp] | None = None,
    hazard_coefficients: dict[str, float] | None = None,
) -> TornadoReport:
    """±1σ one-at-a-time sensitivity of a fixed plan's expected revenue.

    Raises:
        SchemaError: when the plan is not optimal / empty, or when the
            competing-listings channel is active without its fitted coefficient.
    """
    if result.status is not SolveStatus.OPTIMAL or not result.plan:
        raise SchemaError(
            "Tornado requires an optimal plan. There is nothing to perturb."
        )

    coefficients = hazard_coefficients or {}
    beta_inventory = float(coefficients.get("inventory_competition", 0.0))
    if spec.competing_listings_sd > 0 and "inventory_competition" not in coefficients:
        raise SchemaError(
            "competing_listings channel is active but no fitted "
            "inventory_competition coefficient was supplied."
        )

    dates = phase_dates or [pd.Timestamp("1970-01-01")] * len(tensor.phases)
    arrays = _plan_arrays(result, tensor, units, presale_lead_months, dates)
    baseline = _expected_discounted(
        arrays,
        _mean_draws(spec),
        beta_price_hat=spec.beta_price_mean,
        beta_inventory=beta_inventory,
    )

    bars: list[TornadoBar] = []
    for channel in CHANNELS:
        sd = spec.standard_deviations[channel]
        if sd <= 0:
            continue
        mean = spec.beta_price_mean if channel == "beta_price" else 0.0
        low = mean - sd
        high = mean + sd
        if channel == "completion_delay_months":
            low = max(0.0, low)
            high = max(0.0, high)
        rev_low = _expected_discounted(
            arrays,
            _channel_point(spec, channel, low),
            beta_price_hat=spec.beta_price_mean,
            beta_inventory=beta_inventory,
        )
        rev_high = _expected_discounted(
            arrays,
            _channel_point(spec, channel, high),
            beta_price_hat=spec.beta_price_mean,
            beta_inventory=beta_inventory,
        )
        bars.append(
            TornadoBar(
                channel=channel,
                low_value=low,
                high_value=high,
                revenue_at_low_usd=rev_low,
                revenue_at_high_usd=rev_high,
                baseline_usd=baseline,
                fitted=channel == "beta_price",
            )
        )

    bars.sort(key=lambda b: b.swing_usd, reverse=True)
    caveats = [
        "Tornado bars use discounted expected revenue of the fixed plan — no "
        "sale lottery. The Monte Carlo distribution is wider because sales are lumpy.",
    ]
    assumptions = [b.channel for b in bars if not b.fitted]
    if assumptions:
        caveats.append(
            f"Only beta_price's ±1σ is fitted. {assumptions} use user-supplied "
            "standard deviations; their bars are only as good as those assumptions."
        )
    return TornadoReport(
        baseline_usd=baseline,
        bars=bars,
        provenance=dict(result.provenance),
        caveats=caveats,
    )


def shadow_prices(
    tensor: RevenueTensor,
    units: pd.DataFrame,
    *,
    constraints: ConstraintSet | None = None,
    integer_objective_usd: float | None = None,
    provenance: dict[str, Any] | None = None,
    time_limit_seconds: int = 60,
) -> ShadowPriceReport:
    """LP-relaxed duals on the cash-flow floors.

    Args:
        tensor / units: the same formulation the integer plan was solved against.
        constraints: active families. Defaults to all.
        integer_objective_usd: the MIP objective, for the relaxation gap.
        provenance: carried into the report for the API contract.
    """
    import time

    started = time.perf_counter()
    rules = constraints or ConstraintSet()
    program = build_program(tensor, units, rules, relax=True)
    program.problem.solve(_solver(time_limit_seconds))
    elapsed = time.perf_counter() - started

    if program.problem.status != pulp.LpStatusOptimal:
        raise SchemaError(
            f"LP relaxation returned {pulp.LpStatus[program.problem.status]}. "
            "Shadow prices are undefined when the continuous problem is not optimal."
        )

    objective = float(pulp.value(program.problem.objective) or 0.0)
    cash: list[ShadowPrice] = []
    for j, phase in enumerate(tensor.phases):
        floor = float(phase.cash_flow_floor_usd or 0.0)
        if floor <= 0:
            continue
        name = f"{CASH_FLOW_FLOOR}_{j}"
        constraint = program.problem.get_constraint_by_name(name)
        if constraint is None:
            continue
        dual = float(constraint.pi)
        # Max problem, ≥ constraint: binding duals are ≤ 0. Negate so a binding
        # covenant reports a positive "cost per dollar of floor".
        cost = -dual
        cash.append(
            ShadowPrice(
                phase_index=j,
                phase_name=phase.name,
                cash_flow_floor_usd=floor,
                dual=dual,
                objective_cost_per_floor_dollar=cost,
                binds=abs(dual) > 1e-6,
            )
        )

    gap = None if integer_objective_usd is None else objective - integer_objective_usd
    caveats = [
        "Shadow prices come from the LP relaxation (continuous [0,1] variables), "
        "not the integer plan. They bound the marginal cost of the covenant; the "
        "MIP's true marginal cost can be smaller when the integer solution is "
        "integral on the binding cells.",
    ]
    if gap is not None and gap > 1e-3 * max(abs(objective), 1.0):
        caveats.append(
            f"Relaxation gap is ${gap:,.0f} "
            f"({gap / max(abs(integer_objective_usd or 1.0), 1.0):.1%} of the "
            "integer objective). Duals are less informative when the gap is large."
        )
    if not cash:
        caveats.append(
            "No cash-flow floors were active, so there are no covenant duals to report."
        )
    elif not any(s.binds for s in cash):
        caveats.append(
            "No cash-flow floor binds in the LP relaxation — the covenant is not "
            "costing revenue at the margin under these floors."
        )

    logger.info(
        "LP relaxation solved in %.2fs: objective $%.0f, %d cash-flow duals",
        elapsed, objective, len(cash),
    )
    return ShadowPriceReport(
        objective_usd=objective,
        integer_objective_usd=integer_objective_usd,
        relaxation_gap_usd=gap,
        cash_flow=cash,
        provenance=dict(provenance or {}),
        caveats=caveats,
        seconds=elapsed,
    )


def format_tornado(report: TornadoReport) -> str:
    """Human-readable tornado report."""
    lines = [
        "SENSITIVITY TORNADO  (±1σ, fixed plan, expected revenue)",
        f"  baseline discounted expected revenue  ${report.baseline_usd:,.0f}",
        "",
        f"  {'channel':<28} {'−1σ Δ':>14} {'+1σ Δ':>14} {'swing':>14}",
    ]
    for bar in report.bars:
        tag = "fitted" if bar.fitted else "assumed"
        lines.append(
            f"  {bar.channel + ' [' + tag + ']':<28} "
            f"${bar.low_delta_usd:>13,.0f} "
            f"${bar.high_delta_usd:>13,.0f} "
            f"${bar.swing_usd:>13,.0f}"
        )
    lines.append("")
    lines.append("CAVEATS")
    for caveat in [*report.provenance.get("warnings", []), *report.caveats]:
        lines.append(f"  !  {caveat}")
    return "\n".join(lines)


def format_shadow_prices(report: ShadowPriceReport) -> str:
    """Human-readable shadow-price report."""
    lines = [
        f"CASH-FLOW SHADOW PRICES  (LP relaxation in {report.seconds:.2f}s)",
        f"  LP objective     ${report.objective_usd:,.0f}",
    ]
    if report.integer_objective_usd is not None:
        lines.append(f"  integer objective ${report.integer_objective_usd:,.0f}")
    if report.relaxation_gap_usd is not None:
        lines.append(f"  relaxation gap   ${report.relaxation_gap_usd:,.0f}")
    lines.append("")
    if report.cash_flow:
        lines.append(
            f"  {'phase':<14} {'floor':>14} {'cost / $ floor':>16} {'binds':>8}"
        )
        for row in report.cash_flow:
            lines.append(
                f"  {row.phase_name:<14} ${row.cash_flow_floor_usd:>13,.0f} "
                f"${row.objective_cost_per_floor_dollar:>15,.4f} "
                f"{'yes' if row.binds else 'no':>8}"
            )
    else:
        lines.append("  (no cash-flow floors)")
    lines.append("")
    lines.append("CAVEATS")
    for caveat in [*report.provenance.get("warnings", []), *report.caveats]:
        lines.append(f"  !  {caveat}")
    return "\n".join(lines)
