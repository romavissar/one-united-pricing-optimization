"""Re-evaluating a fixed plan under uncertainty, and asking whether it holds up.

**Why this is fast.** Under proportional hazards, changing the linear predictor
by `Δη` maps the fitted survival probability straight to the perturbed one:

    S_new = S_hat ** exp(Δη)      so      P_new = 1 - (1 - P_hat) ** exp(Δη)

`P_hat` and the implied `rel_price_premium` are already on the `RevenueTensor`
from Phase 4, so a draw costs a few array operations and no refit. Ten thousand
draws is a couple of seconds.

**Two layers of randomness, and the difference matters.** A draw first samples
parameters — how elastic is demand, how soft is the market — and then samples
*whether each unit actually sells*. Parameter uncertainty alone would badly
understate the spread across sixty units, and it would answer the wrong
question about a loan covenant: a covenant is breached by realized cash, not by
expected cash. The report separates the two, because "we do not know the
elasticity" and "sales are lumpy" call for different responses. The first is
fixed by better data, the second by a bigger buffer.

**What the baseline claims.** `P(plan > baseline)` compares against pricing
every unit at its comps value on the *same* release schedule, so the number
isolates the pricing decision — which is the decision `beta_price` exists to
inform. It is not a claim about the value of phasing. Both arms share the same
random draws, so the comparison is paired.

Units: revenue USD; `absorption` log points; delays in months.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import SchemaError
from src.optimizer.constraints import ConstraintSet
from src.optimizer.discretize import PriceLadder
from src.optimizer.formulate import RevenueTensor
from src.optimizer.solve import OptimizeResult, PlanRow, SolveStatus, solve_release_plan
from src.simulation.scenarios import LHS, ScenarioDraws, ScenarioSpec, draw_scenarios

logger = logging.getLogger(__name__)

_DAYS_PER_MONTH = 30.4375
_INVENTORY_TERM = "inventory_competition"
# Above this simulated breach rate the cash-flow floor is not really being met,
# whatever the expected-value constraint says. PROJECT_BRIEF §5.
_BREACH_ALARM = 0.05
_CVAR_TAIL = 0.05


@dataclass(frozen=True)
class DistributionSummary:
    """Percentiles and tail risk for one revenue series, in USD."""

    mean: float
    sd: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    cvar5: float
    minimum: float
    maximum: float

    @property
    def is_ordered(self) -> bool:
        return self.p5 <= self.p25 <= self.p50 <= self.p75 <= self.p95

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(values: np.ndarray) -> DistributionSummary:
    """Summarise a revenue sample. CVaR@5% is the mean of the worst 5%."""
    sample = np.asarray(values, dtype="float64")
    if sample.size == 0:
        raise SchemaError("cannot summarise an empty revenue sample")
    cutoff = float(np.quantile(sample, _CVAR_TAIL))
    tail = sample[sample <= cutoff]
    return DistributionSummary(
        mean=float(sample.mean()),
        sd=float(sample.std(ddof=1)) if sample.size > 1 else 0.0,
        p5=cutoff,
        p25=float(np.quantile(sample, 0.25)),
        p50=float(np.quantile(sample, 0.50)),
        p75=float(np.quantile(sample, 0.75)),
        p95=float(np.quantile(sample, 0.95)),
        cvar5=float(tail.mean()) if tail.size else cutoff,
        minimum=float(sample.min()),
        maximum=float(sample.max()),
    )


@dataclass(frozen=True)
class PhaseBreach:
    """How often one phase misses its cash-flow floor once sales are realized."""

    index: int
    name: str
    cash_flow_floor_usd: float
    expected_revenue_usd: float
    p5_revenue_usd: float
    breach_probability: float
    buffered_floor_usd: float | None

    @property
    def is_alarming(self) -> bool:
        return self.breach_probability > _BREACH_ALARM

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "is_alarming": self.is_alarming}


@dataclass
class RevenueDistribution:
    """The distribution of what a fixed plan is worth, and what threatens it."""

    n_draws: int
    discounted_usd: DistributionSummary
    nominal_usd: DistributionSummary
    parameter_only_discounted_usd: DistributionSummary
    phase_breach: list[PhaseBreach]
    scenario: dict[str, Any]
    provenance: dict[str, Any]
    caveats: list[str] = field(default_factory=list)
    baseline_discounted_usd: DistributionSummary | None = None
    uplift_vs_baseline_usd: DistributionSummary | None = None
    prob_beats_baseline: float | None = None
    seconds: float = 0.0

    @property
    def variance_decomposition(self) -> dict[str, float]:
        """Split total revenue variance into parameter and idiosyncratic parts.

        Parameter variance is measured with sales held at their expectation.
        The remainder is the lumpiness of sixty individual sales either
        happening or not. They are reduced by different things: better data
        versus a bigger buffer.
        """
        total = self.discounted_usd.sd ** 2
        parameter = self.parameter_only_discounted_usd.sd ** 2
        if total <= 0:
            return {"parameter_share": 0.0, "idiosyncratic_share": 0.0, "total_sd_usd": 0.0}
        return {
            "parameter_share": min(1.0, parameter / total),
            "idiosyncratic_share": max(0.0, 1.0 - parameter / total),
            "total_sd_usd": self.discounted_usd.sd,
        }

    @property
    def alarming_phases(self) -> list[PhaseBreach]:
        return [p for p in self.phase_breach if p.is_alarming]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_draws": self.n_draws,
            "discounted_usd": self.discounted_usd.as_dict(),
            "nominal_usd": self.nominal_usd.as_dict(),
            "parameter_only_discounted_usd": self.parameter_only_discounted_usd.as_dict(),
            "variance_decomposition": self.variance_decomposition,
            "baseline_discounted_usd": (
                self.baseline_discounted_usd.as_dict() if self.baseline_discounted_usd else None
            ),
            "uplift_vs_baseline_usd": (
                self.uplift_vs_baseline_usd.as_dict()
                if self.uplift_vs_baseline_usd else None
            ),
            "prob_beats_baseline": self.prob_beats_baseline,
            "phase_breach": [p.as_dict() for p in self.phase_breach],
            "scenario": self.scenario,
            "provenance": self.provenance,
            "caveats": self.caveats,
            "seconds": self.seconds,
        }


@dataclass
class _PlanArrays:
    """Per-released-unit constants, pulled out of the tensor once."""

    unit_index: np.ndarray
    phase_index: np.ndarray
    level_index: np.ndarray
    price_ppsf: np.ndarray
    area_sqft: np.ndarray
    probability: np.ndarray
    rel_price_premium: np.ndarray
    discount_factor: np.ndarray
    gate_slack_months: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.price_ppsf.shape[0])


def _gate_slack_months(
    units: pd.DataFrame, tensor: RevenueTensor, rows: list[PlanRow], presale_lead_months: float,
    phase_dates: list[pd.Timestamp],
) -> np.ndarray:
    """Months of construction slippage each released unit can absorb.

    A delay pushes completion later, so it pushes the presale window later too.
    A unit released with two months of slack survives a one-month delay and is
    unsellable under a three-month one.
    """
    completion = pd.to_datetime(
        units.set_index(units["unit_id"].astype(str))["completion_date"], errors="coerce"
    )
    lead = pd.Timedelta(presale_lead_months * _DAYS_PER_MONTH, unit="D")
    slack = np.empty(len(rows), dtype="float64")
    for position, row in enumerate(rows):
        sellable_from = completion.loc[row.unit_id] - lead
        delta = phase_dates[row.phase_index] - sellable_from
        slack[position] = delta.total_seconds() / (86400.0 * _DAYS_PER_MONTH)
    return slack


def _plan_arrays(
    result: OptimizeResult, tensor: RevenueTensor, units: pd.DataFrame,
    presale_lead_months: float, phase_dates: list[pd.Timestamp],
    level_index: np.ndarray | None = None,
) -> _PlanArrays:
    rows = list(result.plan)
    unit_index = np.array([tensor.unit_ids.index(r.unit_id) for r in rows], dtype="int64")
    phase_index = np.array([r.phase_index for r in rows], dtype="int64")
    levels = (
        np.array([r.level_index for r in rows], dtype="int64")
        if level_index is None
        else np.asarray(level_index, dtype="int64")
    )
    return _PlanArrays(
        unit_index=unit_index,
        phase_index=phase_index,
        level_index=levels,
        price_ppsf=tensor.price_ppsf[unit_index, levels],
        area_sqft=tensor.area_sqft[unit_index],
        probability=tensor.probability[unit_index, phase_index, levels],
        rel_price_premium=tensor.rel_price_premium[unit_index, phase_index, levels],
        discount_factor=tensor.discount_factor[phase_index],
        gate_slack_months=_gate_slack_months(
            units, tensor, rows, presale_lead_months, phase_dates
        ),
    )


def perturbed_probability(
    arrays: _PlanArrays,
    draws: ScenarioDraws,
    *,
    beta_price_hat: float,
    beta_inventory: float,
) -> np.ndarray:
    """(draws, rows) sale probabilities under each scenario.

    The shift in the linear predictor is

        Δη = β_draw·rel_new − β̂·rel_old + β_inv·Δinventory + absorption

    where `rel_new` accounts for the comp median moving under `comps_drift`: a
    fixed asking price is a smaller premium when comps rise. Everything else in
    the predictor is unchanged, so it cancels, which is what makes the
    closed-form exact rather than approximate.
    """
    rel = arrays.rel_price_premium[None, :]
    drift = draws.comps_drift[:, None]
    rel_new = (1.0 + rel) / (1.0 + drift) - 1.0

    delta_eta = (
        draws.beta_price[:, None] * rel_new
        - beta_price_hat * rel
        + beta_inventory * draws.competing_listings[:, None]
        + draws.absorption[:, None]
    )
    survival = np.clip(1.0 - arrays.probability[None, :], 0.0, 1.0)
    return 1.0 - survival ** np.exp(delta_eta)


def _alive_under_delay(arrays: _PlanArrays, draws: ScenarioDraws) -> np.ndarray:
    """(draws, rows) mask: False where slippage pushes a unit behind its gate."""
    return draws.completion_delay_months[:, None] <= arrays.gate_slack_months[None, :]


def simulate_plan(
    result: OptimizeResult,
    tensor: RevenueTensor,
    units: pd.DataFrame,
    spec: ScenarioSpec,
    *,
    n_draws: int = 10_000,
    seed: int = 0,
    method: str = "normal",
    presale_lead_months: float = 24.0,
    phase_dates: list[pd.Timestamp] | None = None,
    hazard_coefficients: dict[str, float] | None = None,
    ladder: PriceLadder | None = None,
) -> RevenueDistribution:
    """Re-evaluate a fixed plan across sampled scenarios.

    Args:
        result: a solved, optimal plan.
        tensor: the tensor it was solved against.
        units: validated inventory, for completion dates.
        spec: the uncertainty to sample. `beta_price_mean` must equal the `β̂`
            baked into the tensor.
        n_draws: scenarios. 10,000 finishes in seconds.
        seed: reproducibility. A distribution that moves between runs is not
            something a developer can act on.
        presale_lead_months / phase_dates: needed to age the construction gate
            under a delay. Take them from the `ProjectSpec` the plan was built
            with.
        hazard_coefficients: fitted coefficients by name. Only
            `inventory_competition` is read, and only when that channel is
            active.
        ladder: enables the price-at-comps baseline.

    Raises:
        SchemaError: when the plan is not solved, when the competing-listings
            channel is active but its fitted coefficient was not supplied, or
            when `beta_price_mean` disagrees with the plan's provenance.
    """
    import time

    started = time.perf_counter()
    if result.status is not SolveStatus.OPTIMAL:
        raise SchemaError(
            f"Cannot simulate a plan with status {result.status.value}. Simulating an "
            "infeasible plan produces a distribution of a thing that does not exist."
        )
    if not result.plan:
        raise SchemaError("Cannot simulate an empty plan")

    fitted_beta = result.provenance.get("beta_price")
    if fitted_beta is not None and not np.isclose(fitted_beta, spec.beta_price_mean, atol=1e-9):
        raise SchemaError(
            f"spec.beta_price_mean is {spec.beta_price_mean:+.6f} but the plan was built "
            f"at {fitted_beta:+.6f}. The perturbation is measured from the fitted value, "
            "so a mismatch silently shifts the whole distribution."
        )

    coefficients = hazard_coefficients or {}
    beta_inventory = float(coefficients.get(_INVENTORY_TERM, 0.0))
    if spec.competing_listings_sd > 0 and _INVENTORY_TERM not in coefficients:
        raise SchemaError(
            "The competing_listings channel is active but no fitted "
            f"{_INVENTORY_TERM!r} coefficient was supplied, so the shock has no way to "
            "reach the hazard. Pass hazard_coefficients from the fit, or set "
            "competing_listings_sd to zero."
        )

    dates = phase_dates or [pd.Timestamp("1970-01-01")] * len(tensor.phases)
    arrays = _plan_arrays(result, tensor, units, presale_lead_months, dates)
    draws = draw_scenarios(spec, n_draws, seed=seed, method=method)

    probability = perturbed_probability(
        arrays, draws, beta_price_hat=spec.beta_price_mean, beta_inventory=beta_inventory
    )
    alive = _alive_under_delay(arrays, draws)
    effective = probability * alive

    gross = arrays.price_ppsf * arrays.area_sqft
    discounted_gross = gross * arrays.discount_factor

    rng = np.random.default_rng(seed + 1)
    uniforms = rng.random(effective.shape)
    sold = uniforms < effective

    discounted = (sold * discounted_gross[None, :]).sum(axis=1)
    nominal = (sold * gross[None, :]).sum(axis=1)
    parameter_only = (effective * discounted_gross[None, :]).sum(axis=1)

    breaches = _phase_breaches(result, arrays, sold, gross)
    baseline_summary, uplift_summary, prob_beats = _baseline(
        result, tensor, arrays, draws, units, ladder, uniforms,
        beta_price_hat=spec.beta_price_mean, beta_inventory=beta_inventory,
        presale_lead_months=presale_lead_months, phase_dates=dates,
        discounted=discounted,
    )

    caveats = _simulation_caveats(spec, draws, breaches, ladder)
    elapsed = time.perf_counter() - started
    logger.info(
        "Simulated %d draws in %.2fs: P50=$%.0f P5=$%.0f",
        n_draws, elapsed, float(np.median(discounted)), float(np.quantile(discounted, 0.05)),
    )
    return RevenueDistribution(
        n_draws=n_draws,
        discounted_usd=summarize(discounted),
        nominal_usd=summarize(nominal),
        parameter_only_discounted_usd=summarize(parameter_only),
        phase_breach=breaches,
        scenario=draws.as_dict(),
        provenance=dict(result.provenance),
        caveats=caveats,
        baseline_discounted_usd=baseline_summary,
        uplift_vs_baseline_usd=uplift_summary,
        prob_beats_baseline=prob_beats,
        seconds=elapsed,
    )


def _phase_breaches(
    result: OptimizeResult, arrays: _PlanArrays, sold: np.ndarray, gross: np.ndarray
) -> list[PhaseBreach]:
    """Simulated breach probability per phase, on realized nominal cash."""
    out: list[PhaseBreach] = []
    for phase in result.per_phase:
        floor = float(phase.cash_flow_floor_usd or 0.0)
        members = arrays.phase_index == phase.index
        realized = (sold[:, members] * gross[None, members]).sum(axis=1)
        breach = float((realized < floor).mean()) if floor > 0 else 0.0
        p5 = float(np.quantile(realized, 0.05))
        # Scaling the floor by how far the 5th percentile falls short moves that
        # percentile to roughly the true floor once the plan is re-solved. An
        # approximation to a chance constraint, not a substitute for one.
        buffered = float(floor * floor / p5) if floor > 0 and p5 > 0 and p5 < floor else None
        out.append(
            PhaseBreach(
                index=phase.index,
                name=phase.name,
                cash_flow_floor_usd=floor,
                expected_revenue_usd=float(realized.mean()),
                p5_revenue_usd=p5,
                breach_probability=breach,
                buffered_floor_usd=buffered,
            )
        )
    return out


def price_at_comps_levels(ladder: PriceLadder, unit_ids: tuple[str, ...]) -> np.ndarray:
    """Ladder index closest to each unit's fitted comps value.

    The "do nothing clever" price: what the hedonic surface says the unit is
    worth, with no elasticity reasoning applied.
    """
    out = np.empty(len(unit_ids), dtype="int64")
    for position, unit_id in enumerate(unit_ids):
        row = ladder.index_of(unit_id)
        out[position] = int(
            np.argmin(np.abs(ladder.levels_ppsf[row] - ladder.predicted_ppsf[row]))
        )
    return out


def _baseline(
    result: OptimizeResult,
    tensor: RevenueTensor,
    arrays: _PlanArrays,
    draws: ScenarioDraws,
    units: pd.DataFrame,
    ladder: PriceLadder | None,
    uniforms: np.ndarray,
    *,
    beta_price_hat: float,
    beta_inventory: float,
    presale_lead_months: float,
    phase_dates: list[pd.Timestamp],
    discounted: np.ndarray,
) -> tuple[DistributionSummary | None, DistributionSummary | None, float | None]:
    """Same units, same phases, priced at comps. Paired on the same draws.

    Returns the baseline's own distribution, the distribution of the **paired**
    per-draw uplift, and P(plan > baseline). The paired uplift is the honest
    statistic: because both arms share the sale lottery, `median(plan - base)`
    is the typical gain from repricing, whereas `median(plan) - median(base)` is
    a difference of two separately-ranked samples and carries no interval at all.
    """
    if ladder is None:
        return None, None, None

    levels = price_at_comps_levels(ladder, tuple(r.unit_id for r in result.plan))
    base = _plan_arrays(
        result, tensor, units, presale_lead_months, phase_dates, level_index=levels
    )
    probability = perturbed_probability(
        base, draws, beta_price_hat=beta_price_hat, beta_inventory=beta_inventory
    )
    effective = probability * _alive_under_delay(base, draws)
    # Common random numbers: the same uniforms decide who buys in both arms, so
    # the comparison measures the pricing difference rather than sampling noise.
    sold = uniforms < effective
    revenue = (sold * (base.price_ppsf * base.area_sqft * base.discount_factor)[None, :]).sum(axis=1)
    return (
        summarize(revenue),
        summarize(discounted - revenue),
        float((discounted > revenue).mean()),
    )


def _simulation_caveats(
    spec: ScenarioSpec, draws: ScenarioDraws, breaches: list[PhaseBreach],
    ladder: PriceLadder | None,
) -> list[str]:
    caveats = list(draws.notes)
    assumptions = spec.as_dict()["assumption_channels"]
    if assumptions:
        caveats.append(
            f"Only beta_price's dispersion is fitted. {assumptions} are user "
            "assumptions about the future; nothing in this system estimates them, and "
            "the width of the distribution is only as good as they are."
        )
    for breach in breaches:
        if breach.is_alarming:
            buffered = (
                f" Raising the floor to ${breach.buffered_floor_usd:,.0f} and re-solving "
                "would approximate a 95% chance constraint."
                if breach.buffered_floor_usd
                else ""
            )
            caveats.append(
                f"{breach.name} misses its ${breach.cash_flow_floor_usd:,.0f} cash-flow "
                f"floor in {breach.breach_probability:.1%} of draws. The optimizer's "
                "constraint binds on expected revenue, so it was satisfied in "
                f"expectation and is still breached in practice.{buffered}"
            )
    if ladder is None:
        caveats.append(
            "No price ladder supplied, so the plan was not compared against a "
            "price-at-comps baseline. Without it there is no evidence the optimization "
            "beat doing nothing."
        )
    caveats.append(
        "Sale outcomes are drawn independently across units, matching the optimizer's "
        "independence assumption. Real buyers choosing between two similar units make "
        "these outcomes negatively correlated, which this understates."
    )
    caveats.append(
        "Parameter uncertainty is propagated for beta_price only. Every other fitted "
        "coefficient — floor, area, beds, HOA, submarket, season, competing inventory — "
        "is held at its point estimate. Whether that makes the reported band too narrow "
        "or too wide is NOT established: omitting covariance terms can move an interval "
        "in either direction depending on the signs of the coefficients and of their "
        "correlations, so the honest statement is that the direction of the bias is "
        "unknown, not that the band is a lower bound. What *is* measured is the "
        "generated-regressor effect: beta_price's dispersion is taken from a two-stage "
        "bootstrap that refits the hedonic inside each replication, which on the current "
        "sample is 21% wider than the standard error the Cox model reports. The full "
        "covariance is saved with the bundle as `cox_coefficient_covariance`."
    )
    return caveats


# --- optional: does the plan itself survive, or just its revenue? ---------


@dataclass(frozen=True)
class ScenarioPlan:
    """One re-solved scenario, compared against the base plan."""

    draw_index: int
    beta_price: float
    status: str
    objective_usd: float
    base_plan_objective_usd: float
    units_released: int
    same_phase_share: float
    same_level_share: float
    mean_absolute_level_shift: float

    @property
    def regret_usd(self) -> float:
        """Revenue lost by keeping the base plan instead of re-optimizing."""
        return self.objective_usd - self.base_plan_objective_usd

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "regret_usd": self.regret_usd}


@dataclass
class PlanStability:
    """Whether the recommended plan is a property of the data or of one β."""

    scenarios: list[ScenarioPlan]
    n_solved: int
    mean_same_phase_share: float
    mean_same_level_share: float
    mean_regret_usd: float
    max_regret_usd: float
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_scenarios": len(self.scenarios),
            "n_solved": self.n_solved,
            "mean_same_phase_share": self.mean_same_phase_share,
            "mean_same_level_share": self.mean_same_level_share,
            "mean_regret_usd": self.mean_regret_usd,
            "max_regret_usd": self.max_regret_usd,
            "scenarios": [s.as_dict() for s in self.scenarios],
            "caveats": self.caveats,
        }


def apply_scenario(
    tensor: RevenueTensor,
    draws: ScenarioDraws,
    index: int,
    *,
    beta_price_hat: float,
    beta_inventory: float = 0.0,
    units: pd.DataFrame | None = None,
    presale_lead_months: float = 24.0,
    phase_dates: list[pd.Timestamp] | None = None,
) -> RevenueTensor:
    """A copy of the tensor with one scenario's shocks folded into `R[i,j,k]`.

    Uses the same closed-form hazard shift as `perturbed_probability`, applied
    across the whole tensor rather than only the released cells, so the
    optimizer can choose differently under the scenario. When `units` and
    `phase_dates` are supplied, a construction-delay draw also tightens the
    releasable mask — otherwise the delay channel is ignored in re-solve mode.
    """
    rel = tensor.rel_price_premium
    rel_new = (1.0 + rel) / (1.0 + float(draws.comps_drift[index])) - 1.0
    delta_eta = (
        float(draws.beta_price[index]) * rel_new
        - beta_price_hat * rel
        + beta_inventory * float(draws.competing_listings[index])
        + float(draws.absorption[index])
    )
    survival = np.clip(1.0 - tensor.probability, 0.0, 1.0)
    probability = 1.0 - survival ** np.exp(delta_eta)

    releasable = tensor.releasable.copy()
    delay = float(draws.completion_delay_months[index])
    if delay > 0 and units is not None and phase_dates is not None:
        # Slip completion by `delay` months: the presale window moves with it.
        completion = pd.to_datetime(units["completion_date"], errors="coerce")
        lead = pd.Timedelta(presale_lead_months * _DAYS_PER_MONTH, unit="D")
        slip = pd.Timedelta(delay * _DAYS_PER_MONTH, unit="D")
        sellable_from = completion - lead + slip
        for j, date in enumerate(phase_dates):
            releasable[:, j] = (
                releasable[:, j]
                & sellable_from.notna().to_numpy()
                & (date >= sellable_from).to_numpy()
            )
        probability = np.where(releasable[:, :, None], probability, 0.0)

    expected = tensor.price_ppsf[:, None, :] * tensor.area_sqft[:, None, None] * probability
    discounted = expected * tensor.discount_factor[None, :, None]
    return RevenueTensor(
        unit_ids=tensor.unit_ids,
        phases=tensor.phases,
        price_ppsf=tensor.price_ppsf,
        area_sqft=tensor.area_sqft,
        probability=probability,
        expected_revenue_usd=expected,
        discounted_usd=discounted,
        releasable=releasable,
        discount_factor=tensor.discount_factor,
        rel_price_premium=rel,
        premium_support=tensor.premium_support,
        excluded=tensor.excluded,
        notes=[*tensor.notes, f"perturbed by scenario draw {index}"],
    )


def resolve_scenarios(
    result: OptimizeResult,
    tensor: RevenueTensor,
    units: pd.DataFrame,
    spec: ScenarioSpec,
    provenance: Any,
    *,
    n_scenarios: int = 100,
    seed: int = 0,
    constraints: ConstraintSet | None = None,
    hazard_coefficients: dict[str, float] | None = None,
    presale_lead_months: float = 24.0,
    phase_dates: list[pd.Timestamp] | None = None,
) -> PlanStability:
    """Re-solve the MILP under LHS-sampled scenarios and compare the plans.

    Slow by construction — one CBC solve per scenario. It answers a question the
    fixed-plan mode cannot: whether the recommendation is a property of the
    market or an artefact of one point estimate of `beta_price`. A plan that
    reshuffles completely at `β̂ ± 1σ` should not be presented as *the* plan.

    `regret_usd` is the number to read: how much revenue keeping the base plan
    costs in each scenario, relative to having optimized for it.
    """
    beta_inventory = float((hazard_coefficients or {}).get(_INVENTORY_TERM, 0.0))
    draws = draw_scenarios(spec, n_scenarios, seed=seed, method=LHS)
    base = {row.unit_id: (row.phase_index, row.level_index) for row in result.plan}
    dates = phase_dates or [pd.Timestamp("1970-01-01")] * len(tensor.phases)

    scenarios: list[ScenarioPlan] = []
    for index in range(n_scenarios):
        shocked = apply_scenario(
            tensor,
            draws,
            index,
            beta_price_hat=spec.beta_price_mean,
            beta_inventory=beta_inventory,
            units=units,
            presale_lead_months=presale_lead_months,
            phase_dates=dates,
        )
        solved = solve_release_plan(
            shocked, units, provenance, constraints=constraints, diagnose=False
        )
        if solved.status is not SolveStatus.OPTIMAL:
            logger.warning("Scenario %d returned %s", index, solved.status.value)
            continue

        base_value = sum(
            float(shocked.discounted_usd[shocked.unit_ids.index(u), phase, level])
            for u, (phase, level) in base.items()
            if u in shocked.unit_ids
        )
        matched = [(u, p, k) for u, p, k in
                   ((r.unit_id, r.phase_index, r.level_index) for r in solved.plan)
                   if u in base]
        same_phase = [1.0 if base[u][0] == p else 0.0 for u, p, _ in matched]
        same_level = [1.0 if base[u][1] == k else 0.0 for u, _, k in matched]
        shifts = [abs(base[u][1] - k) for u, _, k in matched]
        scenarios.append(
            ScenarioPlan(
                draw_index=index,
                beta_price=float(draws.beta_price[index]),
                status=solved.status.value,
                objective_usd=solved.objective_usd,
                base_plan_objective_usd=base_value,
                units_released=solved.units_released,
                same_phase_share=float(np.mean(same_phase)) if same_phase else 0.0,
                same_level_share=float(np.mean(same_level)) if same_level else 0.0,
                mean_absolute_level_shift=float(np.mean(shifts)) if shifts else 0.0,
            )
        )

    if not scenarios:
        raise SchemaError(
            f"No scenario out of {n_scenarios} solved to optimal. The constraints are "
            "infeasible across the sampled range, not merely at the point estimate."
        )

    regrets = [s.regret_usd for s in scenarios]
    caveats = [
        "Re-solve mode perturbs the fitted hazard in closed form rather than refitting "
        "the demand model. It answers 'would a different beta change the plan', not "
        "'would different data change the model'.",
    ]
    worst = max(scenarios, key=lambda s: s.regret_usd)
    if worst.regret_usd > 0.02 * max(worst.objective_usd, 1.0):
        caveats.append(
            f"In the worst sampled scenario (beta_price={worst.beta_price:+.3f}) keeping "
            f"the base plan costs ${worst.regret_usd:,.0f}, "
            f"{worst.regret_usd / max(worst.objective_usd, 1.0):.1%} of that scenario's "
            "optimum. The plan is sensitive to the elasticity estimate."
        )
    return PlanStability(
        scenarios=scenarios,
        n_solved=len(scenarios),
        mean_same_phase_share=float(np.mean([s.same_phase_share for s in scenarios])),
        mean_same_level_share=float(np.mean([s.same_level_share for s in scenarios])),
        mean_regret_usd=float(np.mean(regrets)),
        max_regret_usd=float(np.max(regrets)),
        caveats=caveats,
    )


def format_distribution(distribution: RevenueDistribution) -> str:
    """Human-readable Monte Carlo report."""
    d = distribution.discounted_usd
    lines = [
        f"REVENUE DISTRIBUTION  {distribution.n_draws:,} draws in "
        f"{distribution.seconds:.2f}s",
        "",
        "DISCOUNTED EXPECTED REVENUE (USD, realized sales)",
        f"  P5   ${d.p5:>15,.0f}      CVaR@5%  ${d.cvar5:>15,.0f}",
        f"  P25  ${d.p25:>15,.0f}",
        f"  P50  ${d.p50:>15,.0f}      mean     ${d.mean:>15,.0f}",
        f"  P75  ${d.p75:>15,.0f}",
        f"  P95  ${d.p95:>15,.0f}      sd       ${d.sd:>15,.0f}",
    ]
    decomposition = distribution.variance_decomposition
    lines.append("")
    lines.append("WHERE THE RISK COMES FROM")
    lines.append(
        f"  parameter uncertainty: {decomposition['parameter_share']:.0%}   "
        f"lumpy sales: {decomposition['idiosyncratic_share']:.0%}"
    )
    if distribution.baseline_discounted_usd:
        base = distribution.baseline_discounted_usd
        lines.append("")
        lines.append("VERSUS PRICING AT COMPS (same release schedule)")
        lines.append(f"  baseline P50 ${base.p50:,.0f}   plan P50 ${d.p50:,.0f}")
        lines.append(f"  P(plan beats baseline) = {distribution.prob_beats_baseline:.1%}")
        up = distribution.uplift_vs_baseline_usd
        if up:
            lines.append(
                f"  paired uplift  P50 ${up.p50:,.0f}   90% band "
                f"${up.p5:,.0f} to ${up.p95:,.0f}"
            )
    if any(p.cash_flow_floor_usd for p in distribution.phase_breach):
        lines.append("")
        lines.append("CASH-FLOW FLOORS (realized, nominal)")
        for phase in distribution.phase_breach:
            if not phase.cash_flow_floor_usd:
                continue
            mark = "BREACH" if phase.is_alarming else "ok    "
            lines.append(
                f"  [{mark}] {phase.name:<14} floor ${phase.cash_flow_floor_usd:>12,.0f}  "
                f"P5 ${phase.p5_revenue_usd:>12,.0f}  "
                f"P(breach)={phase.breach_probability:.1%}"
            )
    lines.append("")
    lines.append("CAVEATS")
    for caveat in [*distribution.provenance.get("warnings", []), *distribution.caveats]:
        lines.append(f"  !  {caveat}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Solve a plan on synthetic demand, then print its revenue distribution.

    Exit codes: 0 ok, 1 failure. Never reads the real MLS export.
    """
    import argparse
    from pathlib import Path

    from src.config import load_market_config
    from src.data.features import build_features
    from src.data.normalize import normalize_mls
    from src.data.synth import generate_synthetic_mls
    from src.demand.hedonic import fit_hedonic
    from src.demand.survival import (
        CONTROLLED_CATEGORICALS,
        CoxDemandModel,
        available_covariates,
    )
    from src.optimizer.discretize import build_price_ladder
    from src.optimizer.formulate import (
        Comps,
        DemandProvenance,
        Phase,
        ProjectSpec,
        build_revenue_tensor,
    )
    from src.optimizer.solve import SolveStatus, format_plan, solve_release_plan
    from src.simulation.scenarios import ScenarioSpec
    from src.simulation.sensitivity import (
        format_shadow_prices,
        format_tornado,
        shadow_prices,
        tornado,
    )
    from src.utils.validate import format_validation, validate_inventory

    backend_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Monte Carlo revenue distribution for a synthetic-demand plan"
    )
    parser.add_argument("--inspect", action="store_true", help="print the report")
    parser.add_argument("--market", default="miami")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=backend_root / "data" / "project_inputs" / "example_inventory.csv",
    )
    parser.add_argument("--n-listings", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--phase-months", type=float, nargs="+", default=[0.0, 6.0, 12.0, 18.0])
    parser.add_argument("--max-units", type=int, default=20)
    parser.add_argument("--competing-listings", type=float, default=30.0)
    parser.add_argument("--draws", type=int, default=None, help="override monte_carlo_draws")
    parser.add_argument("--absorption-sd", type=float, default=0.15)
    parser.add_argument("--comps-drift-sd", type=float, default=0.03)
    parser.add_argument("--completion-delay-sd", type=float, default=1.0)
    parser.add_argument("--tornado", action="store_true", help="also print the tornado")
    parser.add_argument("--shadow-prices", action="store_true", help="also print LP duals")
    parser.add_argument(
        "--resolve",
        type=int,
        default=0,
        metavar="N",
        help="re-solve under N LHS scenarios (slow; plan stability)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_market_config(args.market)
    defaults = config.get("defaults") or {}
    n_draws = int(args.draws or defaults.get("monte_carlo_draws") or 10_000)

    try:
        inventory = validate_inventory(pd.read_csv(args.inventory), config)
        if not inventory.is_valid:
            print(format_validation(inventory))
        units = inventory.frame

        synthetic = generate_synthetic_mls(
            n=args.n_listings, seed=args.seed, market=args.market, profile="rich"
        )
        frame = build_features(
            normalize_mls(synthetic.frame, config=config).frame, config
        ).frame

        surface = fit_hedonic(frame, config)
        ladder = build_price_ladder(units, surface, config)
        model = CoxDemandModel(
            covariates=available_covariates(frame), categoricals=CONTROLLED_CATEGORICALS
        )
        fit = model.fit(frame)
        beta = float(fit.beta_price.value)
        beta_se = float(fit.beta_price.std_error)

        spec_project = ProjectSpec.from_config(
            config,
            project_start=pd.Timestamp(args.start),
            phases=tuple(
                Phase(
                    index=j,
                    name=f"phase_{j + 1}",
                    start_month=float(month),
                    max_units=args.max_units,
                    competing_listings=args.competing_listings,
                )
                for j, month in enumerate(args.phase_months)
            ),
            comps=Comps(
                {
                    submarket: float(
                        frame.loc[frame["submarket"] == submarket, "cell_median_ppsf"].median()
                    )
                    for submarket in units["submarket"].unique()
                }
            ),
        )
        tensor = build_revenue_tensor(units, ladder, model, spec_project)
        provenance = DemandProvenance(
            demand_model=fit.model_kind,
            beta_price=beta,
            beta_price_se=beta_se,
            beta_price_ci95=(fit.beta_price.ci_low, fit.beta_price.ci_high),
        )
        result = solve_release_plan(tensor, units, provenance)
        if result.status is not SolveStatus.OPTIMAL:
            print(format_plan(result))
            return 1

        hazard_coefficients = {
            name: float(c.value) for name, c in fit.coefficients.items()
        }
        scenario = ScenarioSpec(
            beta_price_mean=beta,
            beta_price_se=beta_se,
            absorption_log_hazard_sd=args.absorption_sd,
            comps_drift_sd=args.comps_drift_sd,
            completion_delay_months_sd=args.completion_delay_sd,
            competing_listings_sd=0.0,
        )
        distribution = simulate_plan(
            result,
            tensor,
            units,
            scenario,
            n_draws=n_draws,
            seed=args.seed,
            presale_lead_months=spec_project.presale_lead_months,
            phase_dates=spec_project.phase_dates(),
            hazard_coefficients=hazard_coefficients,
            ladder=ladder,
        )
    except SchemaError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(format_distribution(distribution))
    if args.tornado:
        report = tornado(
            result,
            tensor,
            units,
            scenario,
            presale_lead_months=spec_project.presale_lead_months,
            phase_dates=spec_project.phase_dates(),
            hazard_coefficients=hazard_coefficients,
        )
        print()
        print(format_tornado(report))
    if args.shadow_prices:
        prices = shadow_prices(
            tensor,
            units,
            integer_objective_usd=result.objective_usd,
            provenance=result.provenance,
        )
        print()
        print(format_shadow_prices(prices))
    if args.resolve > 0:
        stability = resolve_scenarios(
            result,
            tensor,
            units,
            scenario,
            provenance,
            n_scenarios=args.resolve,
            seed=args.seed,
            hazard_coefficients=hazard_coefficients,
            presale_lead_months=spec_project.presale_lead_months,
            phase_dates=spec_project.phase_dates(),
        )
        print()
        print(
            f"PLAN STABILITY  {stability.n_solved}/{args.resolve} solved  "
            f"mean same-phase={stability.mean_same_phase_share:.1%}  "
            f"mean regret=${stability.mean_regret_usd:,.0f}  "
            f"max regret=${stability.max_regret_usd:,.0f}"
        )
        for caveat in stability.caveats:
            print(f"  !  {caveat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
