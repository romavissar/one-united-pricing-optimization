"""Build the MILP, hand it to CBC, and report honestly about what came back.

Two rules govern this module.

**Never return a silent empty plan.** An empty plan and an infeasible problem
look identical in a table of zero rows, and they mean opposite things: one says
"releasing nothing is optimal", the other says "your constraints contradict each
other". Infeasibility triggers the relaxation diagnostic, which re-solves
without each constraint family in turn and names the one whose removal restores
feasibility.

**Every result carries provenance.** `AGENTS.md` §4. The block includes
`is_calibrated_on_real_data`, and when it is False the caller is expected to
render a banner. That flag is why the plan cannot be quietly screenshotted into
a deck.

Units: prices `$/sqft`, revenue USD, phase timing in months.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pulp

from src.exceptions import InfeasibleModelError
from src.optimizer.constraints import (
    RELAXABLE_FAMILIES,
    ConstraintSet,
    add_at_most_once,
    add_cash_flow_floor,
    add_construction_gate,
    add_max_units_per_phase,
    add_monotone_price_path,
    add_phase_activity,
    add_type_diversity,
    comparable_groups,
    constraint_summary,
)
from src.optimizer.formulate import DemandProvenance, Phase, RevenueTensor

logger = logging.getLogger(__name__)

_DEFAULT_TIME_LIMIT_SECONDS = 60
# CBC reports fractional values for variables it has fixed at an integer.
_BINARY_THRESHOLD = 0.5
# Share of released units at their ceiling that triggers the "this is just
# charge-the-maximum" warning. Not 100%: a plan with half the stack pinned to
# the top of its band is already the failure this project exists to avoid, and
# waiting for four fifths lets the common case through unremarked — on the
# example inventory two thirds of units land on their ceiling and nothing said
# so. The threshold is a reporting trigger, not an estimate.
_CEILING_SHARE_ALARM = 0.50
_DAYS_PER_MONTH_APPROX = 30.4375


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    UNDEFINED = "undefined"


_PULP_STATUS = {
    pulp.LpStatusOptimal: SolveStatus.OPTIMAL,
    pulp.LpStatusNotSolved: SolveStatus.UNDEFINED,
    pulp.LpStatusInfeasible: SolveStatus.INFEASIBLE,
    pulp.LpStatusUnbounded: SolveStatus.UNBOUNDED,
    pulp.LpStatusUndefined: SolveStatus.UNDEFINED,
}


@dataclass(frozen=True)
class PlanRow:
    """One released unit: when, at what price, and what that is worth."""

    unit_id: str
    phase_index: int
    phase_name: str
    level_index: int
    price_ppsf: float
    total_price_usd: float
    sale_probability: float
    expected_revenue_usd: float
    discounted_expected_revenue_usd: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseSummary:
    """Per-phase rollup, including whether the cash-flow floor is met."""

    index: int
    name: str
    start_month: float
    units_released: int
    expected_revenue_usd: float
    discounted_expected_revenue_usd: float
    cash_flow_floor_usd: float
    floor_met: bool
    mean_price_ppsf: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtrapolationReport:
    """Whether the recommended prices sit where `beta_price` is evidence.

    A Cox linear predictor evaluates anywhere. Asked what happens at a price
    50% above anything in the fitting sample, it returns a number, and that
    number has the same units and the same decimal places as an estimate. The
    only thing separating them is this report.
    """

    checked: bool
    support: dict[str, float | int] | None
    units_outside: list[str] = field(default_factory=list)
    units_in_tail: list[str] = field(default_factory=list)
    min_premium: float | None = None
    max_premium: float | None = None
    message: str = ""

    @property
    def is_clean(self) -> bool:
        """True when the check ran and every price landed inside the support."""
        return self.checked and not self.units_outside

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "support": self.support,
            "units_outside": self.units_outside,
            "units_in_tail": self.units_in_tail,
            "min_premium": self.min_premium,
            "max_premium": self.max_premium,
            "message": self.message,
        }


@dataclass
class InfeasibilityReport:
    """Which constraint family is doing the blocking, established by re-solving."""

    binding_family: str | None
    tested: dict[str, str] = field(default_factory=dict)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "binding_family": self.binding_family,
            "tested": self.tested,
            "message": self.message,
        }


@dataclass
class OptimizeResult:
    """The plan, its economics, and every caveat attached to it."""

    status: SolveStatus
    plan: list[PlanRow]
    objective_usd: float
    expected_revenue_usd: float
    per_phase: list[PhaseSummary]
    unreleased_unit_ids: list[str]
    excluded_units: list[dict[str, str]]
    provenance: dict[str, Any]
    constraints: dict[str, Any]
    caveats: list[str]
    solve_seconds: float
    extrapolation: ExtrapolationReport | None = None
    infeasibility: InfeasibilityReport | None = None

    @property
    def units_released(self) -> int:
        return len(self.plan)

    @property
    def gross_ask_usd(self) -> float:
        """Total asking value of the released stack if every unit sold."""
        return float(sum(r.total_price_usd for r in self.plan))

    @property
    def unmodelled_probability_mass(self) -> float:
        """`SUM_i (1 - D_i)` over the plan — units the objective never accounts for.

        A unit released at `D = 0.7` contributes `0.7 * p * A` to the objective.
        The other 30% does not reappear anywhere: constraint C1 releases a unit
        at most once and the formulation carries no unsold inventory into a
        later phase, so that share simply vanishes. This number is how much of
        the stack that is. Read with `unmodelled_revenue_usd`: together they say
        what fraction of "expected revenue" is a partial sum rather than a total.
        """
        return float(sum(1.0 - r.sale_probability for r in self.plan))

    @property
    def unmodelled_revenue_usd(self) -> float:
        """Asking value attached to the unmodelled probability mass, in USD."""
        return float(
            sum(r.total_price_usd * (1.0 - r.sale_probability) for r in self.plan)
        )

    def as_frame(self) -> pd.DataFrame:
        """The plan as a table, one row per released unit."""
        if not self.plan:
            return pd.DataFrame(columns=[f.name for f in PlanRow.__dataclass_fields__.values()])
        return pd.DataFrame([row.as_dict() for row in self.plan])

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "objective_usd": self.objective_usd,
            "expected_revenue_usd": self.expected_revenue_usd,
            "gross_ask_usd": self.gross_ask_usd,
            "unmodelled_probability_mass": self.unmodelled_probability_mass,
            "unmodelled_revenue_usd": self.unmodelled_revenue_usd,
            "units_released": self.units_released,
            "plan": [row.as_dict() for row in self.plan],
            "per_phase": [p.as_dict() for p in self.per_phase],
            "unreleased_unit_ids": self.unreleased_unit_ids,
            "excluded_units": self.excluded_units,
            "provenance": self.provenance,
            "constraints": self.constraints,
            "caveats": self.caveats,
            "solve_seconds": self.solve_seconds,
            "extrapolation": self.extrapolation.as_dict() if self.extrapolation else None,
            "infeasibility": self.infeasibility.as_dict() if self.infeasibility else None,
        }


@dataclass
class MilpProgram:
    """A built but unsolved problem, plus the handles needed to read it back."""

    problem: pulp.LpProblem
    y: dict[tuple[int, int, int], pulp.LpVariable]
    tensor: RevenueTensor
    units: pd.DataFrame
    constraints: ConstraintSet
    monotone_groups: dict[str, Any] = field(default_factory=dict)


def build_program(
    tensor: RevenueTensor,
    units: pd.DataFrame,
    constraints: ConstraintSet | None = None,
    *,
    relax: bool = False,
) -> MilpProgram:
    """Assemble the MILP: binaries, linear objective, and the active families.

    Args:
        tensor: precomputed `R[i,j,k]`.
        units: validated inventory, for grouping and unit types.
        constraints: which families to apply. Defaults to all of them.
        relax: when True, variables are continuous on `[0, 1]` instead of
            binary. The LP relaxation is what makes shadow prices available —
            CBC does not report duals on a MIP. Phase 5 uses this for the
            cash-flow floor duals; the integer plan itself still comes from
            the unrelaxed solve.

    Returns:
        An unsolved `MilpProgram`.
    """
    rules = constraints or ConstraintSet()
    problem = pulp.LpProblem("release_plan", pulp.LpMaximize)
    category = "Continuous" if relax else "Binary"

    y = {
        (i, j, k): problem.add_variable(
            f"y_{i}_{j}_{k}", lowBound=0, upBound=1, cat=category
        )
        for i in range(tensor.n_units)
        for j in range(tensor.n_phases)
        for k in range(tensor.n_levels)
    }

    problem += pulp.lpSum(
        float(tensor.discounted_usd[i, j, k]) * y[i, j, k] for (i, j, k) in y
    ), "discounted_expected_revenue"

    if rules.enabled("at_most_once"):
        add_at_most_once(problem, y, tensor)
    if rules.construction_gate:
        add_construction_gate(problem, y, tensor)
    if rules.max_units_per_phase:
        add_max_units_per_phase(problem, y, tensor)
    if rules.cash_flow_floor:
        add_cash_flow_floor(problem, y, tensor, basis=rules.cash_flow_basis)

    monotone: dict[str, Any] = {}
    if rules.monotone_price_path:
        monotone = add_monotone_price_path(problem, y, tensor, units)
    if rules.type_diversity and any(p.min_type_counts for p in tensor.phases):
        active = add_phase_activity(problem, y, tensor)
        add_type_diversity(problem, y, tensor, units, active)

    return MilpProgram(
        problem=problem,
        y=y,
        tensor=tensor,
        units=units,
        constraints=rules,
        monotone_groups=monotone,
    )


@lru_cache(maxsize=1)
def _solver_class() -> type[pulp.LpSolver]:
    """CBC, preferring a system install over PuLP's bundled binary.

    `PULP_CBC_CMD` is deprecated in PuLP 3.x in favour of `COIN_CMD`, but
    `COIN_CMD` needs `cbc` on PATH and the bundled binary is what makes this
    work out of the box. Probe once, prefer the supported entry point, fall
    back to the one that is guaranteed present.
    """
    try:
        if pulp.COIN_CMD(msg=False).available():
            return pulp.COIN_CMD
    except pulp.PulpError:  # pragma: no cover - depends on the local install
        pass
    return pulp.PULP_CBC_CMD


def _solver(time_limit_seconds: int) -> pulp.LpSolver:
    return _solver_class()(msg=False, timeLimit=int(time_limit_seconds))


def _selected(program: MilpProgram) -> list[tuple[int, int, int]]:
    return [key for key, var in program.y.items() if (var.value() or 0.0) > _BINARY_THRESHOLD]


def _plan_rows(program: MilpProgram, chosen: list[tuple[int, int, int]]) -> list[PlanRow]:
    tensor = program.tensor
    rows: list[PlanRow] = []
    for i, j, k in sorted(chosen, key=lambda key: (key[1], key[0])):
        price = float(tensor.price_ppsf[i, k])
        rows.append(
            PlanRow(
                unit_id=tensor.unit_ids[i],
                phase_index=j,
                phase_name=tensor.phases[j].name,
                level_index=k,
                price_ppsf=price,
                total_price_usd=price * float(tensor.area_sqft[i]),
                sale_probability=float(tensor.probability[i, j, k]),
                expected_revenue_usd=float(tensor.expected_revenue_usd[i, j, k]),
                discounted_expected_revenue_usd=float(tensor.discounted_usd[i, j, k]),
            )
        )
    return rows


def _phase_summaries(
    phases: tuple[Phase, ...], rows: list[PlanRow], basis: str
) -> list[PhaseSummary]:
    summaries: list[PhaseSummary] = []
    for phase in phases:
        members = [r for r in rows if r.phase_index == phase.index]
        expected = sum(r.expected_revenue_usd for r in members)
        discounted = sum(r.discounted_expected_revenue_usd for r in members)
        measured = discounted if basis == "discounted" else expected
        floor = float(phase.cash_flow_floor_usd or 0.0)
        summaries.append(
            PhaseSummary(
                index=phase.index,
                name=phase.name,
                start_month=phase.start_month,
                units_released=len(members),
                expected_revenue_usd=expected,
                discounted_expected_revenue_usd=discounted,
                cash_flow_floor_usd=floor,
                # Tolerance is one cent: CBC returns floating-point values and a
                # floor met to eleven decimal places should not read as breached.
                floor_met=measured >= floor - 0.01,
                mean_price_ppsf=(
                    float(np.mean([r.price_ppsf for r in members])) if members else None
                ),
            )
        )
    return summaries


def check_extrapolation(tensor: RevenueTensor, rows: list[PlanRow]) -> ExtrapolationReport:
    """Compare every recommended price against the range `beta_price` was fitted on.

    This is the difference between "the model predicts a 62% chance of sale" and
    "the model, asked about a price it has never seen, returned 0.62". Both are
    floats. Only one is evidence.
    """
    support = tensor.premium_support
    if support is None:
        return ExtrapolationReport(
            checked=False,
            support=None,
            message=(
                "Extrapolation was not checked: the demand model does not report the "
                "rel_price_premium range it was fitted over. Treat prices far from "
                "comps as unverified."
            ),
        )
    if not rows:
        return ExtrapolationReport(
            checked=True, support=support.as_dict(), message="No units released."
        )

    premiums = np.array(
        [
            tensor.rel_price_premium[
                tensor.unit_ids.index(row.unit_id), row.phase_index, row.level_index
            ]
            for row in rows
        ]
    )
    outside = support.outside(premiums)
    in_tail = support.in_tail(premiums)
    report = ExtrapolationReport(
        checked=True,
        support=support.as_dict(),
        units_outside=[rows[i].unit_id for i in np.flatnonzero(outside)],
        units_in_tail=[rows[i].unit_id for i in np.flatnonzero(in_tail)],
        min_premium=float(premiums.min()),
        max_premium=float(premiums.max()),
    )
    if report.units_outside:
        report.message = (
            f"{len(report.units_outside)} of {len(rows)} recommended prices imply a "
            f"rel_price_premium outside [{support.low:+.3f}, {support.high:+.3f}], the "
            f"range beta_price was estimated over. The sale probabilities behind those "
            "prices are extrapolations, not predictions. Narrow the price bands or "
            "refit on data that covers the range you intend to price in."
        )
    elif report.units_in_tail:
        report.message = (
            f"{len(report.units_in_tail)} of {len(rows)} prices sit in the thin tails "
            f"of the fitted range (outside [{support.p1:+.3f}, {support.p99:+.3f}] but "
            "inside its bounds). Supported, but by few listings."
        )
    else:
        report.message = (
            f"All prices imply a rel_price_premium within [{support.p1:+.3f}, "
            f"{support.p99:+.3f}], well inside the fitted range."
        )
    return report


def _caveats(program: MilpProgram, rows: list[PlanRow], horizon_days: int | None = None) -> list[str]:
    tensor = program.tensor
    caveats = list(tensor.notes)
    caveats.append(
        "Sale probabilities are independent across units. Units released together "
        "compete for one buyer pool, so phase revenue is overstated where a phase "
        "releases many near-identical units."
    )
    if rows:
        unmodelled = sum(1.0 - r.sale_probability for r in rows)
        ask = sum(r.total_price_usd for r in rows)
        expected = sum(r.expected_revenue_usd for r in rows)
        caveats.append(
            f"Expected revenue is a partial sum, not a total. Each unit contributes "
            f"only its P(sale) share, and the remaining {unmodelled:.1f} units' worth "
            f"of probability mass across {len(rows)} released units "
            f"({unmodelled / len(rows):.0%} of the stack) is not carried into a later "
            f"phase — the formulation releases a unit at most once and models no "
            f"unsold inventory, so that share vanishes. ${ask - expected:,.0f} of the "
            f"${ask:,.0f} asking value is therefore never counted. Do not compare this "
            "figure to the gross development value of a sold-out project."
        )
    if horizon_days:
        months = horizon_days / _DAYS_PER_MONTH_APPROX
        if tensor.arrival_discount_factor is None:
            # No fitted survival curve to integrate against, so the objective
            # falls back to valuing everything at its phase's release date.
            caveats.append(
                f"Timing is approximated. P(sale) is the probability of selling "
                f"within {horizon_days} days ({months:.1f} months), but the whole of "
                "a unit's expected revenue is discounted to its phase's release date "
                "rather than to when the cash actually arrives, which is spread "
                "across that window. Late sales are therefore under-discounted and "
                "the objective overstates present value by roughly the discount over "
                "half the horizon; the bias also mildly favours later phases "
                "relative to a correct treatment."
            )
        else:
            # Arrival-time discounting is in force. The residual approximation is
            # the horizon cap, not the release-date shortcut — saying otherwise
            # contradicts the note the tensor already carries.
            caveats.append(
                f"Timing is discounted to expected sale time, integrated against the "
                f"fitted survival conditional on selling within {horizon_days} days "
                f"({months:.1f} months) — not to the phase release date. Two "
                "approximations remain. The horizon is capped at the longest "
                "follow-up the demand model actually observed, so a sale that would "
                "arrive later is valued as if it arrived at the cap; and the "
                "conditional arrival distribution is the fitted one, so it inherits "
                "whatever the hazard gets wrong."
            )
    if any(p.cash_flow_floor_usd for p in tensor.phases):
        caveats.append(
            "The cash-flow floor binds on expected revenue, so realized cash flow can "
            "still fall below it. Where the floor is a covenant, simulate the breach "
            "probability and buffer the floor rather than treating it as guaranteed."
        )
    caveats.append(
        "Competitor pricing is exogenous. Nothing here models a rival cutting price "
        "in response to this plan; that belongs in scenario analysis."
    )
    caveats.append(
        "Demand parameters are fitted once and held fixed across every phase. A real "
        "12-18 month sales period drifts; re-fitting between phases is not yet built."
    )
    ceiling_hits = sum(1 for r in rows if r.level_index == tensor.n_levels - 1)
    if rows and ceiling_hits / len(rows) >= _CEILING_SHARE_ALARM:
        caveats.append(
            f"{ceiling_hits} of {len(rows)} released units sit at their price ceiling "
            f"({ceiling_hits / len(rows):.0%}). Charging the maximum is what an "
            "optimizer does when demand barely responds to price — check that "
            "beta_price is negative and materially different from zero, and that the "
            "ceilings are real comps rather than an unbounded band, before treating "
            "these as recommendations."
        )
    return caveats


def diagnose_infeasibility(
    tensor: RevenueTensor,
    units: pd.DataFrame,
    constraints: ConstraintSet,
    *,
    time_limit_seconds: int = _DEFAULT_TIME_LIMIT_SECONDS,
) -> InfeasibilityReport:
    """Drop one constraint family at a time and report which one unblocks the model.

    Only families that were switched on are worth testing. If dropping every one
    individually still leaves the problem infeasible, the constraints conflict in
    combination and the report says exactly that instead of naming a scapegoat.
    """
    tested: dict[str, str] = {}
    binding: str | None = None
    for family in RELAXABLE_FAMILIES:
        if not constraints.enabled(family):
            continue
        relaxed = build_program(tensor, units, constraints.without(family))
        relaxed.problem.solve(_solver(time_limit_seconds))
        status = _PULP_STATUS.get(relaxed.problem.status, SolveStatus.UNDEFINED)
        tested[family] = status.value
        if binding is None and status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
            binding = family

    if binding:
        message = (
            f"The plan is infeasible as specified. Removing {binding!r} restores "
            "feasibility, so that is the binding rule. Loosen it — or accept that no "
            "plan satisfies it — rather than reading anything into an empty plan."
        )
    elif tested:
        message = (
            "The plan is infeasible and no single constraint family is responsible: "
            f"dropping each of {sorted(tested)} on its own leaves it infeasible. The "
            "constraints conflict in combination, or the inventory is too small to "
            "satisfy them at all."
        )
    else:
        message = (
            "The plan is infeasible with no relaxable constraints active. That points "
            "at the inventory or the price bands rather than at the release rules."
        )
    logger.warning("Infeasibility diagnostic: %s", message)
    return InfeasibilityReport(binding_family=binding, tested=tested, message=message)


def solve_release_plan(
    tensor: RevenueTensor,
    units: pd.DataFrame,
    provenance: DemandProvenance,
    *,
    constraints: ConstraintSet | None = None,
    time_limit_seconds: int = _DEFAULT_TIME_LIMIT_SECONDS,
    diagnose: bool = True,
) -> OptimizeResult:
    """Solve for a release plan and return it with its caveats.

    Args:
        tensor: precomputed revenue.
        units: validated inventory.
        provenance: where the demand model came from. Travels into the response.
        constraints: which families to apply.
        time_limit_seconds: CBC wall-clock cap.
        diagnose: run the relaxation diagnostic on infeasibility.

    Returns:
        An `OptimizeResult`. On infeasibility the plan is empty *and*
        `infeasibility` is populated, so the two cases are never confusable.
    """
    rules = constraints or ConstraintSet()
    program = build_program(tensor, units, rules)

    started = time.perf_counter()
    program.problem.solve(_solver(time_limit_seconds))
    elapsed = time.perf_counter() - started
    status = _PULP_STATUS.get(program.problem.status, SolveStatus.UNDEFINED)

    summary = constraint_summary(tensor, rules)
    summary["monotone_groups"] = program.monotone_groups
    excluded = [e.as_dict() for e in tensor.excluded]

    if status is not SolveStatus.OPTIMAL:
        report = (
            diagnose_infeasibility(tensor, units, rules, time_limit_seconds=time_limit_seconds)
            if diagnose and status is SolveStatus.INFEASIBLE
            else None
        )
        logger.warning("Solve returned %s after %.2fs", status.value, elapsed)
        return OptimizeResult(
            status=status,
            plan=[],
            objective_usd=0.0,
            expected_revenue_usd=0.0,
            per_phase=_phase_summaries(tensor.phases, [], rules.cash_flow_basis),
            unreleased_unit_ids=list(tensor.unit_ids),
            excluded_units=excluded,
            provenance=provenance.as_response_block(),
            constraints=summary,
            caveats=[
                f"No plan was produced: the solver returned {status.value}. This is not "
                "a recommendation to release nothing."
            ],
            solve_seconds=elapsed,
            infeasibility=report,
        )

    chosen = _selected(program)
    rows = _plan_rows(program, chosen)
    released = {row.unit_id for row in rows}
    extrapolation = check_extrapolation(tensor, rows)

    caveats = _caveats(program, rows, horizon_days=tensor.horizon_days)
    if not extrapolation.is_clean:
        caveats.insert(0, extrapolation.message)

    result = OptimizeResult(
        status=status,
        plan=rows,
        objective_usd=float(pulp.value(program.problem.objective) or 0.0),
        expected_revenue_usd=float(sum(r.expected_revenue_usd for r in rows)),
        per_phase=_phase_summaries(tensor.phases, rows, rules.cash_flow_basis),
        unreleased_unit_ids=[u for u in tensor.unit_ids if u not in released],
        excluded_units=excluded,
        provenance=provenance.as_response_block(),
        constraints=summary,
        caveats=caveats,
        solve_seconds=elapsed,
        extrapolation=extrapolation,
    )
    logger.info(
        "Solved %s in %.2fs: %d/%d units released, objective $%.0f",
        status.value, elapsed, len(rows), tensor.n_units, result.objective_usd,
    )
    return result


def assert_feasible(result: OptimizeResult) -> OptimizeResult:
    """Raise `InfeasibleModelError` unless a plan came back.

    For callers that would rather fail than branch. The message carries the
    binding constraint so the exception is actionable on its own.
    """
    if result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
        return result
    detail = result.infeasibility.message if result.infeasibility else result.status.value
    raise InfeasibleModelError(detail)


def check_monotone(result: OptimizeResult, units: pd.DataFrame) -> list[str]:
    """Verify monotonicity on the returned plan; return one string per violation.

    Checks the output rather than trusting the formulation. Empty list means the
    plan is clean.
    """
    unit_ids = tuple(row.unit_id for row in result.plan)
    if not unit_ids:
        return []
    groups = comparable_groups(units, unit_ids)
    rows = list(result.plan)
    violations: list[str] = []
    for name, members in groups.items():
        by_phase: dict[int, list[int]] = {}
        for position in members:
            row = rows[position]
            by_phase.setdefault(row.phase_index, []).append(row.level_index)
        phases = sorted(by_phase)
        for a_index, earlier in enumerate(phases):
            for later in phases[a_index + 1:]:
                if min(by_phase[later]) < max(by_phase[earlier]):
                    violations.append(
                        f"group {name}: phase {later} prices at level "
                        f"{min(by_phase[later])}, below phase {earlier}'s level "
                        f"{max(by_phase[earlier])}"
                    )
    return violations


def format_plan(result: OptimizeResult) -> str:
    """Human-readable plan report."""
    lines = [
        f"RELEASE PLAN  status={result.status.value}  "
        f"units={result.units_released}  solve={result.solve_seconds:.2f}s",
        f"  discounted expected revenue: ${result.objective_usd:,.0f}",
        f"  undiscounted expected revenue: ${result.expected_revenue_usd:,.0f}",
        "",
        "PHASES",
    ]
    for phase in result.per_phase:
        mark = "ok " if phase.floor_met else "BREACH"
        price = f"${phase.mean_price_ppsf:,.0f}/sqft" if phase.mean_price_ppsf else "-"
        lines.append(
            f"  [{mark}] {phase.name:<14} m{phase.start_month:>5.1f}  "
            f"{phase.units_released:>3} units  mean {price:>14}  "
            f"exp ${phase.expected_revenue_usd:>14,.0f}  floor ${phase.cash_flow_floor_usd:>12,.0f}"
        )
    if result.unreleased_unit_ids:
        lines.append("")
        lines.append(f"UNRELEASED ({len(result.unreleased_unit_ids)})")
        lines.append(f"  {', '.join(result.unreleased_unit_ids[:20])}")
    if result.excluded_units:
        lines.append("")
        lines.append(f"EXCLUDED BEFORE OPTIMIZATION ({len(result.excluded_units)})")
        for entry in result.excluded_units[:10]:
            lines.append(f"  {entry['unit_id']} [{entry['reason']}] {entry['detail']}")
    if result.extrapolation:
        lines.append("")
        mark = "ok" if result.extrapolation.is_clean else "!!"
        lines.append(f"PRICE SUPPORT [{mark}]")
        lines.append(f"  {result.extrapolation.message}")
        if result.extrapolation.min_premium is not None:
            lines.append(
                f"  chosen rel_price_premium spans "
                f"{result.extrapolation.min_premium:+.3f} to "
                f"{result.extrapolation.max_premium:+.3f}"
            )
    if result.infeasibility:
        lines.append("")
        lines.append("INFEASIBLE")
        lines.append(f"  {result.infeasibility.message}")
        for family, status in result.infeasibility.tested.items():
            lines.append(f"    without {family}: {status}")
    lines.append("")
    lines.append("PROVENANCE")
    for key, value in result.provenance.items():
        if key != "warnings":
            lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("CAVEATS")
    for caveat in [*result.provenance.get("warnings", []), *result.caveats]:
        lines.append(f"  !  {caveat}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Fit on a synthetic export, solve a plan for an inventory, and print it.

    Exit codes: 0 optimal, 1 no plan. Nothing here reads the real MLS export —
    `PROJECT_BRIEF.md` §5 gates that behind a human decision, and a release plan
    is exactly the artefact that must not be produced from real data by
    accident.
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
    from src.exceptions import SchemaError
    from src.optimizer.crowding import apply_crowding, crowded_result
    from src.optimizer.discretize import build_price_ladder
    from src.optimizer.formulate import Comps, ProjectSpec, build_revenue_tensor
    from src.utils.validate import format_validation, validate_inventory

    backend_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Solve a release plan on synthetic demand")
    parser.add_argument("--inspect", action="store_true", help="print the plan report")
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
    parser.add_argument(
        "--crowding-buyer-pool",
        type=float,
        default=None,
        help="apply the crowding correction against this many buyers per phase",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_market_config(args.market)

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

        spec = ProjectSpec.from_config(
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
        if args.crowding_buyer_pool:
            result = crowded_result(
                result,
                apply_crowding(
                    result, units, buyer_pool=args.crowding_buyer_pool, config=config
                ),
            )
    except SchemaError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(format_plan(result))
    violations = check_monotone(result, units)
    if violations:
        print("\nMONOTONICITY VIOLATIONS")
        for violation in violations:
            print(f"  {violation}")
        return 1
    return 0 if result.status is SolveStatus.OPTIMAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
