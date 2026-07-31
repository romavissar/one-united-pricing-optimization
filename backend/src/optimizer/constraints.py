"""The seven constraint families, each added independently so it can be dropped.

Separating them is not tidiness. When CBC returns infeasible, the only useful
answer is *which rule is impossible*, and the way to find that out is to re-solve
without each family in turn. That requires every family to be an addressable
unit rather than a loop in the middle of the formulation.

`PROJECT_BRIEF.md` §4 numbers them 1-7; the names here match.

**On monotonicity.** The rule is "for comparable units, phase j+1 price ≥ phase
j price", and the constraint binds on the **price-ladder index**, not on dollars.
Index `k` means "the k-th step across this unit's own comps band", so index
monotonicity says each later release is at least as aggressive relative to its
own comps as every earlier one. That is the business rule the developer means:
never signal a discount. Binding on raw dollars instead would let a small unit
at the top of its band block a large unit at the bottom of its, purely because
big units carry a higher $/sqft — a size effect masquerading as a price cut.

Comparable is `unit_type × floor_bucket`. Same product, roughly the same
elevation. A studio on 3 and a penthouse on 40 do not constrain each other.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pulp

from src.data.features import floor_bucket
from src.exceptions import SchemaError
from src.optimizer.formulate import RevenueTensor

logger = logging.getLogger(__name__)

AT_MOST_ONCE = "at_most_once"
MONOTONE_PRICE_PATH = "monotone_price_path"
CASH_FLOW_FLOOR = "cash_flow_floor"
MAX_UNITS_PER_PHASE = "max_units_per_phase"
CONSTRUCTION_GATE = "construction_gate"
TYPE_DIVERSITY = "type_diversity"
PRICE_BOUNDS = "price_bounds"

# Order matters only for reporting: the diagnostic reports the first family whose
# removal restores feasibility, and this is the order a developer would try.
RELAXABLE_FAMILIES: tuple[str, ...] = (
    CASH_FLOW_FLOOR,
    TYPE_DIVERSITY,
    MONOTONE_PRICE_PATH,
    MAX_UNITS_PER_PHASE,
    CONSTRUCTION_GATE,
)

NOMINAL = "nominal"
DISCOUNTED = "discounted"


@dataclass(frozen=True)
class ConstraintSet:
    """Which families are switched on.

    `at_most_once` and `price_bounds` are structural: the first is what makes a
    plan a plan, the second is enforced by the ladder itself and there is
    nothing to drop. Neither appears in `RELAXABLE_FAMILIES`.
    """

    at_most_once: bool = True
    monotone_price_path: bool = True
    cash_flow_floor: bool = True
    max_units_per_phase: bool = True
    construction_gate: bool = True
    type_diversity: bool = True
    # A covenant is measured in the dollars that arrive, so the floor binds on
    # nominal expected revenue by default. `PROJECT_BRIEF.md` §4.3 writes the
    # constraint against the discounted R; set DISCOUNTED to match it exactly,
    # and read CF_min as present-value dollars if you do.
    cash_flow_basis: str = NOMINAL

    def __post_init__(self) -> None:
        if self.cash_flow_basis not in {NOMINAL, DISCOUNTED}:
            raise SchemaError(
                f"cash_flow_basis must be {NOMINAL!r} or {DISCOUNTED!r}, "
                f"got {self.cash_flow_basis!r}"
            )

    def enabled(self, family: str) -> bool:
        if family in {AT_MOST_ONCE, PRICE_BOUNDS}:
            return True
        return bool(getattr(self, family))

    def without(self, family: str) -> ConstraintSet:
        """Copy with one family switched off, for the relaxation diagnostic."""
        if family not in RELAXABLE_FAMILIES:
            raise SchemaError(
                f"{family!r} is structural and cannot be relaxed. "
                f"Relaxable: {list(RELAXABLE_FAMILIES)}"
            )
        return replace(self, **{family: False})

    def active_families(self) -> tuple[str, ...]:
        return tuple(
            f
            for f in (
                AT_MOST_ONCE,
                PRICE_BOUNDS,
                MONOTONE_PRICE_PATH,
                CASH_FLOW_FLOOR,
                MAX_UNITS_PER_PHASE,
                CONSTRUCTION_GATE,
                TYPE_DIVERSITY,
            )
            if self.enabled(f)
        )


def comparable_groups(units: pd.DataFrame, unit_ids: Sequence[str]) -> dict[str, list[int]]:
    """Map `unit_type|floor_bucket` -> row positions in `unit_ids`.

    A unit with an unparseable floor gets its own singleton group rather than
    being pooled with everything else: an unknown elevation is not evidence that
    it is comparable to anything.
    """
    lookup = units.set_index(units["unit_id"].astype(str))
    groups: dict[str, list[int]] = {}
    for position, unit_id in enumerate(unit_ids):
        row = lookup.loc[unit_id]
        unit_type = str(row.get("unit_type", "unknown"))
        bucket = floor_bucket(row.get("floor"))
        key = f"{unit_type}|{bucket}" if bucket else f"{unit_type}|floor_unknown:{unit_id}"
        groups.setdefault(key, []).append(position)
    return groups


Variables = dict[tuple[int, int, int], pulp.LpVariable]


def add_at_most_once(problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor) -> None:
    """§4.1 — every unit is released at most once, or never."""
    for i in range(tensor.n_units):
        problem += (
            pulp.lpSum(
                y[i, j, k] for j in range(tensor.n_phases) for k in range(tensor.n_levels)
            )
            <= 1,
            f"{AT_MOST_ONCE}_{i}",
        )


def add_construction_gate(problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor) -> None:
    """§4.5 — nothing is released before its building can be presold."""
    for i in range(tensor.n_units):
        for j in range(tensor.n_phases):
            if tensor.releasable[i, j]:
                continue
            problem += (
                pulp.lpSum(y[i, j, k] for k in range(tensor.n_levels)) == 0,
                f"{CONSTRUCTION_GATE}_{i}_{j}",
            )


def add_max_units_per_phase(problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor) -> None:
    """§4.4 — sales-team capacity."""
    for j, phase in enumerate(tensor.phases):
        if phase.max_units is None:
            continue
        problem += (
            pulp.lpSum(
                y[i, j, k] for i in range(tensor.n_units) for k in range(tensor.n_levels)
            )
            <= int(phase.max_units),
            f"{MAX_UNITS_PER_PHASE}_{j}",
        )


def add_cash_flow_floor(
    problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor, *, basis: str = NOMINAL
) -> None:
    """§4.3 — each phase clears its floor, *in expectation*.

    The expectation is the caveat. A plan whose expected phase revenue exactly
    equals a loan covenant breaches that covenant roughly half the time. Phase 5
    reports the simulated breach probability; buffering the floor is the fix,
    not tightening this constraint.
    """
    revenue = tensor.expected_revenue_usd if basis == NOMINAL else tensor.discounted_usd
    for j, phase in enumerate(tensor.phases):
        floor = float(phase.cash_flow_floor_usd or 0.0)
        if floor <= 0:
            continue
        problem += (
            pulp.lpSum(
                float(revenue[i, j, k]) * y[i, j, k]
                for i in range(tensor.n_units)
                for k in range(tensor.n_levels)
            )
            >= floor,
            f"{CASH_FLOW_FLOOR}_{j}",
        )


def add_type_diversity(
    problem: pulp.LpProblem,
    y: Variables,
    tensor: RevenueTensor,
    units: pd.DataFrame,
    active: Mapping[int, pulp.LpVariable],
) -> None:
    """§4.6 — an active phase carries a minimum count of each named unit type.

    Conditional on the phase being active: an empty phase is allowed, a phase
    that releases only penthouses is not.
    """
    types = _unit_types(units, tensor.unit_ids)
    for j, phase in enumerate(tensor.phases):
        for unit_type, minimum in (phase.min_type_counts or {}).items():
            if int(minimum) <= 0:
                continue
            members = [i for i, t in enumerate(types) if t == str(unit_type)]
            if not members:
                raise SchemaError(
                    f"Phase {phase.name!r} requires at least {minimum} units of type "
                    f"{unit_type!r}, but no unit in the inventory has that type. "
                    "This constraint can never be satisfied."
                )
            problem += (
                pulp.lpSum(y[i, j, k] for i in members for k in range(tensor.n_levels))
                >= int(minimum) * active[j],
                f"{TYPE_DIVERSITY}_{j}_{unit_type}",
            )


def add_phase_activity(
    problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor
) -> dict[int, pulp.LpVariable]:
    """Binary "this phase releases something", linked to the release variables."""
    active: dict[int, pulp.LpVariable] = {}
    for j in range(tensor.n_phases):
        indicator = problem.add_variable(f"phase_active_{j}", cat="Binary")
        released = pulp.lpSum(
            y[i, j, k] for i in range(tensor.n_units) for k in range(tensor.n_levels)
        )
        problem += (released <= tensor.n_units * indicator, f"phase_active_upper_{j}")
        problem += (indicator <= released, f"phase_active_lower_{j}")
        active[j] = indicator
    return active


def add_monotone_price_path(
    problem: pulp.LpProblem, y: Variables, tensor: RevenueTensor, units: pd.DataFrame
) -> dict[str, Any]:
    """§4.2 — within a comparable group, later releases never price below earlier ones.

    Implemented with a per-(group, phase) pair of continuous variables: `hi` is
    at least every chosen level index in that cell, `lo` is at most every chosen
    one. Then `lo[g, j'] >= hi[g, j]` for every `j < j'`.

    All pairs rather than consecutive phases, deliberately. With only
    consecutive links, a phase that releases nothing from the group leaves `hi`
    free to fall to zero and resets the ladder, so a group could price high in
    phase 0, skip phase 1, and price at the bottom of the band in phase 2. The
    quadratic count is `groups × phases²`, which is nothing at these sizes.

    When a cell releases nothing, `hi` settles at 0 and `lo` at K-1 and every
    constraint touching it is slack — vacuous, which is correct.
    """
    groups = comparable_groups(units, tensor.unit_ids)
    if tensor.n_phases < 2:
        # Nothing to order. Building the level variables anyway would add
        # thousands of constraints that can never bind.
        return {"groups": {k: len(v) for k, v in groups.items()}, "ordering_constraints": 0}

    top = tensor.n_levels - 1
    created = 0
    for name, members in groups.items():
        safe = _slug(name)
        lo = {
            j: problem.add_variable(f"lvl_lo_{safe}_{j}", lowBound=0, upBound=top)
            for j in range(tensor.n_phases)
        }
        hi = {
            j: problem.add_variable(f"lvl_hi_{safe}_{j}", lowBound=0, upBound=top)
            for j in range(tensor.n_phases)
        }
        for j in range(tensor.n_phases):
            for i in members:
                for k in range(tensor.n_levels):
                    problem += (hi[j] >= k * y[i, j, k], f"lvl_hi_{safe}_{j}_{i}_{k}")
                    problem += (
                        lo[j] <= k * y[i, j, k] + top * (1 - y[i, j, k]),
                        f"lvl_lo_{safe}_{j}_{i}_{k}",
                    )
        for earlier in range(tensor.n_phases):
            for later in range(earlier + 1, tensor.n_phases):
                problem += (
                    lo[later] >= hi[earlier],
                    f"{MONOTONE_PRICE_PATH}_{safe}_{earlier}_{later}",
                )
                created += 1
    logger.debug("Monotone price path: %d groups, %d ordering constraints", len(groups), created)
    return {"groups": {k: len(v) for k, v in groups.items()}, "ordering_constraints": created}


def _unit_types(units: pd.DataFrame, unit_ids: Iterable[str]) -> list[str]:
    lookup = units.set_index(units["unit_id"].astype(str))["unit_type"]
    return [str(lookup.loc[u]) for u in unit_ids]


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def constraint_summary(tensor: RevenueTensor, constraints: ConstraintSet) -> dict[str, Any]:
    """What the formulation actually asked for, for the response and the report."""
    return {
        "active_families": list(constraints.active_families()),
        "cash_flow_basis": constraints.cash_flow_basis,
        "phases": [p.as_dict() for p in tensor.phases],
        "gated_cells": int((~tensor.releasable).sum()),
        "n_units": tensor.n_units,
        "n_levels": tensor.n_levels,
        "n_binaries": int(np.prod(tensor.shape)),
    }
