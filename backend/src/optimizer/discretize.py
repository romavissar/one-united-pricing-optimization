"""Per-unit price ladders — the K discrete prices the MILP is allowed to choose.

Discretizing is what makes the problem linear. The revenue of a unit at a price
is `price × area × P(sale | price)`, and `P` comes out of a Cox model, so the
product is not linear in price and not concave in it either. Enumerating K
prices per unit and precomputing revenue at each turns a mixed-integer
*nonlinear* program into a MILP that CBC finishes in seconds.

The band each ladder spans is the intersection of two things that are not the
same kind of object:

- the **comps ceiling and floor** from the hedonic surface — fitted, an estimate
  of what the market pays for a unit like this one, plus or minus a residual
  spread;
- the **cost floor**, `cost_basis_ppsf × (1 + min_margin_over_cost)` — a
  business rule the developer sets, not an estimate of anything.

The binding floor is whichever is higher. When the cost floor is the binding
one, that is worth saying out loud: it means the developer's basis, not the
market, is setting the price, and the optimizer's room to respond to elasticity
is correspondingly smaller.

When the cost floor exceeds the comps ceiling, the unit cannot be sold at comps
for the required margin at all. It is excluded from the ladder and named in
`excluded`, rather than being given a one-point ladder that would quietly pin it
at a price the model says will not clear.

Units: every price in this module is `$/sqft`. Nothing here is a total price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import MarketConfig
from src.demand.hedonic import HedonicSurface
from src.exceptions import SchemaError
from src.utils.validate import inventory_scoring_frame

logger = logging.getLogger(__name__)

_DEFAULT_LEVELS = 15
_DEFAULT_MIN_MARGIN = 0.15
# Below this the band is a point and the optimizer has no price decision left.
_MIN_BAND_WIDTH_PPSF = 1.0

FLOOR_FROM_COST = "cost_basis"
FLOOR_FROM_COMPS = "hedonic_comps"


@dataclass(frozen=True)
class ExcludedUnit:
    """A unit that has no feasible price band, and why."""

    unit_id: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"unit_id": self.unit_id, "reason": self.reason, "detail": self.detail}


@dataclass
class PriceLadder:
    """K ascending $/sqft levels per unit, aligned row-wise with `unit_ids`."""

    unit_ids: tuple[str, ...]
    levels_ppsf: np.ndarray
    p_floor_ppsf: np.ndarray
    p_ceiling_ppsf: np.ndarray
    predicted_ppsf: np.ndarray
    floor_source: tuple[str, ...]
    excluded: tuple[ExcludedUnit, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def n_levels(self) -> int:
        return int(self.levels_ppsf.shape[1]) if self.levels_ppsf.size else 0

    def index_of(self, unit_id: str) -> int:
        try:
            return self.unit_ids.index(unit_id)
        except ValueError as exc:
            raise SchemaError(f"{unit_id!r} has no price ladder") from exc

    @property
    def cost_bound_units(self) -> tuple[str, ...]:
        """Units whose floor comes from the cost basis rather than from comps."""
        return tuple(
            uid for uid, src in zip(self.unit_ids, self.floor_source, strict=True)
            if src == FLOOR_FROM_COST
        )

    def as_frame(self) -> pd.DataFrame:
        """One row per unit: the band and its endpoints, for reporting."""
        return pd.DataFrame(
            {
                "unit_id": list(self.unit_ids),
                "predicted_ppsf": self.predicted_ppsf,
                "p_floor_ppsf": self.p_floor_ppsf,
                "p_ceiling_ppsf": self.p_ceiling_ppsf,
                "floor_source": list(self.floor_source),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_units": self.n_units,
            "n_levels": self.n_levels,
            "cost_bound_units": len(self.cost_bound_units),
            "excluded": [e.as_dict() for e in self.excluded],
            "notes": self.notes,
        }


# Categorical controls in the hedonic surface that encode *when* a sale closed
# rather than *what* was sold. An inventory has no such value: the units have
# not been listed yet.
_TIME_CONTROLS: tuple[str, ...] = ("list_quarter", "list_month", "list_year")


@dataclass
class PricingFrame:
    """Inventory ready for hedonic scoring, plus what had to be pinned to get there."""

    frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def as_of_pricing_frame(
    units: pd.DataFrame, surface: HedonicSurface, *, as_of: str | None = None
) -> PricingFrame:
    """Prepare inventory for hedonic scoring, pinned to an observed period.

    The surface carries a time control, because closed prices span the export
    window and a surface without one attributes market drift to whichever
    feature happens to correlate with when things sold. That control has no
    level for a phase in 2028 — the model has never seen 2028 — so scoring a
    future phase against it would either fail or, worse, require inventing a
    level.

    So the comps band is quoted **as of the most recent period the surface was
    fitted on**: "what would this unit fetch in today's market". Projecting the
    surface forward is a forecast, and this system has not made one. Where the
    developer believes comps will move, that belief belongs in
    `Comps.monthly_drift`, where it is visible and owned.

    Returns:
        A `PricingFrame`: the frame `surface.price_bounds` can score, and notes
        naming every control that was pinned to a period.
    """
    frame = inventory_scoring_frame(units)
    notes: list[str] = []
    for name in _TIME_CONTROLS:
        if name not in surface.design.reference_levels:
            continue
        levels = [
            surface.design.reference_levels[name],
            *surface.design.categorical_levels.get(name, []),
        ]
        chosen = str(as_of) if as_of is not None else max(str(level) for level in levels)
        if chosen not in {str(level) for level in levels}:
            raise SchemaError(
                f"as_of={chosen!r} is not a period the surface was fitted on. "
                f"Known: {sorted(str(level) for level in levels)[-5:]}"
            )
        frame[name] = chosen
        notes.append(
            f"comps priced as of {name}={chosen}, the latest period in the hedonic "
            "sample. The band is today's market value, not a forecast of the phase date."
        )
    return PricingFrame(frame=frame, notes=notes)


def build_price_ladder(
    units: pd.DataFrame,
    surface: HedonicSurface,
    config: MarketConfig | None = None,
    *,
    n_levels: int | None = None,
    min_margin_over_cost: float | None = None,
    as_of: str | None = None,
) -> PriceLadder:
    """Build one ascending $/sqft ladder per unit.

    Args:
        units: validated inventory. Needs `unit_id` and `cost_basis_ppsf`, plus
            whatever hedonic covariates the fitted surface uses.
        surface: fitted hedonic surface; supplies the comps band.
        config: market config; supplies `price_ladder_levels` and
            `min_margin_over_cost`.
        n_levels: overrides the configured K.
        min_margin_over_cost: overrides the configured margin, a decimal
            fraction (`0.15` = 15% over the all-in basis).
        as_of: the time-control level to price against, e.g. `"2024Q4"`.
            Defaults to the latest period the surface was fitted on. See
            `as_of_pricing_frame` for why this is not the phase date.

    Returns:
        A `PriceLadder` covering only the units with a feasible band. Units
        without one are in `excluded`, never silently dropped.

    Raises:
        SchemaError: on a missing required column, a non-positive K, or when no
            unit at all has a feasible band.
    """
    for column in ("unit_id", "cost_basis_ppsf"):
        if column not in units.columns:
            raise SchemaError(f"Price ladder requires column {column!r}")

    defaults = (config or {}).get("defaults") or {}
    levels = int(n_levels if n_levels is not None else defaults.get("price_ladder_levels", _DEFAULT_LEVELS))
    if levels < 2:
        raise SchemaError(
            f"price_ladder_levels must be at least 2, got {levels}. A one-level "
            "ladder removes the price decision the optimizer exists to make."
        )
    margin = float(
        min_margin_over_cost
        if min_margin_over_cost is not None
        else defaults.get("min_margin_over_cost", _DEFAULT_MIN_MARGIN)
    )
    if margin < 0:
        raise SchemaError(f"min_margin_over_cost must be non-negative, got {margin}")

    priced = as_of_pricing_frame(units, surface, as_of=as_of)
    bounds = surface.price_bounds(priced.frame)
    cost_floor = pd.to_numeric(units["cost_basis_ppsf"], errors="coerce") * (1.0 + margin)

    kept_ids: list[str] = []
    kept_rows: list[np.ndarray] = []
    floors: list[float] = []
    ceilings: list[float] = []
    predicted: list[float] = []
    sources: list[str] = []
    excluded: list[ExcludedUnit] = []

    for position, index in enumerate(units.index):
        unit_id = str(units.at[index, "unit_id"])
        comps_floor = float(bounds.iat[position, bounds.columns.get_loc("p_floor_ppsf")])
        ceiling = float(bounds.iat[position, bounds.columns.get_loc("p_ceiling_ppsf")])
        centre = float(bounds.iat[position, bounds.columns.get_loc("predicted_ppsf")])
        basis_floor = float(cost_floor.iloc[position]) if pd.notna(cost_floor.iloc[position]) else np.nan

        if not np.isfinite(centre) or not np.isfinite(ceiling) or not np.isfinite(comps_floor):
            excluded.append(
                ExcludedUnit(
                    unit_id,
                    "no_comps",
                    "the hedonic surface cannot price this unit — a covariate it was "
                    "fitted on is missing here, so no comps band exists",
                )
            )
            continue
        if not np.isfinite(basis_floor):
            excluded.append(
                ExcludedUnit(unit_id, "no_cost_basis", "cost_basis_ppsf is missing or unparseable")
            )
            continue

        floor = max(comps_floor, basis_floor)
        source = FLOOR_FROM_COST if basis_floor >= comps_floor else FLOOR_FROM_COMPS
        if ceiling - floor < _MIN_BAND_WIDTH_PPSF:
            excluded.append(
                ExcludedUnit(
                    unit_id,
                    "cost_exceeds_comps",
                    f"required floor ${floor:,.0f}/sqft is at or above the comps ceiling "
                    f"${ceiling:,.0f}/sqft. At a {margin:.0%} margin over a "
                    f"${basis_floor / (1 + margin):,.0f}/sqft basis this unit does not "
                    "clear at market. Cut the basis, cut the margin, or accept a loss — "
                    "the optimizer will not price it for you.",
                )
            )
            continue

        kept_ids.append(unit_id)
        kept_rows.append(np.linspace(floor, ceiling, levels))
        floors.append(floor)
        ceilings.append(ceiling)
        predicted.append(centre)
        sources.append(source)

    if not kept_ids:
        raise SchemaError(
            f"No unit in this inventory has a feasible price band ({len(excluded)} "
            "excluded). There is no plan to produce; look at the exclusion reasons."
        )

    notes: list[str] = list(priced.notes)
    cost_bound = sum(1 for s in sources if s == FLOOR_FROM_COST)
    if cost_bound:
        notes.append(
            f"{cost_bound} of {len(kept_ids)} units have their price floor set by cost "
            f"basis plus a {margin:.0%} margin rather than by comps. Where that binds, "
            "the recommended price reflects the developer's basis, not demand."
        )
    if excluded:
        notes.append(
            f"{len(excluded)} units excluded from the plan entirely; see `excluded`. "
            "They are not priced at a default."
        )

    logger.info(
        "Price ladder: %d units x %d levels, %d cost-bound, %d excluded",
        len(kept_ids), levels, cost_bound, len(excluded),
    )
    return PriceLadder(
        unit_ids=tuple(kept_ids),
        levels_ppsf=np.vstack(kept_rows),
        p_floor_ppsf=np.asarray(floors, dtype="float64"),
        p_ceiling_ppsf=np.asarray(ceilings, dtype="float64"),
        predicted_ppsf=np.asarray(predicted, dtype="float64"),
        floor_source=tuple(sources),
        excluded=tuple(excluded),
        notes=notes,
    )


def ladder_from_bounds(
    unit_ids: list[str],
    p_floor_ppsf: np.ndarray | list[float],
    p_ceiling_ppsf: np.ndarray | list[float],
    *,
    n_levels: int = _DEFAULT_LEVELS,
) -> PriceLadder:
    """Build a ladder directly from explicit bounds.

    For tests and for callers that already hold a band. Production paths should
    use `build_price_ladder`, which derives the band from a fitted surface and
    the cost basis rather than accepting one on faith.
    """
    floors = np.asarray(p_floor_ppsf, dtype="float64")
    ceilings = np.asarray(p_ceiling_ppsf, dtype="float64")
    if not (len(unit_ids) == len(floors) == len(ceilings)):
        raise SchemaError("unit_ids, floors, and ceilings must be the same length")
    if (ceilings <= floors).any():
        raise SchemaError("every ceiling must exceed its floor")
    return PriceLadder(
        unit_ids=tuple(str(u) for u in unit_ids),
        levels_ppsf=np.vstack([np.linspace(lo, hi, n_levels) for lo, hi in zip(floors, ceilings, strict=True)]),
        p_floor_ppsf=floors,
        p_ceiling_ppsf=ceilings,
        predicted_ppsf=(floors + ceilings) / 2.0,
        floor_source=tuple(FLOOR_FROM_COMPS for _ in unit_ids),
        notes=["ladder built from caller-supplied bounds, not from a fitted surface"],
    )
