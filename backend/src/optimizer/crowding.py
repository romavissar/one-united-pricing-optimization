"""Optional post-solve correction for units that compete with each other.

The objective treats each unit's sale probability as independent of everything
else released alongside it. That is false in the direction that flatters the
plan: release twelve near-identical two-bedrooms into one phase and the model
scores each as though it had the whole buyer pool to itself, so the phase's
expected revenue is the sum of twelve probabilities that cannot all be realized.

The correction is deliberately crude:

    crowding_factor[phase, cluster] = 1 - lambda * (units_released / buyer_pool)

`lambda` is a configured severity in `[0, 1]`, default `0.3`. A cluster is
`unit_type x floor_bucket` — the same comparability rule the monotone price path
uses, on the same reasoning that a studio and a penthouse are not substitutes.

Two things this is not. It is **not fitted**: `lambda` and `buyer_pool` are
assumptions the user supplies, and the correction is reported as a separate,
labelled figure rather than folded into the headline number. And it is **not a
fix**: properly, sale probabilities within a cluster should be jointly modelled
against a finite pool of buyers. This scales the answer down by a plausible
amount so the independence error has a size attached to it instead of being a
sentence in a caveat list. `PROJECT_BRIEF.md` §4 asks for exactly that and no
more.

Units: revenue USD, `buyer_pool` a count of buyers per phase.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import MarketConfig
from src.exceptions import SchemaError
from src.optimizer.constraints import comparable_groups
from src.optimizer.solve import OptimizeResult, PhaseSummary, PlanRow

logger = logging.getLogger(__name__)

_DEFAULT_LAMBDA = 0.30
# A factor cannot go below this: a cluster that swamps its pool sells slower,
# it does not sell nothing. Zero would claim a certainty the formula has not
# earned.
_MIN_FACTOR = 0.10


@dataclass(frozen=True)
class ClusterCrowding:
    """One cluster in one phase: how many were released and what it costs."""

    phase_index: int
    phase_name: str
    cluster: str
    units_released: int
    buyer_pool: float
    factor: float
    uncorrected_revenue_usd: float
    corrected_revenue_usd: float

    @property
    def revenue_removed_usd(self) -> float:
        return self.uncorrected_revenue_usd - self.corrected_revenue_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase_index": self.phase_index,
            "phase_name": self.phase_name,
            "cluster": self.cluster,
            "units_released": self.units_released,
            "buyer_pool": self.buyer_pool,
            "factor": self.factor,
            "uncorrected_revenue_usd": self.uncorrected_revenue_usd,
            "corrected_revenue_usd": self.corrected_revenue_usd,
            "revenue_removed_usd": self.revenue_removed_usd,
        }


@dataclass
class CrowdingCorrection:
    """Both figures, labelled, plus the arithmetic that connects them."""

    lambda_: float
    buyer_pool: float
    clusters: list[ClusterCrowding]
    uncorrected_revenue_usd: float
    corrected_revenue_usd: float
    uncorrected_discounted_usd: float
    corrected_discounted_usd: float
    adjusted_plan: list[PlanRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def revenue_removed_usd(self) -> float:
        return self.uncorrected_revenue_usd - self.corrected_revenue_usd

    @property
    def share_removed(self) -> float:
        """Fraction of expected revenue the correction removes; 0 if there is none."""
        if self.uncorrected_revenue_usd <= 0:
            return 0.0
        return self.revenue_removed_usd / self.uncorrected_revenue_usd

    @property
    def worst_cluster(self) -> ClusterCrowding | None:
        """The cluster losing the most revenue — the phase to re-stage first."""
        crowded = [c for c in self.clusters if c.factor < 1.0]
        return max(crowded, key=lambda c: c.revenue_removed_usd) if crowded else None

    def as_dict(self) -> dict[str, Any]:
        worst = self.worst_cluster
        return {
            "lambda": self.lambda_,
            "buyer_pool": self.buyer_pool,
            "uncorrected_revenue_usd": self.uncorrected_revenue_usd,
            "corrected_revenue_usd": self.corrected_revenue_usd,
            "uncorrected_discounted_usd": self.uncorrected_discounted_usd,
            "corrected_discounted_usd": self.corrected_discounted_usd,
            "revenue_removed_usd": self.revenue_removed_usd,
            "share_removed": self.share_removed,
            "worst_cluster": worst.as_dict() if worst else None,
            "clusters": [c.as_dict() for c in self.clusters],
            "notes": self.notes,
        }


def crowding_factor(units_released: int, buyer_pool: float, lambda_: float) -> float:
    """`1 - lambda * released / pool`, floored at `_MIN_FACTOR`.

    Args:
        units_released: units of one cluster released into one phase.
        buyer_pool: buyers plausibly shopping that cluster in that phase.
        lambda_: severity in [0, 1]. Zero disables the correction entirely.

    Raises:
        SchemaError: on a non-positive pool or a lambda outside [0, 1].
    """
    if buyer_pool <= 0:
        raise SchemaError(
            f"buyer_pool must be positive, got {buyer_pool}. With no buyers there is "
            "no plan to correct."
        )
    if not 0.0 <= lambda_ <= 1.0:
        raise SchemaError(f"crowding lambda must be in [0, 1], got {lambda_}")
    return max(_MIN_FACTOR, 1.0 - lambda_ * (units_released / float(buyer_pool)))


def apply_crowding(
    result: OptimizeResult,
    units: pd.DataFrame,
    *,
    buyer_pool: float,
    config: MarketConfig | None = None,
    lambda_: float | None = None,
    buyer_pool_by_phase: Mapping[int, float] | None = None,
) -> CrowdingCorrection:
    """Scale a solved plan's expected revenue for within-cluster competition.

    Args:
        result: a solved plan. An empty plan yields a zero correction rather
            than an error.
        units: validated inventory, for cluster membership.
        buyer_pool: buyers per phase, applied to every cluster unless
            `buyer_pool_by_phase` overrides it. Supplied by the user; nothing
            in this system estimates it.
        config: market config; supplies `crowding_lambda`.
        lambda_: overrides the configured severity.
        buyer_pool_by_phase: per-phase pool sizes, keyed by phase index.

    Returns:
        A `CrowdingCorrection` holding both the corrected and uncorrected
        figures. The caller decides which to show; both are labelled so it
        cannot be shown ambiguously.
    """
    severity = float(
        lambda_
        if lambda_ is not None
        else ((config or {}).get("defaults") or {}).get("crowding_lambda", _DEFAULT_LAMBDA)
    )
    if not 0.0 <= severity <= 1.0:
        raise SchemaError(f"crowding lambda must be in [0, 1], got {severity}")

    rows = list(result.plan)
    uncorrected = sum(r.expected_revenue_usd for r in rows)
    uncorrected_discounted = sum(r.discounted_expected_revenue_usd for r in rows)
    if not rows:
        return CrowdingCorrection(
            lambda_=severity,
            buyer_pool=float(buyer_pool),
            clusters=[],
            uncorrected_revenue_usd=0.0,
            corrected_revenue_usd=0.0,
            uncorrected_discounted_usd=0.0,
            corrected_discounted_usd=0.0,
            notes=["No units released, so there is nothing to crowd."],
        )

    groups = comparable_groups(units, tuple(r.unit_id for r in rows))
    cluster_of = {position: name for name, members in groups.items() for position in members}

    cells: dict[tuple[int, str], list[int]] = {}
    for position, row in enumerate(rows):
        cells.setdefault((row.phase_index, cluster_of[position]), []).append(position)

    clusters: list[ClusterCrowding] = []
    factor_of: dict[int, float] = {}
    for (phase_index, cluster), positions in sorted(cells.items()):
        pool = float((buyer_pool_by_phase or {}).get(phase_index, buyer_pool))
        factor = crowding_factor(len(positions), pool, severity)
        raw = sum(rows[p].expected_revenue_usd for p in positions)
        for position in positions:
            factor_of[position] = factor
        clusters.append(
            ClusterCrowding(
                phase_index=phase_index,
                phase_name=rows[positions[0]].phase_name,
                cluster=cluster,
                units_released=len(positions),
                buyer_pool=pool,
                factor=factor,
                uncorrected_revenue_usd=raw,
                corrected_revenue_usd=raw * factor,
            )
        )

    adjusted = [
        PlanRow(
            unit_id=row.unit_id,
            phase_index=row.phase_index,
            phase_name=row.phase_name,
            level_index=row.level_index,
            price_ppsf=row.price_ppsf,
            total_price_usd=row.total_price_usd,
            sale_probability=row.sale_probability * factor_of[position],
            expected_revenue_usd=row.expected_revenue_usd * factor_of[position],
            discounted_expected_revenue_usd=(
                row.discounted_expected_revenue_usd * factor_of[position]
            ),
        )
        for position, row in enumerate(rows)
    ]

    notes = [
        "Crowding is a post-solve scaling, not a re-optimization. The plan was chosen "
        "under the independence assumption; correcting the revenue afterwards does not "
        "re-stage the phases that caused the crowding.",
        f"lambda={severity:.2f} and buyer_pool={buyer_pool:g} are user assumptions. "
        "Neither is estimated from data.",
    ]
    correction = CrowdingCorrection(
        lambda_=severity,
        buyer_pool=float(buyer_pool),
        clusters=clusters,
        uncorrected_revenue_usd=uncorrected,
        corrected_revenue_usd=sum(c.corrected_revenue_usd for c in clusters),
        uncorrected_discounted_usd=uncorrected_discounted,
        corrected_discounted_usd=sum(r.discounted_expected_revenue_usd for r in adjusted),
        adjusted_plan=adjusted,
        notes=notes,
    )
    logger.info(
        "Crowding correction removes $%.0f (%.1f%%) of expected revenue at lambda=%.2f",
        correction.revenue_removed_usd, 100 * correction.share_removed, severity,
    )
    return correction


def _recount_phase(summary: PhaseSummary, rows: list[PlanRow], basis: str) -> PhaseSummary:
    """Re-roll one phase summary against corrected rows, keeping its floor."""
    members = [r for r in rows if r.phase_index == summary.index]
    expected = sum(r.expected_revenue_usd for r in members)
    discounted = sum(r.discounted_expected_revenue_usd for r in members)
    measured = discounted if basis == "discounted" else expected
    return PhaseSummary(
        index=summary.index,
        name=summary.name,
        start_month=summary.start_month,
        units_released=len(members),
        expected_revenue_usd=expected,
        discounted_expected_revenue_usd=discounted,
        cash_flow_floor_usd=summary.cash_flow_floor_usd,
        floor_met=measured >= summary.cash_flow_floor_usd - 0.01,
        mean_price_ppsf=summary.mean_price_ppsf,
    )


def crowded_result(result: OptimizeResult, correction: CrowdingCorrection) -> OptimizeResult:
    """A copy of `result` reporting corrected revenue, labelled as such.

    The provenance block flips `crowding_correction_applied` to True, which is
    what stops a corrected figure and an uncorrected one from being compared as
    though they were the same quantity.
    """
    provenance = dict(result.provenance)
    provenance["crowding_correction_applied"] = True
    provenance["independence_assumption"] = False
    warnings = list(provenance.get("warnings", []))
    warnings.append(
        f"Revenue is crowding-corrected at lambda={correction.lambda_:.2f} against a "
        f"buyer pool of {correction.buyer_pool:g} per phase. Uncorrected expected "
        f"revenue was ${correction.uncorrected_revenue_usd:,.0f}; the correction "
        f"removes {correction.share_removed:.1%}."
    )

    basis = result.constraints.get("cash_flow_basis", "nominal")
    per_phase = [
        _recount_phase(summary, correction.adjusted_plan, basis) for summary in result.per_phase
    ]
    breached = [p.name for p in per_phase if not p.floor_met]
    if breached:
        warnings.append(
            f"After the crowding correction these phases no longer clear their "
            f"cash-flow floor: {breached}. The plan was optimized without the "
            "correction, so the constraint was satisfied on the uncorrected numbers."
        )
    provenance["warnings"] = warnings

    return OptimizeResult(
        status=result.status,
        plan=correction.adjusted_plan,
        objective_usd=correction.corrected_discounted_usd,
        expected_revenue_usd=correction.corrected_revenue_usd,
        per_phase=per_phase,
        unreleased_unit_ids=list(result.unreleased_unit_ids),
        excluded_units=list(result.excluded_units),
        provenance=provenance,
        constraints=dict(result.constraints),
        caveats=[*result.caveats, *correction.notes],
        solve_seconds=result.solve_seconds,
        extrapolation=result.extrapolation,
        infeasibility=result.infeasibility,
    )
