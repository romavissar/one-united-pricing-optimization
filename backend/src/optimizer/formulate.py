"""Turning a fitted demand model into a linear objective.

The MILP's whole trick lives here. `R[i,j,k] = p_k · area_i · D(p_k, X_i, M_j) · δ_j`
is evaluated for every (unit, phase, price level) *before* the solver runs, so
what CBC sees is a linear function of binary variables. The demand model is
called K·J times on the full inventory and never again.

Two things this module refuses to do:

**It does not invent covariates.** A unit whose features leave the Cox model
unable to score it gets `NaN`, and a `NaN` sale probability becomes an excluded
unit with a stated reason, not a filled-in average. A fabricated probability
here is invisible downstream: it comes out as a confident dollar figure in a
plan.

**It does not hide the independence assumption.** `D` is evaluated per unit as
if it were the only unit released. Ten near-identical units released into one
phase are each scored against the same buyer pool, so their summed revenue
double-counts. `crowding.py` estimates the size of that error; the caveat rides
along in `DemandProvenance` either way.

Units: `price_ppsf` and `cell_median_ppsf` are `$/sqft`; `revenue` is USD;
`start_month` is months from project start; `horizon_days` is days.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.config import MarketConfig
from src.data.features import floor_bucket, season_of
from src.demand.base import PremiumSupport, relative_premium
from src.exceptions import SchemaError
from src.optimizer.discretize import ExcludedUnit, PriceLadder
from src.utils.npv import discount_factors
from src.utils.validate import inventory_scoring_frame

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON_DAYS = 180
_DEFAULT_DISCOUNT_RATE = 0.12
_DEFAULT_PRESALE_LEAD_MONTHS = 24
_DAYS_PER_MONTH = 30.4375


class DemandModel(Protocol):
    """What the optimizer needs from a demand model, and nothing more.

    `CoxDemandModel` satisfies this. Keeping the surface this narrow is what
    lets the elasticity-response test swap in a model with a known `beta_price`
    and check that the plan actually moves.
    """

    kind: str

    def predict_sale_probability(
        self, features: pd.DataFrame, price_ppsf: float | np.ndarray, horizon_days: int
    ) -> np.ndarray:
        """P(sold within `horizon_days`) at `price_ppsf` $/sqft, one per row."""
        ...

    @property
    def premium_support(self) -> PremiumSupport | None:
        """Range of `rel_price_premium` the fit is evidence about, if known.

        Optional in practice: `build_revenue_tensor` reads it with `getattr` and
        records its absence as a caveat, so a model that does not report a
        support degrades to "extrapolation could not be checked" rather than to
        a silent assumption that there is none.
        """
        ...


@dataclass(frozen=True)
class Phase:
    """One release phase: when it opens and what it has to clear.

    Attributes:
        start_month: months from project start. Phase 0 is usually 0.
        cash_flow_floor_usd: minimum expected revenue this phase must produce.
            Binds on the *expected value*, so realized cash flow can still
            breach it — see `PROJECT_BRIEF.md` §5 and the caveat in the result.
        max_units: sales-team capacity, units. None means uncapped.
        min_type_counts: `unit_type` -> minimum count, applied only if the
            phase releases anything at all.
        competing_listings: forecast count of *other* listings entering this
            submarket-month. Exogenous market input, not fitted here. Required
            when the demand model was fitted with `inventory_competition`.
    """

    index: int
    name: str
    start_month: float
    cash_flow_floor_usd: float = 0.0
    max_units: int | None = None
    min_type_counts: Mapping[str, int] = field(default_factory=dict)
    competing_listings: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "start_month": self.start_month,
            "cash_flow_floor_usd": self.cash_flow_floor_usd,
            "max_units": self.max_units,
            "min_type_counts": dict(self.min_type_counts),
            "competing_listings": self.competing_listings,
        }


@dataclass(frozen=True)
class Comps:
    """The submarket-month median $/sqft each unit's price is measured against.

    This is the denominator of `rel_price_premium`, and therefore the thing
    `beta_price` multiplies. It is a **forward-looking market input**, not a
    fitted quantity: the model says how demand responds to being priced above
    comps, it does not say where comps will be in eighteen months.

    `monthly_drift` defaults to zero — flat comps — because any other default
    would be a forecast this system has not made. A caller who supplies drift
    owns it.
    """

    median_ppsf_by_submarket: Mapping[str, float]
    monthly_drift: float = 0.0

    def median_at(self, submarket: str, months: float) -> float:
        """Comp median $/sqft for a submarket `months` after project start."""
        base = self.median_ppsf_by_submarket.get(submarket)
        if base is None:
            return float("nan")
        return float(base) * (1.0 + self.monthly_drift) ** float(months)

    def require_coverage(self, submarkets: Iterable[str]) -> None:
        """Fail loudly on a submarket with no comp median.

        Without this the missing median propagates as a NaN
        `rel_price_premium`, then a NaN sale probability, and the unit is
        dropped with "the demand model could not score it" — which sends the
        user hunting through inventory columns when the actual fault is one
        absent key in this mapping.

        Raises:
            SchemaError: naming every submarket that has no median.
        """
        missing = sorted({str(s) for s in submarkets} - set(self.median_ppsf_by_submarket))
        if missing:
            raise SchemaError(
                f"No comp median $/sqft for {missing}. Every submarket in the "
                "inventory needs one: it is the denominator of rel_price_premium, "
                "so without it there is no price the demand model can respond to. "
                f"Supplied: {sorted(self.median_ppsf_by_submarket)}"
            )


@dataclass(frozen=True)
class ProjectSpec:
    """Timing, money, and market assumptions that frame the whole optimization."""

    market: str
    project_start: pd.Timestamp
    phases: tuple[Phase, ...]
    comps: Comps
    discount_rate_annual: float = _DEFAULT_DISCOUNT_RATE
    horizon_days: int = _DEFAULT_HORIZON_DAYS
    presale_lead_months: int = _DEFAULT_PRESALE_LEAD_MONTHS

    def __post_init__(self) -> None:
        if not self.phases:
            raise SchemaError("A release plan needs at least one phase")
        starts = [p.start_month for p in self.phases]
        if starts != sorted(starts):
            raise SchemaError(
                f"Phases must be given in ascending start_month order, got {starts}"
            )
        if [p.index for p in self.phases] != list(range(len(self.phases))):
            raise SchemaError("Phase indices must be 0..n-1 in order")

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def phase_dates(self) -> list[pd.Timestamp]:
        """Calendar start date of each phase."""
        return [
            self.project_start + pd.Timedelta(p.start_month * _DAYS_PER_MONTH, unit="D")
            for p in self.phases
        ]

    @classmethod
    def from_config(
        cls,
        config: MarketConfig,
        *,
        project_start: pd.Timestamp,
        phases: Sequence[Phase],
        comps: Comps,
        **overrides: Any,
    ) -> ProjectSpec:
        """Build a spec, taking timing and rate defaults from market config."""
        defaults = config.get("defaults") or {}
        return cls(
            market=str(config.get("market", "unknown")),
            project_start=pd.Timestamp(project_start),
            phases=tuple(phases),
            comps=comps,
            discount_rate_annual=float(
                overrides.get("discount_rate_annual")
                or defaults.get("discount_rate_annual", _DEFAULT_DISCOUNT_RATE)
            ),
            horizon_days=int(
                overrides.get("horizon_days") or defaults.get("horizon_days", _DEFAULT_HORIZON_DAYS)
            ),
            presale_lead_months=int(
                overrides.get("presale_lead_months")
                or defaults.get("presale_lead_months", _DEFAULT_PRESALE_LEAD_MONTHS)
            ),
        )


@dataclass
class DemandProvenance:
    """Where the demand numbers came from, in the shape the API has to emit.

    `AGENTS.md` §4: every optimize and simulate response carries this, and
    `is_calibrated_on_real_data` defaults to False. A caller who does not say
    where the model was fitted gets "synthetic", which is the safe direction to
    be wrong in.
    """

    demand_model: str = "cox_ph"
    fitted_on: str = "synthetic"
    is_calibrated_on_real_data: bool = False
    beta_price: float = float("nan")
    beta_price_se: float | None = None
    beta_price_ci95: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_bundle(cls, bundle: Any) -> DemandProvenance:
        """Read provenance off a `registry.ModelBundle`."""
        coefficient = bundle.cox_result.beta_price
        provenance = bundle.provenance
        return cls(
            demand_model=bundle.cox_result.model_kind,
            fitted_on=provenance.data_source,
            is_calibrated_on_real_data=bool(provenance.is_calibrated_on_real_data),
            beta_price=float(coefficient.value),
            beta_price_se=float(coefficient.std_error),
            beta_price_ci95=(float(coefficient.ci_low), float(coefficient.ci_high)),
            warnings=list(provenance.notes),
        )

    def as_response_block(
        self, *, independence_assumption: bool = True, crowding_correction_applied: bool = False
    ) -> dict[str, Any]:
        warnings = list(self.warnings)
        if not self.is_calibrated_on_real_data:
            warnings.insert(
                0,
                "Figures are illustrative: demand model fitted on "
                f"{self.fitted_on} data, not on a real MLS export.",
            )
        if independence_assumption and not crowding_correction_applied:
            warnings.append(
                "Sale probabilities are computed per unit independently. Phases that "
                "release many near-identical units have their revenue overstated; run "
                "the crowding correction to size the error."
            )
        return {
            "demand_model": self.demand_model,
            "fitted_on": self.fitted_on,
            "is_calibrated_on_real_data": self.is_calibrated_on_real_data,
            "beta_price": self.beta_price,
            "beta_price_se": self.beta_price_se,
            "beta_price_ci95": list(self.beta_price_ci95) if self.beta_price_ci95 else None,
            "independence_assumption": independence_assumption,
            "crowding_correction_applied": crowding_correction_applied,
            "warnings": warnings,
        }


def build_phase_features(
    units: pd.DataFrame, phase: Phase, spec: ProjectSpec
) -> pd.DataFrame:
    """Feature frame for every unit as it would look listed in `phase`.

    Derives the time-varying covariates from the phase's calendar position and
    carries the unit's own characteristics through unchanged. `cell_median_ppsf`
    comes from `spec.comps`; `predict_sale_probability` divides the candidate
    price by it to get `rel_price_premium`.

    Columns absent from the inventory stay absent. The design transform will
    mark those rows unusable, which is the intended behaviour: a unit the model
    cannot score should not get a price.
    """
    date = spec.project_start + pd.Timedelta(phase.start_month * _DAYS_PER_MONTH, unit="D")
    out = inventory_scoring_frame(units)

    out["list_date"] = date
    out["list_month"] = str(date.to_period("M"))
    out["list_quarter"] = str(date.to_period("Q"))
    out["season"] = season_of(date)
    out["floor_bucket"] = pd.to_numeric(out["floor"], errors="coerce").map(floor_bucket)

    out["inventory_competition"] = (
        np.nan if phase.competing_listings is None else float(phase.competing_listings)
    )
    out["cell_median_ppsf"] = [
        spec.comps.median_at(str(s), phase.start_month) for s in out["submarket"]
    ]
    return out


@dataclass
class RevenueTensor:
    """Precomputed `R[i,j,k]`, plus everything needed to read it back.

    Attributes:
        price_ppsf: (units, levels) $/sqft.
        probability: (units, phases, levels) P(sale within the horizon).
        expected_revenue_usd: (units, phases, levels) undiscounted USD.
        discounted_usd: the same, times the phase discount factor. This is
            `R[i,j,k]`, the objective coefficient.
        releasable: (units, phases) bool — the construction gate.
        rel_price_premium: (units, phases, levels) the price each cell implies
            relative to its comps. Kept so the plan can be checked against the
            range `beta_price` was estimated over.
        premium_support: that range, or None when the model does not report one.
    """

    unit_ids: tuple[str, ...]
    phases: tuple[Phase, ...]
    price_ppsf: np.ndarray
    area_sqft: np.ndarray
    probability: np.ndarray
    expected_revenue_usd: np.ndarray
    discounted_usd: np.ndarray
    releasable: np.ndarray
    discount_factor: np.ndarray
    rel_price_premium: np.ndarray
    release_discounted_usd: np.ndarray | None = None
    arrival_discount_factor: np.ndarray | None = None
    premium_support: PremiumSupport | None = None
    horizon_days: int = _DEFAULT_HORIZON_DAYS
    excluded: tuple[ExcludedUnit, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.discounted_usd.shape)  # type: ignore[return-value]

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    @property
    def n_levels(self) -> int:
        return int(self.price_ppsf.shape[1])


def build_revenue_tensor(
    units: pd.DataFrame,
    ladder: PriceLadder,
    model: DemandModel,
    spec: ProjectSpec,
) -> RevenueTensor:
    """Evaluate `R[i,j,k]` for every unit, phase, and price level.

    Args:
        units: validated inventory. Must contain `unit_id`, `living_area_sqft`,
            `submarket`, `completion_date`, plus the model's covariates.
        ladder: per-unit price ladders; sets which units are in play.
        model: anything satisfying `DemandModel`.
        spec: timing, discounting, and comps.

    Returns:
        The tensor, restricted to units the demand model could score. Units it
        could not are in `excluded`.

    Raises:
        SchemaError: on missing inventory columns, a submarket with no comp
            median, or when no unit survives.
    """
    for column in ("unit_id", "living_area_sqft", "submarket", "completion_date"):
        if column not in units.columns:
            raise SchemaError(f"Revenue tensor requires inventory column {column!r}")

    frame = units.copy()
    frame["unit_id"] = frame["unit_id"].astype(str)
    ordered = frame.set_index("unit_id").reindex(list(ladder.unit_ids))
    if ordered["living_area_sqft"].isna().any():
        missing = ordered.index[ordered["living_area_sqft"].isna()].tolist()
        raise SchemaError(f"Units in the ladder are absent from the inventory: {missing}")
    ordered = ordered.reset_index()
    spec.comps.require_coverage(ordered["submarket"])

    n_units, n_levels = ladder.levels_ppsf.shape
    n_phases = spec.n_phases
    area = pd.to_numeric(ordered["living_area_sqft"], errors="coerce").to_numpy(dtype="float64")

    probability = np.full((n_units, n_phases, n_levels), np.nan, dtype="float64")
    premium = np.full((n_units, n_phases, n_levels), np.nan, dtype="float64")
    premium_model = getattr(model, "premium_model", None)
    notes: list[str] = list(ladder.notes)

    # Clamp the ladder to the range the coefficient is evidence about, before
    # any probability is computed. A Cox linear predictor extrapolates without
    # limit: asked about a price twice anything in the fitting sample it returns
    # a probability with the same units and decimal places as a real one. The
    # optimizer, maximising `p · D(p)`, will find exactly those cells if the
    # ladder reaches them, so reporting the extrapolation after the solve is too
    # late — the plan has already been chosen on it.
    #
    # The band is the 1st-99th percentile of the fitted premium, not its min and
    # max: a single listing priced at twenty times its hedonic reference should
    # not license the optimizer to price there.
    support = getattr(model, "premium_support", None)
    ladder_levels = ladder.levels_ppsf
    clamped_cells = 0
    support_excluded: list[ExcludedUnit] = []
    if support is not None and premium_model is not None:
        references = np.column_stack([
            premium_model.predict_reference_ppsf(
                build_phase_features(ordered, phase, spec)
            )
            for phase in spec.phases
        ])
        with np.errstate(invalid="ignore"):
            reference = np.nanmean(references, axis=1)
        support_low = reference * (1.0 + support.p1)
        support_high = reference * (1.0 + support.p99)
        # Intersect with the band the ladder already enforces. That band's floor
        # is the developer's cost basis plus margin — a business rule, not an
        # estimate — so it wins over the statistical band: clamping a price
        # below it would recommend selling under basis in order to stay inside
        # the evidence. Where the two bands do not overlap at all, the unit
        # cannot be priced both above its basis and inside the range the
        # coefficient is evidence about, and it is excluded with that stated
        # rather than priced on one side of the contradiction.
        floor = np.maximum(ladder.p_floor_ppsf, support_low)
        ceiling = np.minimum(ladder.p_ceiling_ppsf, support_high)
        overlaps = np.isfinite(floor) & np.isfinite(ceiling) & (ceiling > floor)
        if overlaps.any():
            proposed = np.clip(
                ladder_levels,
                np.where(overlaps, floor, -np.inf)[:, None],
                np.where(overlaps, ceiling, np.inf)[:, None],
            )
            clamped_cells = int((np.abs(proposed - ladder_levels) > 1e-9).sum())
            ladder_levels = proposed
        unsupported = np.isfinite(support_low) & np.isfinite(support_high) & ~overlaps
        if unsupported.all():
            # Every unit fails: that is not sixty separate problems, it is one
            # problem with the two reference surfaces. The price band comes from
            # the close-price hedonic, which uses only covariates a developer's
            # inventory actually carries; the premium reference comes from the
            # list-price hedonic, which is fitted on MLS listings and whose
            # amenity vocabulary — view, waterfront, parking, restrictions — the
            # inventory does not have. Scoring an inventory unit therefore holds
            # most of that surface at its fit-time mean and the prediction
            # regresses toward the middle of the MLS sample, which for this
            # inventory sits about a third below the comps band.
            #
            # Excluding the whole inventory would be a wrong answer stated
            # confidently. The clamp stands down, the plan is produced against
            # the comps band, and the caveat says the extrapolation guard could
            # not be applied and why.
            clamped_cells = 0
            support_excluded = []
            notes.append(
                "PREMIUM REFERENCE NOT COMPARABLE TO THE COMPS BAND — the ladder "
                f"spans ${np.nanmedian(ladder.p_floor_ppsf):,.0f}-"
                f"${np.nanmedian(ladder.p_ceiling_ppsf):,.0f}/sqft at the median unit "
                f"while the range beta_price is evidence about maps to "
                f"${np.nanmedian(support_low):,.0f}-${np.nanmedian(support_high):,.0f}"
                "/sqft, and the two do not overlap for any unit. The price band is "
                "built from the close-price hedonic on covariates the inventory "
                "carries; the premium reference is the list-price hedonic, most of "
                "whose amenity controls the inventory does not carry and which "
                "therefore falls back to fit-time means. The ladder was NOT clamped "
                "and the recommended prices are NOT verified against the fitted "
                "support. Treat every sale probability here as an extrapolation "
                "until the inventory schema carries the hedonic's covariates."
            )
            logger.warning(notes[-1])
        elif unsupported.any():
            support_excluded = [
                ExcludedUnit(
                    ladder.unit_ids[i],
                    "outside_fitted_support",
                    f"this unit's price band (${ladder.p_floor_ppsf[i]:,.0f}-"
                    f"${ladder.p_ceiling_ppsf[i]:,.0f}/sqft) does not overlap the range "
                    f"beta_price is evidence about (${support_low[i]:,.0f}-"
                    f"${support_high[i]:,.0f}/sqft). Pricing it would mean either "
                    "selling below the cost basis or extrapolating the demand model, "
                    "so it is left unpriced rather than guessed.",
                )
                for i in np.flatnonzero(unsupported)
            ]

    for j, phase in enumerate(spec.phases):
        features = build_phase_features(ordered, phase, spec)
        if premium_model is not None:
            # `beta_price` is now the coefficient on a residual against the
            # list-price hedonic, so the reference a candidate price is measured
            # against has to be that same surface's prediction for this unit —
            # not the submarket median. Using the median here would leave the
            # coefficient multiplying a quantity it was never fitted on, which
            # is a units error that produces entirely plausible numbers.
            features["cell_median_ppsf"] = premium_model.predict_reference_ppsf(features)
        median = features["cell_median_ppsf"].to_numpy(dtype="float64")
        for k in range(n_levels):
            probability[:, j, k] = model.predict_sale_probability(
                features, ladder_levels[:, k], spec.horizon_days
            )
            premium[:, j, k] = relative_premium(ladder_levels[:, k], median)

    expected = ladder_levels[:, None, :] * area[:, None, None] * probability
    factors = discount_factors(
        [p.start_month for p in spec.phases], spec.discount_rate_annual
    )
    release_discounted = expected * factors[None, :, None]

    # Discount to when the cash arrives, not to when the phase opens.
    arrival_factor = None
    followup = getattr(model, "max_observed_duration_days", None)
    horizon_cap = float(spec.horizon_days)
    if followup is not None:
        horizon_cap = min(horizon_cap, float(followup))
    if hasattr(model, "baseline_survival") and horizon_cap > 0:
        try:
            grid = np.linspace(0.0, horizon_cap, _ARRIVAL_GRID_POINTS)
            arrival_factor = arrival_discount_factors(
                probability=probability,
                baseline_survival=model.baseline_survival(grid),
                times_days=grid,
                phase_start_months=np.array([p.start_month for p in spec.phases]),
                annual_rate=spec.discount_rate_annual,
            )
        except Exception as exc:  # noqa: BLE001 - degrade loudly, never silently
            logger.warning(
                "Arrival-time discounting unavailable (%s); falling back to "
                "discounting at the phase release date, which overstates "
                "present value.", exc,
            )
            arrival_factor = None

    if arrival_factor is not None:
        discounted = expected * arrival_factor
    else:
        discounted = release_discounted

    scorable = np.isfinite(probability).all(axis=(1, 2))
    if support_excluded:
        unsupported_ids = {e.unit_id for e in support_excluded}
        scorable = scorable & np.array(
            [uid not in unsupported_ids for uid in ladder.unit_ids]
        )
    excluded = [
        ExcludedUnit(
            ladder.unit_ids[i],
            "unscorable",
            "the demand model could not produce a sale probability for this unit — a "
            "covariate it was fitted on is missing here. Priced at nothing rather than "
            "at a guess.",
        )
        for i in np.flatnonzero(~scorable)
    ]
    if not scorable.any():
        raise SchemaError(
            "The demand model could not score a single unit in this inventory. Check "
            "that the inventory carries the covariates the model was fitted on."
        )

    keep = np.flatnonzero(scorable)
    kept_ids = tuple(ladder.unit_ids[i] for i in keep)
    releasable = _construction_gate(ordered.iloc[keep], spec)

    support = getattr(model, "premium_support", None)

    # A non-parametric baseline hazard is only identified over the durations the
    # fit actually saw. lifelines holds the last value past the final event
    # time, so asking for a horizon beyond it returns the probability at that
    # final time — silently, with the same type and precision as a real answer.
    # `horizon_days` is caller-settable through the API, so this is reachable.
    followup = getattr(model, "max_observed_duration_days", None)
    if followup is not None and spec.horizon_days > followup:
        notes.append(
            f"HORIZON BEYOND FOLLOW-UP — sale probabilities are requested at "
            f"{spec.horizon_days} days but the demand model was fitted on durations "
            f"reaching only {followup:.0f} days. Past its last observed event the "
            "baseline hazard is flat, so every probability here is the "
            f"{followup:.0f}-day probability wearing a longer label. Shorten the "
            "horizon to the observed follow-up, or refit on data that covers it."
        )
        logger.warning(
            "horizon_days=%d exceeds the fitted follow-up of %.0f days; the "
            "baseline hazard is flat beyond it",
            spec.horizon_days, followup,
        )

    if support is None:
        notes.append(
            f"the {model.kind} model does not report the rel_price_premium range it "
            "was fitted over, so the plan cannot be checked for extrapolation. Prices "
            "outside that range are guesses, and this one will not tell you."
        )
    if excluded:
        notes.append(f"{len(excluded)} units excluded because the demand model could not score them")
    if arrival_factor is not None:
        release_total = float(release_discounted[keep].sum())
        arrival_total = float(discounted[keep].sum())
        if release_total > 0:
            notes.append(
                f"revenue is discounted to expected sale time, not to the phase "
                f"release date: across the whole grid that is "
                f"{arrival_total / release_total - 1:+.2%} against the release-date "
                f"basis (horizon {spec.horizon_days}d, capped at the fitted "
                f"follow-up of {horizon_cap:.0f}d)"
            )
    if clamped_cells:
        notes.append(
            f"the price ladder was clamped to the 1st-99th percentile of the "
            f"fitted premium support ([{support.p1:+.3f}, {support.p99:+.3f}]): "
            f"{clamped_cells} of {ladder.levels_ppsf.size} grid points sat outside "
            "the range beta_price is evidence about and would have been "
            "extrapolation had the optimizer chosen them"
        )
    blocked = int((~releasable).sum())
    if blocked:
        notes.append(
            f"the construction gate blocks {blocked} of {len(kept_ids) * n_phases} "
            f"unit-phase cells (presale lead {spec.presale_lead_months} months)"
        )

    logger.info(
        "Revenue tensor %dx%dx%d built; %d units unscorable, %d cells gated",
        len(kept_ids), n_phases, n_levels, len(excluded), blocked,
    )
    return RevenueTensor(
        unit_ids=kept_ids,
        phases=spec.phases,
        price_ppsf=ladder_levels[keep],
        area_sqft=area[keep],
        probability=probability[keep],
        expected_revenue_usd=expected[keep],
        discounted_usd=discounted[keep],
        releasable=releasable,
        discount_factor=factors,
        rel_price_premium=premium[keep],
        release_discounted_usd=release_discounted[keep],
        arrival_discount_factor=(
            arrival_factor[keep] if arrival_factor is not None else None
        ),
        premium_support=support,
        horizon_days=spec.horizon_days,
        excluded=(*ladder.excluded, *support_excluded, *excluded),
        notes=notes,
    )


_ARRIVAL_GRID_POINTS = 60


def arrival_discount_factors(
    probability: np.ndarray,
    baseline_survival: np.ndarray,
    times_days: np.ndarray,
    phase_start_months: np.ndarray,
    annual_rate: float,
) -> np.ndarray:
    """`E[δ | the unit sells within the horizon]`, per (unit, phase, level).

    The objective used to discount a unit's whole expected revenue to its
    phase's *release* date, but `D` is the probability of selling at some point
    inside a horizon `T`, and the cash arrives when the sale happens. Everything
    released was therefore being valued as if it settled on day one of its
    phase, which under-penalises late sales and biases the phase ordering.

    A midpoint approximation is not good enough and is not used. Time-to-sale is
    right-skewed — most sales that happen at all happen early — so the mean
    arrival time sits well below `T/2`, and the skew *itself* varies with price:
    a unit priced above its hedonic reference has a lower hazard, so its
    conditional sale time shifts later. That price dependence is precisely the
    thing this correction exists to capture, and a fixed midpoint throws it
    away by construction.

    Instead the discount factor is integrated against the conditional
    sale-time distribution. Under proportional hazards each unit's survival
    curve is the baseline raised to a power, and that power is recoverable from
    the horizon probability already computed:

        S(T) = S0(T) ** r   ⇒   r = log(1 - D) / log(S0(T))

    so one baseline vector reconstructs every cell's whole curve, and

        E[δ | sale ≤ T] = Σ_m δ(t_j + τ_m) · ΔF_m(r) / F(T, r)

    The result is a constant per (unit, phase, level), so the objective stays
    linear and the MILP is unchanged.

    Args:
        probability: (units, phases, levels) P(sale within the horizon).
        baseline_survival: S0 at each of `times_days`, non-increasing.
        times_days: integration grid, days from release, capped at the fitted
            follow-up by the caller.
        phase_start_months: months from project start, one per phase.
        annual_rate: annual decimal discount rate.

    Returns:
        (units, phases, levels) expected discount factor, in [0, 1].
    """
    s0 = np.clip(np.asarray(baseline_survival, dtype="float64"), 1e-12, 1.0)
    grid = np.asarray(times_days, dtype="float64")
    log_s0 = np.log(s0)
    s0_at_horizon = s0[-1]

    # Recover each cell's proportional-hazards multiplier from its horizon
    # probability. A cell with D = 0 carries no sale-time distribution at all;
    # its discount factor is irrelevant because its revenue is zero, so it is
    # given the release-date factor to keep the array finite.
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.log(np.clip(1.0 - probability, 1e-12, 1.0)) / np.log(
            max(s0_at_horizon, 1e-12)
        )
    power = np.where(np.isfinite(power) & (power > 0), power, 0.0)

    # Survival on the grid for every cell: S(τ) = S0(τ) ** r.
    survival = np.exp(power[..., None] * log_s0)
    increments = -np.diff(survival, axis=-1)          # ΔF over each grid step
    total = np.clip(1.0 - survival[..., -1], 1e-12, None)

    months = grid / _DAYS_PER_MONTH
    factors = np.empty_like(probability, dtype="float64")
    for j, start in enumerate(np.asarray(phase_start_months, dtype="float64")):
        # Midpoint of each step, in months from project start.
        arrival_months = start + 0.5 * (months[:-1] + months[1:])
        deltas = discount_factors(arrival_months, annual_rate)
        weighted = (increments[:, j, :, :] * deltas[None, None, :]).sum(axis=-1)
        factors[:, j, :] = weighted / total[:, j, :]
    # A cell with no sale mass falls back to the release-date factor.
    release = discount_factors(np.asarray(phase_start_months, dtype="float64"), annual_rate)
    empty = ~np.isfinite(factors) | (probability <= 0)
    factors = np.where(empty, np.broadcast_to(release[None, :, None], factors.shape), factors)
    return np.clip(factors, 0.0, 1.0)


def _construction_gate(units: pd.DataFrame, spec: ProjectSpec) -> np.ndarray:
    """(units, phases) mask: True where the phase opens on or after presale start.

    A unit becomes sellable `presale_lead_months` before its building completes.
    A phase starting earlier than that cannot release it — there is nothing to
    sell yet, however good the price would be.
    """
    completion = pd.to_datetime(units["completion_date"], errors="coerce")
    lead = pd.Timedelta(spec.presale_lead_months * _DAYS_PER_MONTH, unit="D")
    sellable_from = completion - lead
    phase_dates = spec.phase_dates()
    gate = np.zeros((len(units), spec.n_phases), dtype=bool)
    for j, date in enumerate(phase_dates):
        gate[:, j] = (sellable_from.notna() & (date >= sellable_from)).to_numpy()
    return gate
