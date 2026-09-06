"""Cox proportional-hazards model of time-to-sale — the primary estimator.

The quantity of interest is the coefficient on `rel_price_premium`: the change
in the log hazard of sale per one unit of relative price, where one unit means
"priced 100% above the submarket-month median". A listing priced 10% above
comps carries `rel_price_premium = 0.10`, so at `beta_price = -1.6` its sale
hazard is `exp(-0.16) ≈ 0.85` times the comparable listing at the median.

Censoring is the reason this model is primary rather than the logistic one.
`EXPIRED`, `WITHDRAWN`, and `CANCELED` listings were observed for their duration
and did not sell; that is right-censoring, information about how long the
listing survived, not a missing outcome. Throwing them away or scoring them as
failures both bias the estimate.

Units:
- `duration_days`: days. `horizon_days`: days.
- coefficients: per unit of covariate on the log-hazard scale.
- `price_ppsf` in prediction: $/sqft, never a total price.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

from src.data.features import DEMAND_COVARIATES
from src.demand.base import (
    DEFAULT_CATEGORICALS,
    MIN_LEVEL_COUNT,
    Coefficient,
    DesignMatrix,
    FitResult,
    PremiumSupport,
    build_design_matrix,
    premium_support,
    relative_premium,
    transform_to_design,
    unscale_coefficients,
)
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

DURATION_COL = "duration_days"
EVENT_COL = "event_sold"

# Absorbs the part of unobserved quality that is constant within a tower —
# brand, amenities, construction, position. Without it, rel_price_premium
# carries building quality as well as the seller's pricing choice, and its
# coefficient is attenuated toward zero by classical errors-in-variables.
BUILDING_FE_COLUMN = "building_name"

# Share of fixed-effect levels pooled into `__rare__` above which the control is
# not doing what its name says. On the Miami export 2,136 of 2,596 towers hold
# fewer than five listings, so 82% collapse into one bucket that absorbs nothing
# building-specific. Reporting thresholds, not estimates.
_FE_POOLED_SHARE_ALARM = 0.50
# Listings per retained level below which incidental-parameters bias is material.
_FE_MIN_ROWS_PER_LEVEL = 40.0

# The brief's covariate list plus the two hedonic controls that measurably
# reduce attenuation on synthetic data: log area (the surface is multiplicative
# in area, so raw sqft leaves curvature in the residual) and view, where the
# export carries it. Whatever these do not explain stays inside
# rel_price_premium as unobserved quality and biases beta_price toward zero.
CONTROLLED_COVARIATES: tuple[str, ...] = tuple(
    [c for c in DEMAND_COVARIATES if c != "living_area_sqft"] + ["log_living_area"]
)
CONTROLLED_CATEGORICALS: tuple[str, ...] = DEFAULT_CATEGORICALS

# Multi-valued MLS fields reach the model as indicator blocks built in
# `features.py`, never as raw strings. `Unit View` reads
# "Bay, Skyline View, Water View" — three facts about the unit, not one
# categorical level. Treated as a category it has 1,139 distinct values on the
# 2023-2026 pull and consumed 420 of 443 design columns while pooling 694
# levels into `__rare__`; split into atoms it has 17, of which `Direct Ocean`
# is a materially different and more valuable thing from `Ocean View`. Same
# rule the list-price hedonic already follows.
# Only `view_`. The waterfront, parking and restriction blocks stay in the
# list-price hedonic — which explains what sellers ask and is fitted on MLS
# listings — but must not enter the demand model, because a developer's
# inventory carries no such fields and every unit would become unscorable.
# That asymmetry is the same reference-comparability problem tracked as R2 in
# `PLAN_FIX.md`; narrowing here keeps the demand model scorable against the
# schema `inventory_scoring_frame` can actually produce.
TOKEN_COVARIATE_PREFIXES: tuple[str, ...] = ("view_",)


def token_covariates(
    frame: pd.DataFrame, prefixes: Sequence[str] = TOKEN_COVARIATE_PREFIXES
) -> tuple[str, ...]:
    """Indicator columns produced by `tokenize_multivalue`, in a stable order.

    Selected by prefix rather than named in a constant because which tokens
    clear the frequency threshold depends on the export.
    """
    return tuple(
        sorted(
            c
            for c in frame.columns
            if any(c.startswith(p) for p in prefixes)
            and pd.api.types.is_numeric_dtype(frame[c])
            and frame[c].notna().any()
        )
    )


def _premium_is_residual(frame: pd.DataFrame) -> bool:
    """True when the frame's premium is a hedonic residual, not a cell-median ratio."""
    column = frame.get("rel_price_premium_spec")
    if column is None:
        return False
    return bool((column.astype(str) == "hedonic_residual").any())


def available_covariates(
    frame: pd.DataFrame,
    covariates: Sequence[str] = CONTROLLED_COVARIATES,
    *,
    include_tokens: bool = True,
) -> tuple[str, ...]:
    """Filter a covariate list to the columns this export actually carries.

    An all-null column counts as absent. Normalization fills every canonical
    field the export omitted with nulls, so a field the export lacks is present
    and empty rather than missing; keeping it would delete the entire sample
    through listwise deletion.

    Dropping one is honest — the premium it carries is still in the price, it is
    simply unobserved — but the caller has to record that it happened, because
    an unobserved hedonic is a direct attenuation of `beta_price`.
    `run_diagnostics` reports exactly that.

    Amenity indicators are appended when present, so the view signal enters as
    ~17 binary columns rather than as a 1,139-level string.
    """
    present = tuple(c for c in covariates if c in frame.columns and frame[c].notna().any())
    if not include_tokens:
        return present
    return present + token_covariates(frame)


class CoxDemandModel:
    """Cox PH on time-to-sale, fitted on a feature frame.

    Args:
        covariates: overrides `DEMAND_COVARIATES`.
        building_fixed_effects: absorb `building_name`. Strongly reduces
            quality contamination of `rel_price_premium` where quality is
            building-level, at the cost of one parameter per tower.
        penalizer: lifelines ridge penalty. Left at zero by default because a
            penalty shrinks `beta_price` toward zero, which is the direction
            that flatters the optimizer.
    """

    kind = "cox"

    def __init__(
        self,
        *,
        covariates: Sequence[str] = DEMAND_COVARIATES,
        categoricals: Sequence[str] = DEFAULT_CATEGORICALS,
        building_fixed_effects: bool = False,
        penalizer: float = 0.0,
        min_level_count: int = MIN_LEVEL_COUNT,
        premium_model: Any = None,
    ) -> None:
        self.covariates = tuple(covariates)
        self.categoricals = tuple(categoricals)
        self.building_fixed_effects = building_fixed_effects
        self.penalizer = float(penalizer)
        self.min_level_count = int(min_level_count)
        self._fitter: CoxPHFitter | None = None
        self._design: DesignMatrix | None = None
        self._result: FitResult | None = None
        self._premium_support: PremiumSupport | None = None
        self._max_duration_days: float | None = None
        # The list-price hedonic whose residual is `rel_price_premium`. Carried
        # on the model so that scoring measures a candidate price against the
        # same reference the coefficient was fitted against. Pass it explicitly
        # from `FeatureResult.premium_model`; the `frame.attrs` lookup in `fit`
        # is a fallback that `pd.concat` can defeat.
        self.premium_model: Any = premium_model

    # --- fitting ---------------------------------------------------------
    @property
    def fixed_effects(self) -> tuple[str, ...]:
        return (BUILDING_FE_COLUMN,) if self.building_fixed_effects else ()

    @property
    def effective_covariates(self) -> tuple[str, ...]:
        """Covariates after removing any that the fixed effects already absorb.

        Every building sits in exactly one submarket, so the submarket dummies
        are an exact linear combination of the building dummies and the design
        is singular if both are present. Building fixed effects strictly nest
        submarket, so submarket is the one to go.
        """
        if not self.building_fixed_effects:
            return self.covariates
        return tuple(c for c in self.covariates if c != "submarket")

    def fit(self, frame: pd.DataFrame) -> FitResult:
        """Fit the Cox model.

        Args:
            frame: feature frame carrying the covariates plus `duration_days`
                and `event_sold`.

        Raises:
            SchemaError: when duration or event columns are missing or unusable.
            IdentificationError: when the partial likelihood fails to converge.
        """
        for column in (DURATION_COL, EVENT_COL):
            if column not in frame.columns:
                raise SchemaError(f"Cox fit requires column {column!r}")

        outcome = pd.DataFrame(
            {
                DURATION_COL: pd.to_numeric(frame[DURATION_COL], errors="coerce"),
                EVENT_COL: pd.to_numeric(frame[EVENT_COL], errors="coerce"),
            },
            index=frame.index,
        )
        # A duration of zero or less carries no survival information and makes
        # the partial likelihood undefined; drop before the design is built so
        # the loss is attributed to duration rather than to a covariate.
        usable = frame.loc[
            outcome[DURATION_COL].gt(0) & outcome[EVENT_COL].isin([0, 1])
        ]
        dropped_outcome = len(frame) - len(usable)

        design = build_design_matrix(
            usable,
            self.effective_covariates,
            categoricals=self.categoricals,
            fixed_effects=self.fixed_effects,
            min_level_count=self.min_level_count,
        )
        fit_frame = design.X.copy()
        fit_frame[DURATION_COL] = outcome.loc[design.index, DURATION_COL].to_numpy()
        fit_frame[EVENT_COL] = outcome.loc[design.index, EVENT_COL].astype(int).to_numpy()

        n_events = int(fit_frame[EVENT_COL].sum())
        if n_events < 2:
            raise SchemaError(
                f"Cox fit needs at least 2 sales, found {n_events} in "
                f"{len(fit_frame)} usable listings"
            )

        fitter = CoxPHFitter(penalizer=self.penalizer)
        # lifelines reports complete separation, non-unique solutions, and
        # high-norm Newton steps as warnings, not exceptions — they go to stderr
        # and vanish. A fit that "succeeded" with a non-unique solution returns
        # coefficients with the same type and precision as a converged one, so
        # AGENTS.md §3 requires they be captured and carried, not logged away.
        convergence_warnings: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitter.fit(fit_frame, duration_col=DURATION_COL, event_col=EVENT_COL)
            convergence_warnings = [
                str(w.message).strip().split("\n")[0]
                for w in caught
                if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))
            ]
        except (ConvergenceError, np.linalg.LinAlgError) as exc:
            raise IdentificationError(
                "Cox partial likelihood did not converge. This usually means "
                "collinear covariates or a fixed effect with no within-level "
                f"variation in outcome. Underlying error: {exc}"
            ) from exc

        coefficients = unscale_coefficients(_summary_to_raw(fitter.summary), design)

        notes = list(design.notes)
        if dropped_outcome:
            notes.append(
                f"dropped {dropped_outcome} rows with a non-positive duration or a "
                "non-binary event flag"
            )
        if self.building_fixed_effects:
            kept_levels = len(design.categorical_levels.get(BUILDING_FE_COLUMN, [])) + 1
            pooled = design.pooled_rare_levels.get(BUILDING_FE_COLUMN, 0)
            notes.append(
                f"building fixed effects absorbed {kept_levels} towers; "
                "submarket dropped because the fixed effects nest it"
            )
            total_levels = kept_levels + pooled
            if total_levels and pooled / total_levels > _FE_POOLED_SHARE_ALARM:
                notes.append(
                    f"BUILDING FIXED EFFECTS ARE MOSTLY NOT FIXED EFFECTS — "
                    f"{pooled} of {total_levels} towers ({pooled / total_levels:.0%}) hold "
                    f"fewer than {self.min_level_count} listings and were pooled into a "
                    "single residual level. A pooled bucket absorbs nothing "
                    "building-specific, so the quality contamination these effects exist "
                    "to remove is still inside rel_price_premium for most of the sample. "
                    "Read the controlled estimate as barely more absorbed than the naive "
                    "one, not as a clean within-building comparison."
                )
            density = len(fit_frame) / max(kept_levels, 1)
            if density < _FE_MIN_ROWS_PER_LEVEL:
                notes.append(
                    f"THIN FIXED EFFECTS — {density:.1f} listings per retained tower. A "
                    "Cox partial likelihood carrying one dummy per level is subject to "
                    "incidental-parameters bias that attenuates every coefficient toward "
                    "zero as levels thin. Measured on synthetic data with the "
                    "contamination fully building-level, recovery of a planted -1.6 falls "
                    "from 74% of the planted value at ~60 listings per tower to 44% at "
                    "~34 (audit/a11_building_fe_coupling.py). Treat beta_price from this "
                    "fit as a lower bound in magnitude, not as a corrected estimate."
                )
        if convergence_warnings:
            unique = list(dict.fromkeys(convergence_warnings))
            notes.append(
                "CONVERGENCE WARNINGS FROM THE FITTER — "
                + " | ".join(unique[:4])
                + ". lifelines raises these as warnings rather than errors, so the fit "
                "returned coefficients anyway. Complete separation or a non-unique "
                "solution means those coefficients are not identified by this data; "
                "do not read them as estimates."
            )

        result = FitResult(
            model_kind=self.kind,
            coefficients=coefficients,
            n_observations=len(fit_frame),
            n_events=n_events,
            fit_stats={
                "concordance": float(fitter.concordance_index_),
                "log_likelihood": float(fitter.log_likelihood_),
                "aic_partial": float(fitter.AIC_partial_),
            },
            design=design,
            covariates=self.effective_covariates,
            notes=notes,
        )
        self._fitter, self._design, self._result = fitter, design, result
        self._max_duration_days = float(fit_frame[DURATION_COL].max())
        if self.premium_model is None:
            self.premium_model = frame.attrs.get("premium_model")
        if self.premium_model is None and _premium_is_residual(frame):
            raise IdentificationError(
                "This frame's rel_price_premium is a hedonic residual, but no "
                "premium model reached the fit — `frame.attrs` does not survive "
                "pd.concat and none was passed explicitly. Continuing would fit "
                "beta_price against one reference and later score prices against "
                "another (the submarket median), which is a units error that "
                "produces entirely plausible numbers. Pass "
                "`FeatureResult.premium_model` into CoxDemandModel."
            )
        # Recorded over the rows that survived listwise deletion, not the input
        # frame: those are the listings the coefficient is actually evidence
        # about. Downstream, the optimizer checks its prices against this.
        self._premium_support = premium_support(
            usable.loc[design.index, "rel_price_premium"]
            if "rel_price_premium" in usable.columns
            else pd.Series(dtype="float64")
        )
        logger.info(
            "Cox fitted: n=%d events=%d beta_price=%.4f concordance=%.3f",
            result.n_observations,
            n_events,
            result.beta_price.value,
            result.fit_stats["concordance"],
        )
        return result

    # --- prediction ------------------------------------------------------
    def baseline_survival(self, times: Sequence[float]) -> np.ndarray:
        """Baseline survival S0(t) at each of `times` (days).

        Exposed because the optimizer needs the *shape* of the sale-time
        distribution, not only its value at the horizon. Under proportional
        hazards every unit's curve is S0(t) raised to a power, so this one
        vector plus the already-computed horizon probability reconstructs the
        whole distribution for every unit at every price without a second pass
        through the fitter.

        Values past the last observed event are the last observed value —
        lifelines carries it forward and so does this. Callers integrating
        against the curve must cap at `max_observed_duration_days` rather than
        treat the flat region as information.
        """
        if self._fitter is None:
            raise IdentificationError("Cox model has not been fitted")
        baseline = self._fitter.baseline_survival_
        grid = baseline.index.to_numpy(dtype="float64")
        values = baseline.iloc[:, 0].to_numpy(dtype="float64")
        wanted = np.asarray(times, dtype="float64")
        # Left of the first event time survival is 1; right of the last it is
        # held flat, which is what np.interp does at both ends given these
        # endpoints.
        return np.interp(wanted, grid, values, left=1.0, right=float(values[-1]))

    @property
    def coefficient_covariance(self) -> pd.DataFrame | None:
        """Full covariance of the fitted coefficients, in **raw** covariate units.

        lifelines fits on the scaled design, so its `variance_matrix_` is in
        scaled units; dividing entry (m, n) by `scale[m] * scale[n]` returns it
        to the same units as the reported coefficients.

        This exists because the marginal standard errors alone are not enough to
        resample the model. `beta_price` is correlated with the other
        coefficients — with `log_floor` at about -0.27 on the current synthetic
        fit — so drawing it on its own, or drawing several coefficients
        independently, describes a joint distribution the data never supported.
        Persisting it is what lets a simulation propagate the whole fit's
        uncertainty rather than one coefficient's.
        """
        if self._fitter is None or self._design is None:
            return None
        matrix = self._fitter.variance_matrix_
        scales = pd.Series(
            [self._design.scale.get(c, 1.0) or 1.0 for c in matrix.columns],
            index=matrix.columns, dtype="float64",
        )
        return matrix.div(scales, axis=0).div(scales, axis=1)

    @property
    def max_observed_duration_days(self) -> float | None:
        """Longest duration the fit saw, in days. None before fitting.

        The Cox baseline hazard is non-parametric and therefore identified only
        over this range. Beyond it lifelines carries the last value forward, so
        the survival function flattens and `predict_sale_probability` returns
        the final observed probability for any later horizon without saying so.
        Callers that accept a horizon from a user should compare against this.
        """
        return self._max_duration_days

    @property
    def premium_support(self) -> PremiumSupport | None:
        """Range of `rel_price_premium` this fit is evidence about.

        None before fitting, or when the fitting sample carried no usable
        premium at all. The optimizer treats None as "cannot check" and says so
        rather than assuming the prices are in range.
        """
        return self._premium_support

    @property
    def result(self) -> FitResult:
        if self._result is None:
            raise IdentificationError("Cox model has not been fitted")
        return self._result

    def predict_sale_probability(
        self,
        features: pd.DataFrame,
        price_ppsf: float | np.ndarray,
        horizon_days: int,
    ) -> np.ndarray:
        """P(sold within `horizon_days`) at an asking price of `price_ppsf` $/sqft.

        The price enters only through `rel_price_premium`, recomputed from
        `features["cell_median_ppsf"]`, so a unit's sale probability depends on
        where it sits against its comps rather than on its absolute level.

        Returns:
            One probability per row of `features`, NaN where a covariate is
            missing. NaN rather than a filled-in guess: a fabricated sale
            probability propagates straight into the optimizer's objective.
        """
        if horizon_days <= 0:
            raise SchemaError(f"horizon_days must be positive, got {horizon_days}")
        if self._fitter is None or self._design is None:
            raise IdentificationError("Cox model has not been fitted")
        if "cell_median_ppsf" not in features.columns:
            raise SchemaError(
                "predict_sale_probability requires cell_median_ppsf ($/sqft) to "
                "convert an asking price into rel_price_premium"
            )

        scored = features.copy()
        scored["rel_price_premium"] = relative_premium(
            price_ppsf, scored["cell_median_ppsf"].to_numpy(dtype="float64")
        )
        matrix, usable = transform_to_design(scored, self._design)

        out = np.full(len(features), np.nan, dtype="float64")
        if not usable.any():
            logger.warning("No rows in the prediction frame have complete covariates")
            return out

        survival = self._fitter.predict_survival_function(matrix, times=[float(horizon_days)])
        out[np.flatnonzero(usable.to_numpy())] = 1.0 - survival.to_numpy().ravel()
        return out

    def survival_curve(self, features: pd.DataFrame, price_ppsf: float | np.ndarray,
                       times: Sequence[float]) -> pd.DataFrame:
        """Survival probabilities at each of `times` (days), one column per row."""
        if self._fitter is None or self._design is None:
            raise IdentificationError("Cox model has not been fitted")
        scored = features.copy()
        scored["rel_price_premium"] = relative_premium(
            price_ppsf, scored["cell_median_ppsf"].to_numpy(dtype="float64")
        )
        matrix, _ = transform_to_design(scored, self._design)
        return self._fitter.predict_survival_function(matrix, times=[float(t) for t in times])


def _summary_to_raw(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Extract lifelines' summary table into the shape `unscale_coefficients` wants."""
    out: dict[str, dict[str, float]] = {}
    for name, row in summary.iterrows():
        out[str(name)] = {
            "value": float(row["coef"]),
            "std_error": float(row["se(coef)"]),
            "ci_low": float(row["coef lower 95%"]),
            "ci_high": float(row["coef upper 95%"]),
            "p_value": float(row["p"]),
        }
    return out


def fit_cox(frame: pd.DataFrame, **kwargs: Any) -> tuple[CoxDemandModel, FitResult]:
    """Convenience wrapper: construct, fit, and return both model and result."""
    model = CoxDemandModel(**kwargs)
    return model, model.fit(frame)


def beta_price_of(result: FitResult) -> Coefficient:
    """The elasticity coefficient, named for readability at call sites."""
    return result.beta_price
