"""Hedonic price surface — what a floor, a view, and a square foot are worth.

This is a different question from elasticity and must not be confused with it.
The Cox model answers "how much slower does a unit sell when it is priced above
its comps"; this one answers "what should the comp be". It is fitted on closed
sales only, because a list price is an asking price and an unsold asking price
is evidence about seller optimism rather than about market value.

Its two outputs downstream are the floor premium curve and the per-unit price
band the optimizer searches within.

On `premium(floor) = alpha * ln(floor + 1) + gamma`: `alpha` is the coefficient
on `log_floor` in the surface and is estimated. `gamma` is *not* a second free
parameter — in a log-linear specification it is collinear with the intercept,
which already carries the base $/sqft — so it is the normalization that pins the
premium to zero at a reference floor: `gamma = -alpha * ln(reference + 1)`. The
reference is a reporting convention held in market config (floor 1: premiums are
quoted against a first-floor unit), not an estimate. Everything with information
in it, `alpha` and its standard error, comes from the data.

Anyone tempted to "estimate gamma properly" should note that regressing the
floor partial residual on `[1, ln(floor+1)]` returns a gamma of exactly zero,
because OLS residuals are orthogonal to both regressors. There is no second
parameter hiding there to find.

Units:
- `close_ppsf`, `p_floor_ppsf`, `p_ceiling_ppsf`: $/sqft.
- `alpha`, `gamma`, and `premium`: log points (multiply price by `exp(premium)`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import MarketConfig
from src.demand.base import (
    Coefficient,
    DesignMatrix,
    FitResult,
    build_design_matrix,
    transform_to_design,
    unscale_coefficients,
)
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

# Unit characteristics plus a time control: closed prices span the whole export
# window, and a surface with no time term attributes market drift to whichever
# feature happens to correlate with when units sold.
HEDONIC_COVARIATES: tuple[str, ...] = (
    "log_floor",
    "log_living_area",
    "beds",
    "baths_full",
    "hoa_per_sqft",
    "is_new_construction",
    "submarket",
    "view_description",
    "list_quarter",
)
HEDONIC_CATEGORICALS: tuple[str, ...] = ("submarket", "view_description", "list_quarter")

_FLOOR_TERM = "log_floor"
_DEFAULT_BOUND_SD_MULTIPLE = 1.5
_DEFAULT_MIN_SALES = 50
# Floor the premium curve is quoted against. A reporting convention, not an
# estimate: premium(1) = 0 means "relative to a first-floor unit".
_DEFAULT_REFERENCE_FLOOR = 1.0


@dataclass(frozen=True)
class FloorPremium:
    """Fitted floor premium curve, `premium(floor) = alpha*ln(floor+1) + gamma`.

    Values are log points: multiply a $/sqft figure by `exp(premium)`. `alpha`
    is estimated from the hedonic surface; `gamma` is the normalization that
    makes `premium(reference_floor)` zero, so its uncertainty is `alpha`'s
    scaled by the same constant.
    """

    alpha: float
    gamma: float
    alpha_std_error: float
    gamma_std_error: float
    reference_floor: float

    def premium(self, floor: float | np.ndarray) -> np.ndarray:
        """Log premium at a given floor. Multiply price by `exp(premium)`."""
        level = np.asarray(floor, dtype="float64")
        with np.errstate(invalid="ignore"):
            out = np.where(level >= 0, self.alpha * np.log(level + 1.0) + self.gamma, np.nan)
        return np.asarray(out, dtype="float64")

    def multiplier(self, floor: float | np.ndarray) -> np.ndarray:
        """Multiplicative $/sqft factor attributable to the floor."""
        return np.exp(self.premium(floor))

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "alpha_std_error": self.alpha_std_error,
            "gamma_std_error": self.gamma_std_error,
            "reference_floor": self.reference_floor,
        }


@dataclass
class HedonicSurface:
    """Fitted surface plus everything the optimizer needs to price against it."""

    fit: FitResult
    floor_premium: FloorPremium
    residual_sd: float
    bound_sd_multiple: float
    _design: DesignMatrix = field(repr=False)
    _model: object = field(repr=False)

    @property
    def design(self) -> DesignMatrix:
        """The fitted design, for callers that need its level vocabulary.

        The optimizer reads the time-control levels off it to price the comps
        band as of an observed period rather than a future phase date.
        """
        return self._design

    def predict_ppsf(self, units: pd.DataFrame) -> np.ndarray:
        """Predicted $/sqft for each row; NaN where a covariate is missing.

        The prediction is `exp(mean of log ppsf)`, the conditional *median*
        rather than the mean. That is the right central estimate for a
        log-normal surface and it is what the price band is centred on; a
        smearing correction would push the whole band up by half a variance and
        flatter the optimizer.
        """
        matrix, usable = transform_to_design(units, self._design)
        out = np.full(len(units), np.nan, dtype="float64")
        if not usable.any():
            return out
        predicted = self._model.predict(sm.add_constant(matrix, has_constant="add"))
        out[np.flatnonzero(usable.to_numpy())] = np.exp(np.asarray(predicted, dtype="float64"))
        return out

    def price_bounds(self, units: pd.DataFrame) -> pd.DataFrame:
        """Comps-based $/sqft band per unit.

        Returns a frame with `predicted_ppsf`, `p_floor_ppsf`, `p_ceiling_ppsf`.
        The band is the fitted value times `exp(±k·residual_sd)`, so it is
        symmetric in log price and therefore asymmetric in dollars — which is
        how price dispersion actually behaves. This is the *comps* half of the
        bound only; the optimizer must still intersect it with the cost basis.
        """
        predicted = self.predict_ppsf(units)
        spread = float(self.bound_sd_multiple) * self.residual_sd
        return pd.DataFrame(
            {
                "predicted_ppsf": predicted,
                "p_floor_ppsf": predicted * float(np.exp(-spread)),
                "p_ceiling_ppsf": predicted * float(np.exp(spread)),
            },
            index=units.index,
        )


def _sold_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Closed sales with a usable $/sqft. Pending listings have no close price."""
    for column in ("close_ppsf", "event_sold"):
        if column not in frame.columns:
            raise SchemaError(f"Hedonic fit requires column {column!r}")
    ppsf = pd.to_numeric(frame["close_ppsf"], errors="coerce")
    return frame.loc[ppsf.notna() & ppsf.gt(0)]


def _floor_premium(
    coefficients: dict[str, Coefficient], reference_floor: float
) -> FloorPremium:
    """Assemble the floor premium curve from the fitted log_floor coefficient."""
    if _FLOOR_TERM not in coefficients:
        raise IdentificationError(
            "The hedonic surface has no log_floor term, so the floor premium is "
            "not identified. Every sold listing is missing a numeric floor."
        )
    if reference_floor < 0:
        raise SchemaError(f"reference_floor must be non-negative, got {reference_floor}")
    alpha = coefficients[_FLOOR_TERM]
    shift = float(np.log(float(reference_floor) + 1.0))
    return FloorPremium(
        alpha=alpha.value,
        gamma=-alpha.value * shift,
        alpha_std_error=alpha.std_error,
        gamma_std_error=alpha.std_error * shift,
        reference_floor=float(reference_floor),
    )


def fit_hedonic(
    frame: pd.DataFrame,
    config: MarketConfig | None = None,
    *,
    covariates: Sequence[str] = HEDONIC_COVARIATES,
    categoricals: Sequence[str] = HEDONIC_CATEGORICALS,
    bound_sd_multiple: float | None = None,
    min_sales: int | None = None,
    reference_floor: float | None = None,
) -> HedonicSurface:
    """Fit `log(close_ppsf)` on unit characteristics over closed sales.

    Args:
        frame: feature frame from `build_features`.
        config: market config; supplies the bound width and the minimum sale
            count. Both are bounds on the procedure, not estimates.
        covariates: overrides `HEDONIC_COVARIATES`; absent or all-null columns
            are skipped (same rule as demand `available_covariates`).

    Raises:
        SchemaError: when there are too few closed sales to fit a surface.
        IdentificationError: when the design is singular.
    """
    defaults = (config or {}).get("defaults") or {}
    k = float(
        bound_sd_multiple
        if bound_sd_multiple is not None
        else defaults.get("hedonic_bound_sd_multiple", _DEFAULT_BOUND_SD_MULTIPLE)
    )
    floor_n = int(
        min_sales if min_sales is not None else defaults.get("min_hedonic_sales", _DEFAULT_MIN_SALES)
    )
    reference = float(
        reference_floor
        if reference_floor is not None
        else defaults.get("floor_premium_reference_floor", _DEFAULT_REFERENCE_FLOOR)
    )

    sold = _sold_sample(frame)
    if len(sold) < floor_n:
        raise SchemaError(
            f"Hedonic surface needs at least {floor_n} closed sales, found {len(sold)}. "
            "A surface fitted on fewer will not support per-unit price bounds."
        )

    # Normalization fills omitted fields with nulls, so view_description is
    # present-and-empty on the Miami export. Keeping it would listwise-delete
    # every closed sale. Same honesty rule as available_covariates in survival.
    present = tuple(
        c for c in covariates if c in sold.columns and sold[c].notna().any()
    )
    skipped = [c for c in covariates if c not in present]
    design = build_design_matrix(
        sold, present, categoricals=[c for c in categoricals if c in present]
    )
    y = np.log(pd.to_numeric(sold.loc[design.index, "close_ppsf"], errors="coerce").to_numpy())

    X = sm.add_constant(design.X, has_constant="add")
    try:
        model = sm.OLS(y, X).fit()
    except np.linalg.LinAlgError as exc:
        raise IdentificationError(
            f"Hedonic surface is singular — collinear unit features. Error: {exc}"
        ) from exc

    conf = model.conf_int()
    raw = {
        str(name): {
            "value": float(model.params[name]),
            "std_error": float(model.bse[name]),
            "ci_low": float(conf.loc[name, 0]),
            "ci_high": float(conf.loc[name, 1]),
            "p_value": float(model.pvalues[name]),
        }
        for name in model.params.index
    }
    coefficients = unscale_coefficients(raw, design)

    notes = list(design.notes)
    if skipped:
        notes.append(
            f"covariates absent or all-null in this export and therefore unpriced: "
            f"{skipped}. Their contribution stays in the residual as unobserved quality."
        )

    residual_sd = float(np.sqrt(model.mse_resid))
    result = FitResult(
        model_kind="hedonic",
        coefficients=coefficients,
        n_observations=int(model.nobs),
        n_events=None,
        fit_stats={
            "r_squared": float(model.rsquared),
            "r_squared_adj": float(model.rsquared_adj),
            "residual_sd_log_ppsf": residual_sd,
        },
        design=design,
        covariates=present,
        notes=notes,
    )
    floor_premium = _floor_premium(coefficients, reference)

    logger.info(
        "Hedonic fitted on %d sales: R2=%.3f alpha=%.4f gamma=%.4f residual_sd=%.4f",
        result.n_observations,
        result.fit_stats["r_squared"],
        floor_premium.alpha,
        floor_premium.gamma,
        residual_sd,
    )
    return HedonicSurface(
        fit=result,
        floor_premium=floor_premium,
        residual_sd=residual_sd,
        bound_sd_multiple=k,
        _design=design,
        _model=model,
    )
