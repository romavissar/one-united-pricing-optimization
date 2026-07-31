"""Logistic model of P(sold within H days) — the secondary estimator.

It exists as a cross-check. The Cox model uses the full timing information and
is the one to report; this one collapses timing to a single yes/no at a fixed
horizon, which throws away information but is far easier to reason about and
gives a coefficient sign that can be eyeballed. When the two disagree in sign,
the specification is wrong, not one of the estimators.

The subtlety that makes or breaks it is the label. A listing withdrawn on day 30
has an *unknown* outcome at a 180-day horizon: it might have sold on day 90 had
it stayed listed. Scoring it as "did not sell" turns every early withdrawal into
a fake failure and biases the model toward pessimism. So:

- sold on or before H                  -> y = 1
- observed at least H days, no sale    -> y = 0
- censored before H                    -> excluded, outcome unknown

The excluded count is reported, not swallowed. It is the price of collapsing a
survival problem into a binary one, and it is exactly the information the Cox
model keeps.

Units:
- `horizon_days`, `duration_days`: days.
- coefficients: per unit of covariate on the log-odds scale.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.data.features import DEMAND_COVARIATES
from src.demand.base import (
    DEFAULT_CATEGORICALS,
    MIN_LEVEL_COUNT,
    DesignMatrix,
    FitResult,
    build_design_matrix,
    relative_premium,
    transform_to_design,
    unscale_coefficients,
)
from src.demand.survival import BUILDING_FE_COLUMN, DURATION_COL, EVENT_COL
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

INTERCEPT = "const"
_DEFAULT_TEST_SIZE = 0.25
_DEFAULT_SEED = 20240101


@dataclass
class HorizonLabel:
    """Binary sale-within-horizon outcome and the rows it could be built for."""

    y: pd.Series
    usable: pd.Series
    horizon_days: int
    n_sold_by_horizon: int
    n_survived_horizon: int
    n_unknown_at_horizon: int

    def as_dict(self) -> dict[str, int]:
        return {
            "horizon_days": self.horizon_days,
            "sold_by_horizon": self.n_sold_by_horizon,
            "survived_horizon": self.n_survived_horizon,
            "unknown_at_horizon": self.n_unknown_at_horizon,
        }


def build_horizon_label(frame: pd.DataFrame, horizon_days: int) -> HorizonLabel:
    """Label each listing as sold / not sold / unknown at `horizon_days`.

    Raises:
        SchemaError: when duration or event columns are absent, or the horizon
            is not positive.
    """
    if horizon_days <= 0:
        raise SchemaError(f"horizon_days must be positive, got {horizon_days}")
    for column in (DURATION_COL, EVENT_COL):
        if column not in frame.columns:
            raise SchemaError(f"Horizon label requires column {column!r}")

    duration = pd.to_numeric(frame[DURATION_COL], errors="coerce")
    event = pd.to_numeric(frame[EVENT_COL], errors="coerce")
    valid = duration.notna() & duration.gt(0) & event.isin([0, 1])

    sold_by = valid & event.eq(1) & duration.le(horizon_days)
    survived = valid & ~sold_by & duration.ge(horizon_days)
    usable = sold_by | survived

    y = pd.Series(np.nan, index=frame.index, dtype="float64")
    y.loc[sold_by] = 1.0
    y.loc[survived] = 0.0

    return HorizonLabel(
        y=y,
        usable=usable,
        horizon_days=int(horizon_days),
        n_sold_by_horizon=int(sold_by.sum()),
        n_survived_horizon=int(survived.sum()),
        n_unknown_at_horizon=int((~usable).sum()),
    )


class LogisticDemandModel:
    """P(sold within a fixed horizon) as a logit on the same covariates as the Cox."""

    kind = "logistic"

    def __init__(
        self,
        horizon_days: int = 180,
        *,
        covariates: Sequence[str] = DEMAND_COVARIATES,
        categoricals: Sequence[str] = DEFAULT_CATEGORICALS,
        building_fixed_effects: bool = False,
        min_level_count: int = MIN_LEVEL_COUNT,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.covariates = tuple(covariates)
        self.categoricals = tuple(categoricals)
        self.building_fixed_effects = building_fixed_effects
        self.min_level_count = int(min_level_count)
        self.test_size = float(test_size)
        self.seed = int(seed)
        self._fitted: Any = None
        self._design: DesignMatrix | None = None
        self._result: FitResult | None = None
        self._label: HorizonLabel | None = None

    @property
    def fixed_effects(self) -> tuple[str, ...]:
        return (BUILDING_FE_COLUMN,) if self.building_fixed_effects else ()

    @property
    def effective_covariates(self) -> tuple[str, ...]:
        """Covariates minus any the fixed effects already absorb (see the Cox model)."""
        if not self.building_fixed_effects:
            return self.covariates
        return tuple(c for c in self.covariates if c != "submarket")

    def fit(self, frame: pd.DataFrame) -> FitResult:
        """Fit the logit and score held-out AUC.

        Coefficients come from the full-sample fit; AUC comes from a held-out
        split, because an in-sample AUC on a model with building fixed effects
        measures memorisation rather than discrimination.

        Raises:
            IdentificationError: on perfect separation or a singular design.
        """
        label = build_horizon_label(frame, self.horizon_days)
        labelled = frame.loc[label.usable]
        if labelled.empty:
            raise SchemaError(
                f"No listing has a known outcome at {self.horizon_days} days; "
                "every record is censored before the horizon"
            )

        design = build_design_matrix(
            labelled,
            self.effective_covariates,
            categoricals=self.categoricals,
            fixed_effects=self.fixed_effects,
            min_level_count=self.min_level_count,
        )
        X = sm.add_constant(design.X, has_constant="add")
        y = label.y.loc[design.index].astype(float)
        if y.nunique() < 2:
            raise IdentificationError(
                f"Every usable listing has the same outcome at {self.horizon_days} "
                "days, so no logit can be fitted"
            )

        try:
            fitted = sm.Logit(y, X).fit(disp=0, maxiter=200)
        except (PerfectSeparationError, np.linalg.LinAlgError) as exc:
            raise IdentificationError(
                "Logistic fit failed — perfect separation or a singular design. "
                "A fixed effect whose levels perfectly predict the outcome is the "
                f"usual cause. Underlying error: {exc}"
            ) from exc

        conf = fitted.conf_int()
        raw = {
            str(name): {
                "value": float(fitted.params[name]),
                "std_error": float(fitted.bse[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
                "p_value": float(fitted.pvalues[name]),
            }
            for name in fitted.params.index
        }
        coefficients = unscale_coefficients(raw, design)

        auc, auc_note = self._holdout_auc(X, y)
        notes = list(design.notes)
        notes.append(
            f"{label.n_unknown_at_horizon} listings were censored before "
            f"{self.horizon_days} days and have no known outcome; they are excluded "
            "from the logit but retained by the Cox model"
        )
        if auc_note:
            notes.append(auc_note)

        result = FitResult(
            model_kind=self.kind,
            coefficients=coefficients,
            n_observations=int(len(y)),
            n_events=int(y.sum()),
            fit_stats={
                "auc_holdout": auc,
                "pseudo_r2_mcfadden": float(fitted.prsquared),
                "log_likelihood": float(fitted.llf),
                "horizon_days": float(self.horizon_days),
            },
            design=design,
            covariates=self.effective_covariates,
            notes=notes,
        )
        self._fitted, self._design, self._result, self._label = fitted, design, result, label
        logger.info(
            "Logit fitted at H=%d: n=%d sold=%d beta_price=%.4f auc=%.3f",
            self.horizon_days,
            result.n_observations,
            result.n_events or 0,
            result.beta_price.value,
            auc,
        )
        return result

    def _holdout_auc(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, str | None]:
        """AUC on a stratified held-out split; NaN when the split is not viable."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed, stratify=y
            )
            split_fit = sm.Logit(y_train, X_train).fit(disp=0, maxiter=200)
            scores = split_fit.predict(X_test)
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            return float("nan"), f"held-out AUC unavailable: {exc}"
        if y_test.nunique() < 2:
            return float("nan"), "held-out AUC unavailable: test split is single-class"
        return float(roc_auc_score(y_test, scores)), None

    @property
    def result(self) -> FitResult:
        if self._result is None:
            raise IdentificationError("Logistic model has not been fitted")
        return self._result

    @property
    def label(self) -> HorizonLabel:
        if self._label is None:
            raise IdentificationError("Logistic model has not been fitted")
        return self._label

    def predict_sale_probability(
        self,
        features: pd.DataFrame,
        price_ppsf: float | np.ndarray,
        horizon_days: int | None = None,
    ) -> np.ndarray:
        """P(sold within the fitted horizon) at `price_ppsf` dollars per square foot.

        Unlike the Cox model this one cannot be evaluated at an arbitrary
        horizon: the label was built at a single H, so asking for another one is
        a request the fitted object cannot honour.

        Raises:
            SchemaError: when `horizon_days` differs from the fitted horizon.
        """
        if horizon_days is not None and int(horizon_days) != self.horizon_days:
            raise SchemaError(
                f"This logit was fitted at {self.horizon_days} days and cannot be "
                f"evaluated at {horizon_days}. Refit, or use the Cox model, whose "
                "survival function is defined at every horizon."
            )
        if self._fitted is None or self._design is None:
            raise IdentificationError("Logistic model has not been fitted")
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
        out[np.flatnonzero(usable.to_numpy())] = self._fitted.predict(
            sm.add_constant(matrix, has_constant="add")
        ).to_numpy()
        return out

    def predicted_probabilities(self) -> pd.Series:
        """In-sample fitted probabilities, indexed like the fitting sample."""
        if self._fitted is None or self._design is None:
            raise IdentificationError("Logistic model has not been fitted")
        return pd.Series(
            self._fitted.predict().astype(float), index=self._design.index, name="p_hat"
        )

    def observed_outcomes(self) -> pd.Series:
        """The binary labels the model was fitted on, indexed like the sample."""
        return self.label.y.loc[self.result.design.index].astype(float)


def fit_logistic(
    frame: pd.DataFrame, horizon_days: int = 180, **kwargs: Any
) -> tuple[LogisticDemandModel, FitResult]:
    """Convenience wrapper: construct, fit, and return both model and result."""
    model = LogisticDemandModel(horizon_days, **kwargs)
    return model, model.fit(frame)
