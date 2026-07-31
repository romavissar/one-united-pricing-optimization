"""Shared types and the design-matrix builder for the demand estimators.

Both the Cox and logistic models regress on the same covariates, so the messy
part — categorical expansion, fixed effects, null handling, scaling — lives here
once. Two rules govern it:

Nothing is imputed. A row with a null covariate leaves the sample and the loss
is counted per covariate in `DesignMatrix.dropped_by_null`, because a covariate
that quietly costs 40% of the listings is a bigger threat to `beta_price` than
one that is merely noisy.

Columns are scaled to unit standard deviation for numerical conditioning only.
`living_area_sqft` is ~1400 and `rel_price_premium` is ~0.05; without scaling
the Cox Newton step is badly conditioned. Because both models are linear in the
covariates, dividing a column by its sd and multiplying its coefficient back is
exact, so every reported coefficient is in raw units.

Units:
- `rel_price_premium`: ratio minus one (0.10 = priced 10% above the cell median).
- `living_area_sqft`: sqft. `hoa_per_sqft`: $/sqft/month.
- coefficients: per one unit of the covariate, on the log-hazard (Cox) or
  log-odds (logistic) scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.linalg import qr as scipy_qr

from src.exceptions import SchemaError

# Categorical covariates get one-hot expanded against a dropped reference level.
DEFAULT_CATEGORICALS: tuple[str, ...] = ("submarket", "season")

# Levels thinner than this are pooled into a single residual level. A fixed
# effect with one observation fits that observation exactly, which separates the
# likelihood in logistic and inflates the standard errors in Cox.
MIN_LEVEL_COUNT = 5
RARE_LEVEL = "__rare__"


@dataclass(frozen=True)
class Coefficient:
    """One fitted coefficient in raw covariate units, with its uncertainty."""

    name: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        """True when the 95% CI lies wholly on one side of zero."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "value": self.value,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
        }


@dataclass
class DesignMatrix:
    """Model matrix plus an account of everything that happened to build it."""

    X: pd.DataFrame
    index: pd.Index
    rows_in: int
    rows_used: int
    dropped_by_null: dict[str, int] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    reference_levels: dict[str, str] = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_dependent: list[str] = field(default_factory=list)
    pooled_rare_levels: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def retention(self) -> float:
        """Share of input rows that survived listwise deletion."""
        return self.rows_used / self.rows_in if self.rows_in else 0.0


@dataclass
class FitResult:
    """A fitted demand model's coefficients and fit statistics.

    n_events is the number of sales; None for models where the notion does not
    apply (the hedonic surface, which is fitted on sales only).
    """

    model_kind: str
    coefficients: dict[str, Coefficient]
    n_observations: int
    n_events: int | None
    fit_stats: dict[str, float]
    design: DesignMatrix
    covariates: tuple[str, ...]
    notes: list[str] = field(default_factory=list)

    def coefficient(self, name: str) -> Coefficient:
        """Fetch a coefficient by covariate name.

        Raises:
            KeyError: when the covariate was dropped or never entered the fit.
        """
        if name not in self.coefficients:
            raise KeyError(
                f"{name!r} is not in the fitted {self.model_kind} model; "
                f"available: {sorted(self.coefficients)}"
            )
        return self.coefficients[name]

    @property
    def beta_price(self) -> Coefficient:
        """The coefficient on rel_price_premium — the number this project exists for."""
        return self.coefficient("rel_price_premium")


@dataclass(frozen=True)
class PremiumSupport:
    """The range of `rel_price_premium` a model was actually fitted over.

    A coefficient is evidence about the region of the data it was estimated on.
    `beta_price` fitted on listings priced between 30% below comps and 60% above
    says nothing about a unit asked at double its comps — the Cox linear
    predictor will happily produce a number there, and that number is an
    extrapolation wearing the same units as an estimate.

    Both bounds are carried: the observed min and max, which mark where the
    model has *no* evidence at all, and the 1st and 99th percentiles, which mark
    where it has very little. Crossing the outer pair is extrapolation; sitting
    between the inner and outer pair is thin support.

    Units: ratio minus one, matching `rel_price_premium` everywhere else.
    """

    low: float
    high: float
    p1: float
    p99: float
    n_observations: int

    def outside(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask of values beyond anything the fit ever saw."""
        premium = np.asarray(values, dtype="float64")
        return (premium < self.low) | (premium > self.high)

    def in_tail(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask of values inside the range but out in its thin tails."""
        premium = np.asarray(values, dtype="float64")
        return ~self.outside(premium) & ((premium < self.p1) | (premium > self.p99))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "low": self.low,
            "high": self.high,
            "p1": self.p1,
            "p99": self.p99,
            "n_observations": self.n_observations,
        }


def premium_support(values: pd.Series | np.ndarray) -> PremiumSupport | None:
    """Summarise the `rel_price_premium` range of a fitting sample.

    Returns None when there is nothing usable to summarise, which the caller
    must report rather than treat as "no extrapolation".
    """
    premium = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if premium.empty:
        return None
    return PremiumSupport(
        low=float(premium.min()),
        high=float(premium.max()),
        p1=float(premium.quantile(0.01)),
        p99=float(premium.quantile(0.99)),
        n_observations=int(len(premium)),
    )


@runtime_checkable
class DemandModel(Protocol):
    """What the optimizer and the API are allowed to assume about an estimator."""

    kind: str

    def fit(self, frame: pd.DataFrame) -> FitResult:
        """Fit on a feature frame from `src.data.features.build_features`."""
        ...

    def predict_sale_probability(
        self,
        features: pd.DataFrame,
        price_ppsf: float | np.ndarray,
        horizon_days: int,
    ) -> np.ndarray:
        """P(sold within horizon_days) at `price_ppsf` dollars per square foot.

        `features` must carry `cell_median_ppsf`, since the model responds to
        the price *relative* to its submarket-month comps, not to the level.
        """
        ...


def relative_premium(
    price_ppsf: float | np.ndarray, cell_median_ppsf: float | np.ndarray
) -> np.ndarray:
    """Convert a $/sqft asking price into `rel_price_premium` (ratio minus one).

    This is the single place price enters the demand models, so it is the single
    place the $/sqft-versus-total-price confusion can happen. Both arguments are
    $/sqft. A non-positive or missing median yields NaN rather than a silently
    plausible zero premium.
    """
    price = np.asarray(price_ppsf, dtype="float64")
    median = np.asarray(cell_median_ppsf, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(median > 0, price / median - 1.0, np.nan)
    return np.asarray(out, dtype="float64")


def _to_numeric(name: str, series: pd.Series) -> pd.Series:
    """Coerce a covariate column to float, mapping nullable booleans to 0/1.

    Raises when coercion destroys data. A text column that was not declared
    categorical would otherwise turn into a column of NaN, every row would fail
    listwise deletion, and the covariate would vanish from the model without a
    word — which is how `view_description` silently stopped being controlled for.

    Raises:
        SchemaError: when a non-null value cannot be read as a number.
    """
    if pd.api.types.is_bool_dtype(series) or series.dtype == "boolean":
        return series.astype("Float64").astype("float64")
    coerced = pd.to_numeric(series, errors="coerce")
    destroyed = series.notna() & coerced.isna()
    if destroyed.any():
        sample = series[destroyed].astype(str).unique()[:3].tolist()
        raise SchemaError(
            f"Covariate {name!r} is not numeric — {int(destroyed.sum())} values such as "
            f"{sample} cannot be read as numbers. Declare it in `categoricals` if it "
            "is a category."
        )
    return coerced


def _pool_rare(levels: pd.Series, min_count: int) -> tuple[pd.Series, int]:
    """Replace categorical levels below `min_count` with a shared residual level."""
    counts = levels.value_counts()
    rare = set(counts[counts < min_count].index)
    if not rare:
        return levels, 0
    pooled = levels.where(~levels.isin(rare), RARE_LEVEL)
    return pooled, len(rare)


def _dependent_columns(matrix: pd.DataFrame, tolerance: float = 1e-9) -> list[str]:
    """Columns that are exact linear combinations of earlier ones.

    Uses a column-pivoted QR: pivoting orders columns by how much new direction
    each adds, so everything past the numerical rank is redundant. Dropping
    those is not a modelling choice — a rank-deficient design has no unique
    solution, and both lifelines and statsmodels fail with an opaque "singular
    matrix" rather than saying which column caused it.

    Sparse real exports produce these routinely: after listwise deletion a
    submarket can end up containing exactly one season, and its dummy becomes a
    copy of that season's.
    """
    if matrix.shape[1] < 2:
        return []
    values = matrix.to_numpy(dtype="float64")
    _, r, pivots = scipy_qr(values, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return []
    rank = int((diagonal > tolerance * max(matrix.shape) * diagonal[0]).sum())
    if rank >= matrix.shape[1]:
        return []
    columns = list(matrix.columns)
    return sorted(columns[i] for i in pivots[rank:])


def build_design_matrix(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    categoricals: Sequence[str] = DEFAULT_CATEGORICALS,
    fixed_effects: Sequence[str] = (),
    min_level_count: int = MIN_LEVEL_COUNT,
    scale: bool = True,
) -> DesignMatrix:
    """Build a numeric model matrix from a feature frame.

    Args:
        frame: feature frame from `build_features`.
        covariates: covariate names; categoricals among them are expanded.
        categoricals: which of `covariates` are categorical.
        fixed_effects: extra categorical columns to absorb (e.g.
            `building_name`). They are not in `covariates` because they are
            nuisance controls, not quantities anyone reports.
        min_level_count: levels rarer than this are pooled into `__rare__`.
        scale: divide each column by its sd for conditioning. Coefficients are
            returned to raw units by the caller via `DesignMatrix.scale`.

    Returns:
        DesignMatrix whose `index` selects the surviving rows of `frame`.

    Raises:
        SchemaError: when a requested covariate is absent, or when listwise
            deletion leaves too few rows to fit anything.
    """
    requested = list(dict.fromkeys([*covariates, *fixed_effects]))
    missing = [c for c in requested if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"Design matrix requires columns absent from the feature frame: {missing}"
        )

    rows_in = len(frame)
    categorical_set = {c for c in categoricals if c in requested} | {
        c for c in fixed_effects if c in requested
    }
    numeric_names = [c for c in requested if c not in categorical_set]

    prepared: dict[str, pd.Series] = {}
    for name in numeric_names:
        prepared[name] = _to_numeric(name, frame[name])
    for name in sorted(categorical_set):
        column = frame[name]
        prepared[name] = column.where(column.notna()).astype("object")

    dropped_by_null = {
        name: int(series.isna().sum()) for name, series in prepared.items()
    }
    keep = pd.Series(True, index=frame.index)
    for series in prepared.values():
        keep &= series.notna()
    index = frame.index[keep]
    if len(index) < 2:
        raise SchemaError(
            f"Listwise deletion left {len(index)} of {rows_in} rows. "
            f"Null counts by covariate: {dropped_by_null}"
        )

    blocks: list[pd.DataFrame] = []
    notes: list[str] = []
    categorical_levels: dict[str, list[str]] = {}
    reference_levels: dict[str, str] = {}
    pooled_rare: dict[str, int] = {}

    numeric_block = pd.DataFrame(
        {name: prepared[name].loc[index].astype("float64") for name in numeric_names},
        index=index,
    )
    if not numeric_block.empty:
        blocks.append(numeric_block)

    for name in sorted(categorical_set):
        levels = prepared[name].loc[index].astype(str)
        levels, n_pooled = _pool_rare(levels, min_level_count)
        if n_pooled:
            pooled_rare[name] = n_pooled
            notes.append(
                f"{name}: pooled {n_pooled} levels with fewer than {min_level_count} "
                f"listings into {RARE_LEVEL}"
            )
        unique = sorted(levels.unique())
        if len(unique) < 2:
            notes.append(f"{name}: only one level present, dropped from the design")
            continue
        reference = unique[0]
        reference_levels[name] = reference
        categorical_levels[name] = unique[1:]
        dummies = pd.get_dummies(levels, prefix=name, prefix_sep="=", dtype="float64")
        dummies = dummies.drop(columns=[f"{name}={reference}"])
        blocks.append(dummies)

    if not blocks:
        raise SchemaError("Design matrix has no usable columns")
    matrix = pd.concat(blocks, axis=1)

    constant = [c for c in matrix.columns if float(matrix[c].std(ddof=0)) == 0.0]
    if constant:
        matrix = matrix.drop(columns=constant)
        notes.append(f"dropped {len(constant)} constant columns: {constant[:5]}")
    if matrix.empty or matrix.shape[1] == 0:
        raise SchemaError("Every design column was constant after row filtering")

    dependent = _dependent_columns(matrix)
    if dependent:
        matrix = matrix.drop(columns=dependent)
        notes.append(
            f"dropped {len(dependent)} linearly dependent columns: {dependent[:5]}. "
            "Listwise deletion can leave a category perfectly predicted by others; "
            "keeping them gives a singular information matrix and no fit at all."
        )

    scales = {c: 1.0 for c in matrix.columns}
    if scale:
        for column in matrix.columns:
            sd = float(matrix[column].std(ddof=0))
            if sd > 0:
                scales[column] = sd
                matrix[column] = matrix[column] / sd

    return DesignMatrix(
        X=matrix,
        index=index,
        rows_in=rows_in,
        rows_used=len(index),
        dropped_by_null=dropped_by_null,
        scale=scales,
        categorical_levels=categorical_levels,
        reference_levels=reference_levels,
        dropped_constant=constant,
        dropped_dependent=dependent,
        pooled_rare_levels=pooled_rare,
        notes=notes,
    )


def unscale_coefficients(
    raw: dict[str, dict[str, float]], design: DesignMatrix
) -> dict[str, Coefficient]:
    """Return coefficients to raw covariate units.

    `raw` maps design-column name to a dict with `value`, `std_error`,
    `ci_low`, `ci_high`, `p_value` as fitted on the scaled matrix. Because both
    estimators are linear in the covariates, dividing every location and scale
    statistic by the column's sd is an exact change of units; the p-value is a
    pivot and is unchanged.
    """
    out: dict[str, Coefficient] = {}
    for name, stats in raw.items():
        divisor = design.scale.get(name, 1.0) or 1.0
        out[name] = Coefficient(
            name=name,
            value=float(stats["value"]) / divisor,
            std_error=float(stats["std_error"]) / divisor,
            ci_low=float(stats["ci_low"]) / divisor,
            ci_high=float(stats["ci_high"]) / divisor,
            p_value=float(stats["p_value"]),
        )
    return out


def numeric_columns(design: DesignMatrix) -> list[str]:
    """Design columns that came from numeric covariates rather than dummies."""
    return [c for c in design.X.columns if "=" not in c]


def transform_to_design(
    features: pd.DataFrame, design: DesignMatrix
) -> tuple[pd.DataFrame, pd.Series]:
    """Re-encode a scoring frame into a fitted design's column space.

    Replays the reference levels, rare-level pooling, and column scaling fixed
    at fit time. Unseen categorical levels raise rather than falling back to the
    reference: scoring a Sunny Isles unit as if it sat in the reference
    submarket is a wrong answer wearing the right shape.

    Returns:
        (encoded matrix restricted to scorable rows, boolean mask over
        `features.index` marking which rows were scorable).
    """
    numeric = numeric_columns(design)
    categorical = sorted(set(design.reference_levels))
    missing = [c for c in (*numeric, *categorical) if c not in features.columns]
    if missing:
        raise SchemaError(f"Scoring frame is missing fitted covariates: {missing}")

    usable = pd.Series(True, index=features.index)
    encoded = pd.DataFrame(0.0, index=features.index, columns=design.X.columns)

    for name in numeric:
        values = pd.to_numeric(features[name], errors="coerce")
        usable &= values.notna()
        encoded[name] = values.astype("float64").fillna(0.0)

    for name in categorical:
        known = set(design.categorical_levels.get(name, []))
        reference = str(design.reference_levels[name])
        raw = features[name]
        usable &= raw.notna()
        levels = raw.astype(str)
        unknown = sorted(set(levels[raw.notna()].unique()) - (known | {reference}))
        if unknown:
            if RARE_LEVEL not in known:
                raise SchemaError(
                    f"{name} has levels never seen at fit time: {unknown[:5]}. "
                    "Refit the model or map them explicitly; they cannot be scored."
                )
            levels = levels.where(~levels.isin(unknown), RARE_LEVEL)
        for level in known:
            column = f"{name}={level}"
            if column in encoded.columns:
                encoded[column] = (levels == level).astype("float64")

    for column in encoded.columns:
        divisor = design.scale.get(column, 1.0) or 1.0
        encoded[column] = encoded[column] / divisor

    return encoded.loc[usable], usable


def design_summary(design: DesignMatrix) -> dict[str, Any]:
    """Serializable account of the sample the model actually saw."""
    return {
        "rows_in": design.rows_in,
        "rows_used": design.rows_used,
        "retention": round(design.retention, 4),
        "dropped_by_null": dict(sorted(design.dropped_by_null.items())),
        "n_design_columns": int(design.X.shape[1]),
        "dropped_dependent": design.dropped_dependent,
        "reference_levels": design.reference_levels,
        "pooled_rare_levels": design.pooled_rare_levels,
        "notes": design.notes,
    }
