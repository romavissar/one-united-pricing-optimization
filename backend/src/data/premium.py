"""The identification variable, respecified as a hedonic residual.

**What was wrong with the old one.** `rel_price_premium` was a listing's $/sqft
divided by the median $/sqft of its (submarket, month) cell, minus one. The cell
holds studios and penthouses alike, so the median controls for *where* and
*when* but not for *what*. On the old export 37% of that variable's variance was
explained by unit characteristics rather than by the seller's pricing decision.
A regressor that is 37% something else is a regressor measured with error, and
classical errors-in-variables attenuates its coefficient toward zero by roughly
the signal share — at any sample size. No amount of extra data fixes it, which
is why the coefficient barely moved when the sample grew.

**What this replaces it with.** Regress log $/sqft on everything about the unit
that a buyer can see, plus submarket and month fixed effects, and define the
premium as what is left:

    log(list_ppsf) = f(unit characteristics) + submarket + month + premium

The residual is orthogonal to every control by construction, so anything the
hedonic can explain is no longer inside the identifying variable. What remains
is the part of the asking price that the unit does not account for — which is
the seller's pricing choice, plus whatever quality no column captures.

**Three rules govern what may enter the right-hand side.**

*Nothing post-listing.* DOM, CDOM, Current Price, Sale Price, and every terminal
date are functions of the outcome. Putting any of them here would launder the
outcome into the regressor and produce a beautiful, worthless coefficient.

*Nothing about the seller.* Terms Considered, Occupancy Information, and
Special Information are available at listing time and are not outcome
variables, but they describe the seller's situation rather than the asset. They
are excluded deliberately: the residual is *supposed* to contain seller
behaviour, and a control that absorbs seller motivation strips out the exact
variation `beta_price` is meant to measure. Excluding them is not an oversight
and adding them would look like an improvement while making the estimate worse.

*Fit on every listing, not just the sold ones.* This is a model of what sellers
ask, not of what buyers pay. Restricting to closed sales would select the
sample on the outcome. That is the opposite choice from `demand/hedonic.py`,
which fits `close_ppsf` on sold listings only because it answers a different
question — what a unit is worth — and feeds the optimizer's price bounds.

Units: `reference_ppsf` is $/sqft. `premium_log` is log points.
`premium_ratio` is a ratio minus one, the same convention the old variable used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

# Physical and building attributes only. Anything describing the seller's
# situation is deliberately absent — see the module docstring.
HEDONIC_NUMERIC: tuple[str, ...] = (
    "log_living_area",
    "log_floor",
    "beds",
    "baths_full",
    "hoa_per_sqft",
    "building_age_years",
    "total_stories",
    "min_lease_days_value",
    "min_lease_days_missing",
)
HEDONIC_CATEGORICAL: tuple[str, ...] = (
    "submarket",
    "list_month",
    "furnished_info_level",
    "special_assessment_level",
    "association_type_level",
)
# Prefixes of the tokenised multi-value blocks built in `features.py`.
HEDONIC_TOKEN_PREFIXES: tuple[str, ...] = ("view_", "wf_", "park_", "rest_")

_MIN_ROWS = 200
_MIN_LEVEL_COUNT = 20


@dataclass
class PremiumFit:
    """The fitted list-price hedonic and what it explains."""

    r_squared: float
    r_squared_adj: float
    residual_sd: float
    n_observations: int
    n_parameters: int
    total_log_variance: float
    columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def explained_share(self) -> float:
        """Share of log-price variance the unit and its location/time explain."""
        return float(self.r_squared)

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_squared": self.r_squared,
            "r_squared_adj": self.r_squared_adj,
            "residual_sd_log": self.residual_sd,
            "residual_variance_log": self.residual_sd ** 2,
            "total_log_variance": self.total_log_variance,
            "n_observations": self.n_observations,
            "n_parameters": self.n_parameters,
            "notes": self.notes,
        }


@dataclass
class PremiumModel:
    """A fitted list-price hedonic that can price a unit it has not seen.

    The optimizer needs this. `beta_price` is now the coefficient on a residual
    against *this* surface, so at scoring time a candidate price has to be
    turned into a premium against the same reference the fit used. Handing the
    demand model a premium measured against anything else — a submarket median,
    a different hedonic — silently changes the units of the coefficient it is
    multiplying.
    """

    model: Any = field(repr=False)
    columns: list[str] = field(default_factory=list)
    vocabulary: DesignVocabulary = field(default_factory=lambda: DesignVocabulary())

    def predict_reference_ppsf(self, units: pd.DataFrame) -> np.ndarray:
        """Predicted $/sqft per row, replaying the fit-time encoding exactly."""
        design, _notes, _vocab = hedonic_design(units, self.vocabulary)
        aligned = pd.DataFrame(0.0, index=units.index, columns=self.columns)
        for column in self.columns:
            if column == "const":
                aligned[column] = 1.0
            elif column in design.columns:
                aligned[column] = pd.to_numeric(design[column], errors="coerce")
        aligned = aligned.fillna(0.0)
        return np.exp(np.asarray(self.model.predict(aligned.to_numpy(dtype="float64"))))


@dataclass
class PremiumResult:
    """Per-row premium plus the fit that produced it."""

    premium_log: pd.Series
    premium_ratio: pd.Series
    reference_ppsf: pd.Series
    source: pd.Series
    fit: PremiumFit
    model: PremiumModel | None = None


@dataclass
class DesignVocabulary:
    """Every choice the fit made about how to encode a row.

    Recomputing these at scoring time is the classic train/predict encoding
    bug: which categorical levels are rare enough to pool, which level is the
    dropped reference, and what a missing numeric is filled with all depend on
    the sample they are computed over. Derive them from a 200-row scoring frame
    and the same listing gets a different design row than it had at fit time,
    so the prediction is wrong while looking entirely normal. Frozen here and
    replayed verbatim.
    """

    numeric_fill: dict[str, float] = field(default_factory=dict)
    numeric_has_isna: set[str] = field(default_factory=set)
    token_columns: list[str] = field(default_factory=list)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    categorical_reference: dict[str, str] = field(default_factory=dict)
    categorical_rare: dict[str, set[str]] = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)


def hedonic_design(
    frame: pd.DataFrame, vocabulary: DesignVocabulary | None = None
) -> tuple[pd.DataFrame, list[str], DesignVocabulary]:
    """Assemble the right-hand side, with missingness encoded rather than dropped.

    A numeric control that is missing would delete the row under listwise
    deletion. Since the whole point is to keep the sample, each numeric column
    is mean-filled and paired with a `*_isna` indicator, so the fit uses the
    information where it exists and the indicator absorbs the rest. That is a
    within-hedonic imputation of a *control*, not of the identifying variable —
    `AGENTS.md` §2 forbids quietly imputing data, and nothing here is imputed
    into the premium itself or into any reported quantity.
    """
    notes: list[str] = []
    blocks: list[pd.DataFrame] = []
    vocab = vocabulary or DesignVocabulary()
    learning = vocabulary is None

    numeric = [c for c in HEDONIC_NUMERIC if c in frame.columns]
    if learning:
        block = pd.DataFrame(index=frame.index)
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            missing = values.isna()
            if missing.all():
                notes.append(f"{column}: all-null, dropped")
                continue
            vocab.numeric_fill[column] = float(values.mean())
            block[column] = values.fillna(vocab.numeric_fill[column])
            if missing.any():
                vocab.numeric_has_isna.add(column)
                block[f"{column}_isna"] = missing.astype("float64")
        blocks.append(block)
    else:
        block = pd.DataFrame(index=frame.index)
        for column, fill in vocab.numeric_fill.items():
            values = (
                pd.to_numeric(frame[column], errors="coerce")
                if column in frame.columns
                else pd.Series(np.nan, index=frame.index, dtype="float64")
            )
            missing = values.isna()
            block[column] = values.fillna(fill)
            if column in vocab.numeric_has_isna:
                block[f"{column}_isna"] = missing.astype("float64")
        blocks.append(block)

    if learning:
        vocab.token_columns = [
            c
            for c in frame.columns
            if any(c.startswith(p) for p in HEDONIC_TOKEN_PREFIXES)
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
    if vocab.token_columns:
        tokens = pd.DataFrame(0.0, index=frame.index, columns=vocab.token_columns)
        for column in vocab.token_columns:
            if column in frame.columns:
                tokens[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        blocks.append(tokens)

    for column in HEDONIC_CATEGORICAL:
        if learning and column not in frame.columns:
            continue
        if not learning and column not in vocab.categorical_levels:
            continue
        raw = (
            frame[column].astype("string").fillna("__missing__")
            if column in frame.columns
            else pd.Series("__missing__", index=frame.index, dtype="string")
        )
        if learning:
            counts = raw.value_counts()
            rare = set(counts[counts < _MIN_LEVEL_COUNT].index)
            levels = raw.where(~raw.isin(rare), "__rare__") if rare else raw
            if rare:
                notes.append(
                    f"{column}: pooled {len(rare)} levels under {_MIN_LEVEL_COUNT}"
                )
            unique = sorted(levels.unique())
            if len(unique) < 2:
                notes.append(f"{column}: single level, dropped")
                continue
            vocab.categorical_rare[column] = rare
            # First level is the dropped reference, exactly as at fit time.
            vocab.categorical_reference[column] = unique[0]
            vocab.categorical_levels[column] = unique[1:]
        else:
            rare = vocab.categorical_rare.get(column, set())
            levels = raw.where(~raw.isin(rare), "__rare__") if rare else raw
            # The dropped reference is a level the fit knew about even though it
            # has no column; leaving it out of `known` would divert every
            # reference row into the pooled bucket and give it the wrong price.
            known = set(vocab.categorical_levels[column]) | {
                vocab.categorical_reference.get(column, "")
            }
            # A level never seen at fit time joins the pooled bucket rather than
            # silently landing on the reference, which would price it as
            # whichever level happened to sort first.
            levels = levels.where(levels.isin(known | {"__rare__"}), "__rare__")

        dummies = pd.DataFrame(0.0, index=frame.index,
                               columns=[f"{column}={lv}" for lv in vocab.categorical_levels[column]])
        for level in vocab.categorical_levels[column]:
            dummies[f"{column}={level}"] = (levels == level).astype("float64")
        blocks.append(dummies)

    if not blocks:
        raise SchemaError("The list-price hedonic has no usable right-hand side")
    design = pd.concat(blocks, axis=1)

    if learning:
        vocab.dropped_constant = [
            c for c in design.columns if float(design[c].std(ddof=0)) == 0.0
        ]
        if vocab.dropped_constant:
            notes.append(f"dropped {len(vocab.dropped_constant)} constant columns")
    if vocab.dropped_constant:
        design = design.drop(columns=vocab.dropped_constant, errors="ignore")
    return design, notes, vocab


def fit_price_premium(frame: pd.DataFrame) -> PremiumResult:
    """Fit log(list_ppsf) on unit, submarket and month; return the residual.

    Raises:
        SchemaError: when `list_ppsf` is absent or too few rows are usable.
        IdentificationError: when the design is singular.
    """
    if "list_ppsf" not in frame.columns:
        raise SchemaError("The list-price hedonic requires list_ppsf ($/sqft)")

    ppsf = pd.to_numeric(frame["list_ppsf"], errors="coerce")
    usable = ppsf.notna() & (ppsf > 0)
    if int(usable.sum()) < _MIN_ROWS:
        raise SchemaError(
            f"The list-price hedonic needs at least {_MIN_ROWS} listings with a "
            f"usable $/sqft, found {int(usable.sum())}"
        )

    design, notes, vocabulary = hedonic_design(frame)
    design = design.loc[usable]
    y = np.log(ppsf[usable].to_numpy(dtype="float64"))

    X = sm.add_constant(design, has_constant="add")
    try:
        model = sm.OLS(y, X.to_numpy(dtype="float64")).fit()
    except np.linalg.LinAlgError as exc:
        raise IdentificationError(
            f"The list-price hedonic is singular — collinear controls. Error: {exc}"
        ) from exc

    resid = pd.Series(np.nan, index=frame.index, dtype="float64")
    resid.loc[usable] = model.resid
    reference = pd.Series(np.nan, index=frame.index, dtype="float64")
    reference.loc[usable] = np.exp(model.fittedvalues)

    source = pd.Series("missing", index=frame.index, dtype="object")
    source.loc[usable] = "hedonic_residual"

    fit = PremiumFit(
        r_squared=float(model.rsquared),
        r_squared_adj=float(model.rsquared_adj),
        residual_sd=float(np.std(model.resid, ddof=1)),
        n_observations=int(model.nobs),
        n_parameters=int(X.shape[1]),
        total_log_variance=float(np.var(y, ddof=1)),
        columns=list(X.columns) if hasattr(X, "columns") else [],
        notes=notes,
    )
    logger.info(
        "List-price hedonic: n=%d R2=%.4f residual sd=%.4f (log points) over %d parameters",
        fit.n_observations, fit.r_squared, fit.residual_sd, fit.n_parameters,
    )
    premium_model = PremiumModel(
        model=model, columns=["const", *design.columns], vocabulary=vocabulary
    )
    return PremiumResult(
        premium_log=resid,
        # The same quantity as a ratio minus one, so it reads on the scale the
        # project has always used: 0.10 is "priced 10% above what this unit,
        # here, this month, predicts".
        premium_ratio=np.expm1(resid),
        reference_ppsf=reference,
        source=source,
        fit=fit,
        model=premium_model,
    )
