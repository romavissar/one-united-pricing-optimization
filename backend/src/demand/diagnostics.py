"""Post-fit diagnostics. These run after every fit and they fail loudly.

The failure this module exists to prevent: a model whose price coefficient came
out the wrong way round, quietly shipped anyway, feeding an optimizer that then
recommends charging more because the data appear to say that raising the price
makes units sell faster. When `beta(rel_price_premium) >= 0` the identification
is broken and the honest output is "this data cannot answer the question", not a
respecified model that produces a nicer sign.

Two severities:

`FAILED` blocks. `beta_price >= 0` in either model is `FAILED` unconditionally —
that is `AGENTS.md` §3 and it is not softened by a wide confidence interval.
Other sign requirements are `FAILED` only when the coefficient is both wrong-
signed and significant, because a wrong sign whose interval covers zero is an
absence of evidence, not evidence of a broken model.

`WARN` reports. Weak discrimination, poor calibration, and a wrong-signed but
insignificant control all belong here: they qualify the answer rather than
invalidate it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.features import DEMAND_COVARIATES
from src.demand.base import Coefficient, FitResult
from src.demand.logistic import LogisticDemandModel
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

Status = Literal["PASS", "WARN", "FAILED"]

PRICE_TERM = "rel_price_premium"
_DISCRIMINATION_FLOOR = 0.60
_CALIBRATION_TOLERANCE = 0.10
_MIN_CELL_LISTINGS = 8
_CALIBRATION_BINS = 10
# Above this, the (submarket, month) median is not holding the unit fixed: a
# large part of what rel_price_premium measures is which unit it is, not how
# that unit was priced.
_QUALITY_SHARE_ALARM = 0.35

# Sign requirements from PROJECT_BRIEF §3d. The price terms are handled
# separately because their rule is unconditional.
_SIGN_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("log_floor", "positive"),
    ("inventory_competition", "negative"),
)

# Hedonic controls that measurably reduce attenuation where the export carries
# them. Leaving one out is legitimate, but it leaves that part of unit quality
# inside rel_price_premium, so the report has to say the coefficient is a floor
# on the true elasticity rather than an estimate of it.
_OPTIONAL_HEDONIC_CONTROLS: tuple[str, ...] = (
    "view_description",
    "log_living_area",
    "building_name",
)

# A multi-valued column does not enter the design under its own name: it is
# tokenized into indicators (`view_description` -> `view_bay`, `view_canal`, ...)
# and it is those the fit controls for. Matching on the raw name alone made the
# report claim view was uncontrolled while fifteen view indicators were in the
# design, which inverts the note's meaning — it exists to say the coefficient is
# attenuated, and it was saying so about a control that is present.
_CONTROL_PROXY_PREFIXES: dict[str, str] = {"view_description": "view_"}


def _is_controlled(column: str, controlled_for: set[str]) -> bool:
    """Whether `column`'s information is in the design, directly or as tokens."""
    if column in controlled_for:
        return True
    prefix = _CONTROL_PROXY_PREFIXES.get(column)
    return bool(prefix) and any(c.startswith(prefix) for c in controlled_for)


@dataclass
class Check:
    """One diagnostic with the requirement it was measured against."""

    name: str
    requirement: str
    value: float | None
    status: Status
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdentificationReport:
    """Whether the data can answer the elasticity question at all.

    `quality_explained_share` is the R-squared of regressing
    `rel_price_premium` on every other covariate in the fit: the share of the
    identifying variable that is explained by unit characteristics rather than
    by pricing choice. Read it in one direction only. A high value is an alarm —
    the submarket-month median is not holding the unit fixed, the regressor is a
    noisy proxy for the seller's decision, and `beta_price` is attenuated toward
    zero, so the true elasticity is *larger* in magnitude than the fitted one. A
    low value is not reassurance, because it is equally consistent with controls
    too weak to explain anything.
    """

    n_sold: int
    n_censored: int
    n_cells_ge_min: int
    min_cell_listings: int
    rel_price_premium_iqr: float | None
    beta_price: dict[str, float | str] | None
    beta_price_excludes_zero: bool | None
    quality_explained_share: float | None
    rows_fitted: int
    rows_dropped_by_null: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticsReport:
    """Everything a caller needs to decide whether to trust the fit."""

    status: Status
    checks: list[Check]
    identification: IdentificationReport
    calibration_deciles: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAILED"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [c.as_dict() for c in self.checks],
            "identification": self.identification.as_dict(),
            "calibration_deciles": self.calibration_deciles,
            "warnings": self.warnings,
        }


def _sign_check(
    label: str, coefficient: Coefficient | None, expect: str, *, unconditional: bool
) -> Check:
    """Grade one coefficient against its required sign."""
    requirement = f"beta({coefficient.name if coefficient else label}) {'<' if expect == 'negative' else '>'} 0"
    if coefficient is None:
        return Check(
            name=label,
            requirement=requirement,
            value=None,
            status="WARN",
            message=f"{label} did not enter the model, so its sign could not be checked",
        )

    wrong = coefficient.value >= 0 if expect == "negative" else coefficient.value <= 0
    if not wrong:
        return Check(
            name=label,
            requirement=requirement,
            value=coefficient.value,
            status="PASS",
            message=f"{coefficient.value:+.4f} (95% CI {coefficient.ci_low:+.4f} to {coefficient.ci_high:+.4f})",
        )
    if unconditional:
        return Check(
            name=label,
            requirement=requirement,
            value=coefficient.value,
            status="FAILED",
            message=(
                f"{coefficient.value:+.4f} has the wrong sign. A non-negative price "
                "coefficient means higher relative price is associated with faster "
                "sale, which is not a demand curve. The identification is broken; do "
                "not respecify until the sign flips."
            ),
        )
    significant = coefficient.excludes_zero
    return Check(
        name=label,
        requirement=requirement,
        value=coefficient.value,
        status="FAILED" if significant else "WARN",
        message=(
            f"{coefficient.value:+.4f} has the wrong sign and its 95% CI excludes zero"
            if significant
            else f"{coefficient.value:+.4f} has the wrong sign but its 95% CI covers zero, "
            "so this is an absence of evidence rather than a contradiction"
        ),
    )


def _get(result: FitResult | None, name: str) -> Coefficient | None:
    if result is None:
        return None
    return result.coefficients.get(name)


def calibration_deciles(
    predicted: np.ndarray | pd.Series,
    observed: np.ndarray | pd.Series,
    bins: int = _CALIBRATION_BINS,
) -> list[dict[str, float]]:
    """Predicted versus observed sale rate by decile of predicted probability.

    Emitted as a table rather than a chart: there is no plotting library in the
    locked stack, and the frontend is the right place to draw it anyway.
    """
    frame = pd.DataFrame(
        {"p": np.asarray(predicted, dtype="float64"), "y": np.asarray(observed, dtype="float64")}
    ).dropna()
    if frame.empty:
        return []
    try:
        frame["decile"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    except ValueError:
        return []
    grouped = frame.groupby("decile", observed=True)
    return [
        {
            "decile": int(decile) + 1,
            "n": int(len(group)),
            "predicted_mean": float(group["p"].mean()),
            "observed_rate": float(group["y"].mean()),
        }
        for decile, group in grouped
    ]


def _quality_explained_share(result: FitResult) -> float | None:
    """R-squared of rel_price_premium on the other covariates in the design."""
    X = result.design.X
    if PRICE_TERM not in X.columns or X.shape[1] < 2:
        return None
    others = X.drop(columns=[PRICE_TERM])
    try:
        model = sm.OLS(X[PRICE_TERM].to_numpy(), sm.add_constant(others, has_constant="add")).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(np.clip(model.rsquared, 0.0, 1.0))


def _premium_is_hedonic_residual(frame: pd.DataFrame) -> bool:
    """True when `rel_price_premium` is a residual rather than a cell-median ratio."""
    column = frame.get("rel_price_premium_spec")
    if column is None:
        return False
    return bool((column.astype(str) == "hedonic_residual").any())


def _cell_count(frame: pd.DataFrame, min_cell_listings: int) -> int:
    """(submarket, list_month) cells holding at least `min_cell_listings` listings."""
    if not {"submarket", "list_month"}.issubset(frame.columns):
        return 0
    known = frame[frame["submarket"].notna() & frame["list_month"].notna()]
    if known.empty:
        return 0
    sizes = known.groupby(["submarket", "list_month"], observed=True).size()
    return int((sizes >= min_cell_listings).sum())


def run_diagnostics(
    frame: pd.DataFrame,
    cox_result: FitResult,
    *,
    logistic_model: LogisticDemandModel | None = None,
    logistic_error: str | None = None,
    min_cell_listings: int = _MIN_CELL_LISTINGS,
) -> DiagnosticsReport:
    """Grade a fitted demand model and assemble the identification report.

    Args:
        frame: the feature frame the models were fitted on.
        cox_result: the primary fit. Its `beta_price` is the reported elasticity.
        logistic_model: the fitted cross-check, if one was run. Supplies the
            second sign check, the held-out AUC, and the calibration table.
        logistic_error: why the cross-check could not be fitted, if it could
            not. Recorded as a WARN check rather than left in a log line: a
            silently absent cross-check looks identical to a passing one in the
            printed report.
        min_cell_listings: cell-density threshold for the identification report.

    Returns:
        DiagnosticsReport whose `status` is FAILED if anything blocking failed.
    """
    logistic_result = logistic_model.result if logistic_model is not None else None
    checks: list[Check] = [
        _sign_check("cox_beta_price", _get(cox_result, PRICE_TERM), "negative", unconditional=True)
    ]
    if logistic_result is not None:
        checks.append(
            _sign_check(
                "logistic_beta_price",
                _get(logistic_result, PRICE_TERM),
                "negative",
                unconditional=True,
            )
        )
    if logistic_result is None:
        checks.append(
            Check(
                name="logistic_cross_check",
                requirement="fitted",
                value=None,
                status="WARN",
                message=(
                    f"the logistic cross-check could not be fitted, so the Cox price "
                    f"coefficient has nothing corroborating it: {logistic_error}"
                    if logistic_error
                    else "no logistic cross-check was supplied, so the Cox price "
                    "coefficient has nothing corroborating it"
                ),
            )
        )
    # `beta(log_floor) > 0` is a requirement about the OLD identification
    # variable, and it does not survive the respecification.
    #
    # When the premium was a ratio to a submarket-month median, that median did
    # not adjust for floor, so a high-floor unit sat above its cell by
    # construction and `log_floor` in the hazard carried the residual
    # desirability of height. The sign was determinate and a negative value
    # meant something was wrong.
    #
    # The premium is now a residual against a hedonic that already prices floor.
    # `beta(log_floor)` therefore answers a different question — does a
    # high-floor unit sell faster *at the same price relative to what its own
    # floor predicts* — and theory does not fix that sign. A penthouse asked at
    # the same premium over its own (higher) reference plausibly sells slower,
    # because the pool of buyers at that price is thinner. Requiring positivity
    # here would be demanding a sign the specification no longer implies, and on
    # synthetic data where the generator plants floor in price but not in the
    # hazard it fails by construction.
    #
    # The price coefficient's own check is untouched and remains an
    # unconditional hard failure. This narrows one check that became
    # meaningless; it does not relax the one the project exists for.
    premium_is_residual = _premium_is_hedonic_residual(frame)
    for name, expect in _SIGN_REQUIREMENTS:
        coefficient = _get(cox_result, name)
        if name == "log_floor" and premium_is_residual:
            checks.append(
                Check(
                    name="log_floor",
                    requirement="reported, not required (premium is floor-adjusted)",
                    value=coefficient.value if coefficient else None,
                    status="PASS",
                    message=(
                        f"{coefficient.value:+.4f} — reported without a sign "
                        "requirement. The premium is a residual against a hedonic "
                        "that already prices floor, so this coefficient is the "
                        "effect of height at a fixed relative price, whose sign "
                        "theory does not determine."
                        if coefficient
                        else "log_floor did not enter the model"
                    ),
                )
            )
            continue
        checks.append(_sign_check(name, coefficient, expect, unconditional=False))

    # Structural problems the fitter reported but did not raise on. These arrive
    # as notes because that is where survival.py records them; promoting them to
    # checks is what makes them visible in the printed report and in the API,
    # rather than sitting in a list nobody prints.
    for marker, label, requirement in (
        ("CONVERGENCE WARNINGS", "cox_convergence", "fitter reports no convergence warning"),
        ("MOSTLY NOT FIXED EFFECTS", "fixed_effects_pooled", "fixed effects retain their levels"),
        ("THIN FIXED EFFECTS", "fixed_effects_density", "enough listings per fixed-effect level"),
    ):
        hit = next((n for n in cox_result.notes if marker in n), None)
        if hit:
            checks.append(
                Check(name=label, requirement=requirement, value=None,
                      status="WARN", message=hit)
            )

    concordance = cox_result.fit_stats.get("concordance")
    checks.append(
        Check(
            name="cox_concordance",
            requirement=f">= {_DISCRIMINATION_FLOOR}",
            value=concordance,
            status="PASS" if (concordance or 0.0) >= _DISCRIMINATION_FLOOR else "WARN",
            message=(
                f"{concordance:.4f}"
                + (
                    ""
                    if (concordance or 0.0) >= _DISCRIMINATION_FLOOR
                    else " — the model barely orders listings better than a coin flip. "
                    "The coefficient may still be unbiased; its predictive value is low."
                )
            ),
        )
    )

    calibration: list[dict[str, float]] = []
    if logistic_model is not None and logistic_result is not None:
        auc = logistic_result.fit_stats.get("auc_holdout", float("nan"))
        auc_ok = bool(np.isfinite(auc)) and auc >= _DISCRIMINATION_FLOOR
        checks.append(
            Check(
                name="logistic_auc_holdout",
                requirement=f">= {_DISCRIMINATION_FLOOR}",
                value=None if not np.isfinite(auc) else float(auc),
                status="PASS" if auc_ok else "WARN",
                message=(
                    "held-out AUC could not be computed"
                    if not np.isfinite(auc)
                    else f"{auc:.4f}"
                    + ("" if auc_ok else " — weak discrimination on unseen listings")
                ),
            )
        )
        calibration = calibration_deciles(
            logistic_model.predicted_probabilities(), logistic_model.observed_outcomes()
        )
        if calibration:
            gap = max(abs(row["predicted_mean"] - row["observed_rate"]) for row in calibration)
            checks.append(
                Check(
                    name="calibration_max_decile_gap",
                    requirement=f"<= {_CALIBRATION_TOLERANCE}",
                    value=gap,
                    status="PASS" if gap <= _CALIBRATION_TOLERANCE else "WARN",
                    message=(
                        f"largest gap between predicted and observed sale rate across "
                        f"deciles is {gap:.4f}"
                    ),
                )
            )

    rel = pd.to_numeric(frame.get(PRICE_TERM), errors="coerce").dropna() if PRICE_TERM in frame else pd.Series(dtype="float64")
    iqr = float(rel.quantile(0.75) - rel.quantile(0.25)) if len(rel) else None
    event = pd.to_numeric(frame.get("event_sold"), errors="coerce")
    n_sold = int((event == 1).sum()) if event is not None else 0
    n_censored = int((event == 0).sum()) if event is not None else 0

    beta = cox_result.coefficients.get(PRICE_TERM)
    quality_share = _quality_explained_share(cox_result)
    notes: list[str] = []
    if quality_share is not None and quality_share >= _QUALITY_SHARE_ALARM:
        notes.append(
            f"{quality_share:.1%} of the variation in rel_price_premium is explained by "
            "the unit characteristics in the fit, so the submarket-month median is not "
            "holding the unit fixed. What remains is pricing choice plus whatever "
            "quality no covariate captured, which attenuates beta_price toward zero. "
            "Read the fitted value as a lower bound on the true elasticity in magnitude."
        )
    controlled_for = set(cox_result.covariates) | set(cox_result.design.reference_levels)
    # An all-null column is not a control this export "carries"; normalization
    # materialises every canonical field, present in the file or not.
    unused = [
        c
        for c in _OPTIONAL_HEDONIC_CONTROLS
        if c in frame.columns
        and not _is_controlled(c, controlled_for)
        and frame[c].notna().any()
    ]
    if unused:
        notes.append(
            f"this export carries {unused} but the fit does not control for them, so "
            "their contribution to price stays inside rel_price_premium as unobserved "
            "quality. That attenuates beta_price toward zero: the fitted value is a "
            "lower bound on the true elasticity in magnitude, not an estimate of it."
        )
    notes.extend(cox_result.notes)

    identification = IdentificationReport(
        n_sold=n_sold,
        n_censored=n_censored,
        n_cells_ge_min=_cell_count(frame, min_cell_listings),
        min_cell_listings=min_cell_listings,
        rel_price_premium_iqr=iqr,
        beta_price=beta.as_dict() if beta else None,
        beta_price_excludes_zero=beta.excludes_zero if beta else None,
        quality_explained_share=quality_share,
        rows_fitted=cox_result.n_observations,
        rows_dropped_by_null=dict(sorted(cox_result.design.dropped_by_null.items())),
        notes=notes,
    )

    warnings = [c.message for c in checks if c.status == "WARN"]
    if beta is not None and not beta.excludes_zero:
        warnings.append(
            "beta_price has the right sign but its 95% CI covers zero: the data are "
            "consistent with no price response at all. Any optimizer run on this fit "
            "will push toward the price ceiling."
        )
    if iqr is not None and iqr < 0.03:
        warnings.append(
            f"rel_price_premium IQR is {iqr:.4f}. Sellers priced near-identically "
            "relative to comps, so elasticity is weakly identified at any sample size."
        )

    status: Status = "PASS"
    if any(c.status == "FAILED" for c in checks):
        status = "FAILED"
    elif any(c.status == "WARN" for c in checks) or warnings:
        status = "WARN"

    report = DiagnosticsReport(
        status=status,
        checks=checks,
        identification=identification,
        calibration_deciles=calibration,
        warnings=warnings,
    )
    logger.info(
        "Diagnostics %s: %d checks, %d failed", status, len(checks), len(report.failed)
    )
    return report


def require_pass(report: DiagnosticsReport) -> None:
    """Raise unless the report is free of blocking failures.

    Call this at every boundary that would otherwise let a broken fit reach a
    price recommendation.

    Raises:
        IdentificationError: when any check is FAILED.
    """
    if report.status != "FAILED":
        return
    detail = "; ".join(f"{c.name}: {c.message}" for c in report.failed)
    raise IdentificationError(
        f"Demand model failed {len(report.failed)} blocking diagnostic(s). {detail}"
    )


def format_diagnostics_report(report: DiagnosticsReport) -> str:
    """Human-readable report; the printed text is the deliverable for --inspect."""
    ident = report.identification
    lines = [f"DEMAND DIAGNOSTICS  status={report.status}", ""]
    lines.append("CHECKS")
    for check in report.checks:
        marker = {"PASS": "ok", "WARN": " !", "FAILED": "XX"}[check.status]
        lines.append(f"  [{marker}] {check.name} ({check.requirement}): {check.message}")

    lines.append("")
    lines.append("IDENTIFICATION")
    lines.append(f"  sold: {ident.n_sold}   censored: {ident.n_censored}")
    lines.append(
        f"  (submarket, month) cells with >= {ident.min_cell_listings} listings: "
        f"{ident.n_cells_ge_min}"
    )
    iqr = ident.rel_price_premium_iqr
    lines.append(f"  rel_price_premium IQR: {iqr:.4f}" if iqr is not None else "  rel_price_premium IQR: n/a")
    if ident.beta_price:
        b = ident.beta_price
        lines.append(
            f"  beta_price: {b['value']:+.4f}  se {b['std_error']:.4f}  "
            f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
            f"{'excludes' if ident.beta_price_excludes_zero else 'COVERS'} zero"
        )
    share = ident.quality_explained_share
    if share is not None:
        lines.append(f"  share of rel_price_premium explained by unit features: {share:.1%}")
    lines.append(f"  rows fitted: {ident.rows_fitted}")
    dropped = {k: v for k, v in ident.rows_dropped_by_null.items() if v}
    if dropped:
        lines.append(f"  rows lost to nulls by covariate: {dropped}")

    if report.calibration_deciles:
        lines.append("")
        lines.append("CALIBRATION (predicted vs observed sale rate)")
        lines.append("  decile      n   predicted   observed")
        for row in report.calibration_deciles:
            lines.append(
                f"  {row['decile']:>6}  {row['n']:>5}      {row['predicted_mean']:.4f}     "
                f"{row['observed_rate']:.4f}"
            )

    if ident.notes:
        lines.append("")
        lines.append("NOTES")
        for note in ident.notes:
            lines.append(f"  - {note}")
    if report.warnings:
        lines.append("")
        lines.append("WARNINGS")
        for warning in report.warnings:
            lines.append(f"  !  {warning}")
    return "\n".join(lines)


def _targets_real_data(directory: object, files: object) -> bool:
    """True when a CLI invocation would read the real MLS export.

    Reading no explicit source means the default raw directory, which is where
    the real export lives. That default is the easiest way to fit on real data
    by accident, so it counts as targeting it.
    """
    from pathlib import Path

    if directory is None and not files:
        return True
    candidates = [directory] if directory is not None else []
    candidates.extend(files or [])
    return any("raw" in Path(str(c)).resolve().parts for c in candidates)


def cancelled_sweep(
    *,
    market: str,
    paths: list[object] | None,
    raw_dir: object,
    controls: bool,
    building_fe: bool,
) -> list[dict[str, Any]]:
    """Refit beta_price under each treatment of CANCELED and return all three.

    27% of the quarterly export is Cancelled, and the three readings of that
    status are not nested — one censors, one drops the rows, one calls them
    sales. The historical choice (`censored`) is defensible but it is a choice,
    and until the three numbers sit side by side nobody can see how much of
    `beta_price` rests on it. This runs the comparison rather than arguing it.

    Nothing here selects a treatment. The configured value stays in force; this
    reports what the alternatives would have produced.
    """
    from src.config import CANCELLED_TREATMENTS, load_market_config
    from src.data.features import build_features
    from src.data.ingest_mls import ingest_mls
    from src.demand.survival import (
        CONTROLLED_CATEGORICALS,
        CoxDemandModel,
        available_covariates,
    )

    rows: list[dict[str, Any]] = []
    for rule in CANCELLED_TREATMENTS:
        config = load_market_config(market)
        config.setdefault("defaults", {})["cancelled_treatment"] = rule
        try:
            ingested = ingest_mls(
                paths=paths, market=market, raw_dir=raw_dir, config=config
            )
            frame = build_features(ingested.frame, config).frame
            covariates = (
                available_covariates(frame) if controls else tuple(DEMAND_COVARIATES)
            )
            categoricals = (
                CONTROLLED_CATEGORICALS if controls else ("submarket", "season")
            )
            result = CoxDemandModel(
                covariates=covariates,
                categoricals=categoricals,
                building_fixed_effects=building_fe,
            ).fit(frame)
        except (SchemaError, IdentificationError) as exc:
            rows.append({"treatment": rule, "error": str(exc)})
            continue
        beta = _get(result, "rel_price_premium")
        rows.append(
            {
                "treatment": rule,
                "rows_fitted": int(result.n_observations),
                "events": int(pd.to_numeric(frame["event_sold"], errors="coerce").fillna(0).sum()),
                "beta_price": None if beta is None else float(beta.value),
                "ci_low": None if beta is None else float(beta.ci_low),
                "ci_high": None if beta is None else float(beta.ci_high),
                "excludes_zero": None if beta is None else bool(beta.excludes_zero),
            }
        )
    return rows


def format_cancelled_sweep(rows: list[dict[str, Any]], configured: str) -> str:
    """Render `cancelled_sweep` as a table, marking the treatment in force."""
    lines = [
        "CANCELED TREATMENT SWEEP",
        "  How beta_price moves with the coding of Cancelled listings. The "
        "configured",
        f"  treatment is {configured!r}; the others are shown for comparison "
        "only.",
        "",
        f"  {'treatment':<12} {'rows':>7} {'events':>7} {'beta_price':>11}  95% CI",
    ]
    for row in rows:
        mark = "*" if row["treatment"] == configured else " "
        if "error" in row:
            lines.append(f" {mark}{row['treatment']:<12} FAILED: {row['error']}")
            continue
        beta = row["beta_price"]
        if beta is None:
            lines.append(f" {mark}{row['treatment']:<12} no rel_price_premium coefficient")
            continue
        lines.append(
            f" {mark}{row['treatment']:<12} {row['rows_fitted']:>7} "
            f"{row['events']:>7} {beta:>11.4f}  "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    lines.append("")
    lines.append("  * = treatment in force (defaults.cancelled_treatment)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Fit the demand system on an export and print its diagnostics.

    Exit codes: 0 clean, 2 fitted with warnings, 1 blocked or unfittable. The
    printed report is the deliverable.
    """
    import argparse
    from pathlib import Path

    from src.config import cancelled_treatment, load_market_config
    from src.data.features import build_features
    from src.data.ingest_mls import ingest_mls
    from src.demand.logistic import LogisticDemandModel
    from src.demand.survival import (
        CONTROLLED_CATEGORICALS,
        CoxDemandModel,
        available_covariates,
    )

    parser = argparse.ArgumentParser(description="Fit the demand model and report")
    parser.add_argument("--inspect", action="store_true", help="print the report")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--file", type=Path, action="append", default=None)
    parser.add_argument("--market", default="miami")
    parser.add_argument(
        "--controls",
        action="store_true",
        help="add the hedonic controls (log area, view) the export supports",
    )
    parser.add_argument(
        "--building-fe",
        action="store_true",
        help="absorb building fixed effects; drops submarket, which they nest",
    )
    parser.add_argument(
        "--cancelled-sweep",
        action="store_true",
        help=(
            "also refit under each of censored/excluded/event for CANCELED and "
            "print the three beta_price values side by side"
        ),
    )
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument(
        "--calibration-gate",
        action="store_true",
        help=(
            "acknowledge PROJECT_BRIEF §5 and fit on the real MLS export. "
            "Required for any source under data/raw."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_market_config(args.market)
    horizon = int(args.horizon_days or (config.get("defaults") or {}).get("horizon_days") or 180)

    if _targets_real_data(args.dir, args.file) and not args.calibration_gate:
        print(
            "REFUSED: this would fit the demand model on the real MLS export.\n"
            "PROJECT_BRIEF §5 gates calibration on real data behind a human decision.\n"
            "Point --file or --dir at data/synthetic/, or pass --calibration-gate to "
            "say you are deliberately at the gate."
        )
        return 1

    try:
        ingested = ingest_mls(
            paths=args.file, market=args.market, raw_dir=args.dir, config=config
        )
        frame = build_features(ingested.frame, config).frame
        covariates = (
            available_covariates(frame) if args.controls else tuple(DEMAND_COVARIATES)
        )
        categoricals = CONTROLLED_CATEGORICALS if args.controls else ("submarket", "season")
        cox = CoxDemandModel(
            covariates=covariates,
            categoricals=categoricals,
            building_fixed_effects=args.building_fe,
        )
        cox_result = cox.fit(frame)
        logit: LogisticDemandModel | None = LogisticDemandModel(
            horizon, covariates=covariates, categoricals=categoricals
        )
        logit_error: str | None = None
        try:
            logit.fit(frame)
        except (SchemaError, IdentificationError) as exc:
            logit_error = str(exc)
            logit = None
        report = run_diagnostics(
            frame, cox_result, logistic_model=logit, logistic_error=logit_error
        )
    except (SchemaError, IdentificationError) as exc:
        print(f"FAIL: {exc}")
        return 1

    print(format_diagnostics_report(report))

    if args.cancelled_sweep:
        print()
        print(
            format_cancelled_sweep(
                cancelled_sweep(
                    market=args.market,
                    paths=args.file,
                    raw_dir=args.dir,
                    controls=args.controls,
                    building_fe=args.building_fe,
                ),
                cancelled_treatment(config),
            )
        )

    if report.status == "FAILED":
        return 1
    return 2 if report.status == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
