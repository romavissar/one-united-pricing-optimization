"""Standard errors for `beta_price` when the regressor is itself estimated.

`rel_price_premium` is no longer a column read off the export. It is the
residual from a hedonic fitted on the same listings, which makes it a
**generated regressor**: its value for each listing depends on coefficients
estimated with error from the whole sample. The Cox model's reported standard
error conditions on that residual as if it were data, so it is the standard
error of the wrong quantity.

The direction of the error is not obvious, which is exactly why it has to be
measured rather than reasoned about. Two effects run against each other:
estimating the hedonic adds sampling noise the conventional SE ignores, which
argues the true interval is wider; but the residual is *orthogonal by
construction* to every control in the first stage, and that orthogonality
removes covariance terms a naive calculation would carry, which can pull the
other way. Pagan's results on generated regressors give conditions for either
sign. So this resamples and looks.

The resample is over **listings**, and it repeats the whole chain each time —
refit the hedonic, recompute the residual, refit the Cox, keep `beta_price`.
Anything less (resampling residuals, or holding the first stage fixed) would
hold constant the very thing whose uncertainty is being measured.

Cached, because it is expensive: one bootstrap replication is two full model
fits over 45,000 listings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.premium import fit_price_premium
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_CACHE_VERSION = 2


@dataclass
class BootstrapResult:
    """The bootstrap distribution of `beta_price` and what it implies."""

    n_replications: int
    n_succeeded: int
    point_estimate: float
    conventional_se: float
    bootstrap_se: float
    ci_low: float
    ci_high: float
    se_ratio: float
    draws: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        """Whether treating the residual as data understated or overstated the SE."""
        if self.se_ratio > 1.05:
            return "understated"
        if self.se_ratio < 0.95:
            return "overstated"
        return "materially unchanged"

    def as_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "draws"},
            "direction": self.direction,
            "n_draws_retained": len(self.draws),
        }


def _cache_path(market: str, key: str) -> Path:
    directory = _BACKEND_ROOT / "data" / "processed" / market / "bootstrap"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"beta_price_{key}.json"


def _fingerprint(frame: pd.DataFrame, n_replications: int, seed: int) -> str:
    """Identify the input so a cached result cannot be served for other data."""
    parts = [
        str(len(frame)),
        str(int(pd.to_numeric(frame.get("event_sold"), errors="coerce").sum())),
        f"{float(pd.to_numeric(frame.get('list_ppsf'), errors='coerce').sum()):.4f}",
        str(n_replications),
        str(seed),
        str(_CACHE_VERSION),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def bootstrap_beta_price(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
    categoricals: tuple[str, ...],
    point_estimate: float,
    conventional_se: float,
    n_replications: int = 200,
    seed: int = 20260731,
    market: str = "miami",
    use_cache: bool = True,
) -> BootstrapResult:
    """Two-stage bootstrap: resample listings, refit both stages, collect beta.

    Args:
        frame: the feature frame the point estimate was fitted on.
        covariates / categoricals: the Cox specification to replicate.
        point_estimate: `beta_price` on the full sample.
        conventional_se: the standard error the Cox model reported, which
            treats the generated regressor as data.
        n_replications: bootstrap replications. 200 gives a standard-error
            estimate accurate to roughly 5%, which is enough to establish
            direction and rough magnitude.

    Raises:
        SchemaError: when the frame cannot support even one replication.
    """
    from src.demand.survival import CoxDemandModel

    key = _fingerprint(frame, n_replications, seed)
    path = _cache_path(market, key)
    if use_cache and path.exists():
        cached = json.loads(path.read_text())
        logger.info("Using cached bootstrap at %s", path)
        return BootstrapResult(**cached)

    rng = np.random.default_rng(seed)
    n = len(frame)
    draws: list[float] = []
    failures = 0

    for replication in range(n_replications):
        idx = rng.integers(0, n, size=n)
        sample = frame.iloc[idx].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Stage one: the hedonic, refitted on this resample.
                premium = fit_price_premium(sample)
                sample = sample.copy()
                keep = premium.premium_ratio.notna() & premium.reference_ppsf.notna()
                sample["rel_price_premium"] = premium.premium_ratio.where(keep)
                sample["cell_median_ppsf"] = premium.reference_ppsf.where(keep)
                # Stage two: the demand model, on this resample's residual.
                result = CoxDemandModel(
                    covariates=covariates, categoricals=categoricals
                ).fit(sample)
            draws.append(float(result.beta_price.value))
        except (SchemaError, IdentificationError, np.linalg.LinAlgError, ValueError) as exc:
            failures += 1
            if failures <= 3:
                logger.warning("Bootstrap replication %d failed: %s", replication, exc)

    if not draws:
        raise SchemaError(
            f"Every one of {n_replications} bootstrap replications failed; the "
            "two-stage standard error cannot be estimated on this sample."
        )

    values = np.asarray(draws, dtype="float64")
    boot_se = float(values.std(ddof=1))
    notes: list[str] = []
    if failures:
        notes.append(
            f"{failures} of {n_replications} replications failed to fit and are "
            "excluded; the reported spread is over those that converged."
        )
    ratio = boot_se / conventional_se if conventional_se > 0 else float("nan")
    notes.append(
        "The resample is over listings and refits the hedonic inside each "
        "replication, so the spread includes first-stage estimation error that "
        "the conventional standard error conditions away."
    )

    outcome = BootstrapResult(
        n_replications=n_replications,
        n_succeeded=len(draws),
        point_estimate=float(point_estimate),
        conventional_se=float(conventional_se),
        bootstrap_se=boot_se,
        ci_low=float(np.quantile(values, 0.025)),
        ci_high=float(np.quantile(values, 0.975)),
        se_ratio=float(ratio),
        draws=[float(v) for v in values],
        notes=notes,
    )
    if use_cache:
        path.write_text(json.dumps(asdict(outcome), indent=2))
        logger.info("Cached bootstrap to %s", path)
    logger.info(
        "Two-stage bootstrap: beta_price %.4f, conventional SE %.4f, bootstrap SE "
        "%.4f (ratio %.2f, %s)",
        point_estimate, conventional_se, boot_se, ratio, outcome.direction,
    )
    return outcome
