"""Persistence for fitted demand models, and the provenance that travels with them.

The reason this module exists is `AGENTS.md` §4. Every optimize and simulate
response has to carry `is_calibrated_on_real_data`, and the only moment that
fact is known for certain is when the model is fitted. Recording it later, from
an environment variable or a guess, is how a model fitted on synthetic data ends
up in a client deck with no banner on it.

So the flag defaults to `False` and has to be asserted, not inferred. A caller
who does not say where the data came from gets "synthetic", which is the safe
direction to be wrong in.

Model objects are pickled; the metadata sits beside them as JSON so it can be
read without unpickling and without importing lifelines. Pickle is version-
fragile — a bundle written under one lifelines release may not load under the
next — which is why `metadata.json` carries the library versions and why the
coefficients are duplicated there in plain numbers.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lifelines
import sklearn
import statsmodels

from src.config import validate_market_id
from src.demand.base import FitResult, design_summary
from src.demand.diagnostics import DiagnosticsReport
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PICKLE = "bundle.pkl"
_METADATA_JSON = "metadata.json"

_BUNDLE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SYNTHETIC = "synthetic"
REAL_MLS = "real_mls"


@dataclass
class Provenance:
    """Where a fitted model came from and whether its numbers mean anything.

    is_calibrated_on_real_data is the flag the frontend keys its non-dismissible
    banner off. It defaults to False and is only True when the caller says the
    fit ran on a real MLS export.
    """

    market: str
    data_source: str = SYNTHETIC
    is_calibrated_on_real_data: bool = False
    source_paths: list[str] = field(default_factory=list)
    fitted_at: str = ""
    rows_ingested: int | None = None
    horizon_days: int | None = None
    planted_truth: dict[str, Any] | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.data_source not in {SYNTHETIC, REAL_MLS}:
            raise SchemaError(
                f"data_source must be {SYNTHETIC!r} or {REAL_MLS!r}, got {self.data_source!r}"
            )
        if self.is_calibrated_on_real_data and self.data_source != REAL_MLS:
            raise SchemaError(
                "is_calibrated_on_real_data is True but data_source is "
                f"{self.data_source!r}. These cannot disagree — the flag is the only "
                "thing standing between a synthetic number and a client deck."
            )
        if not self.fitted_at:
            self.fitted_at = datetime.now(UTC).isoformat(timespec="seconds")
        if not self.library_versions:
            self.library_versions = {
                "lifelines": lifelines.__version__,
                "statsmodels": statsmodels.__version__,
                "scikit-learn": sklearn.__version__,
            }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBundle:
    """A fitted demand system: the models, their diagnostics, and provenance."""

    provenance: Provenance
    cox: Any
    cox_result: FitResult
    diagnostics: DiagnosticsReport
    logistic: Any = None
    logistic_result: FitResult | None = None
    hedonic: Any = None
    # Diagnostics of the list-price hedonic whose residual IS the identifying
    # variable. Without this a saved bundle cannot say what its own
    # `rel_price_premium` was built from.
    premium_fit: dict[str, Any] | None = None

    @property
    def beta_price(self) -> float:
        """The reported own-price elasticity coefficient."""
        return self.cox_result.beta_price.value


def _coefficient_block(result: FitResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "model_kind": result.model_kind,
        "n_observations": result.n_observations,
        "n_events": result.n_events,
        "fit_stats": result.fit_stats,
        "covariates": list(result.covariates),
        "coefficients": {name: c.as_dict() for name, c in result.coefficients.items()},
        "design": design_summary(result.design),
        "notes": result.notes,
    }


def build_metadata(bundle: ModelBundle) -> dict[str, Any]:
    """Plain-JSON summary of a bundle: readable without unpickling anything."""
    metadata: dict[str, Any] = {
        "provenance": bundle.provenance.as_dict(),
        "beta_price": bundle.cox_result.beta_price.as_dict(),
        "diagnostics": bundle.diagnostics.as_dict(),
        "cox": _coefficient_block(bundle.cox_result),
        "logistic": _coefficient_block(bundle.logistic_result),
    }
    # The full coefficient covariance, in raw units, so a later phase can
    # resample the whole fit instead of one coefficient. Marginal standard
    # errors alone cannot reconstruct it, and Phase 4 is where its absence
    # first bites: a revenue distribution built from beta_price's SE alone
    # omits every other coefficient's uncertainty.
    covariance = getattr(bundle.cox, "coefficient_covariance", None)
    if covariance is not None:
        metadata["cox_coefficient_covariance"] = {
            "columns": [str(c) for c in covariance.columns],
            "matrix": [[float(v) for v in row] for row in covariance.to_numpy()],
            "units": "raw covariate units, matching the reported coefficients",
        }

    if bundle.premium_fit:
        metadata["premium_hedonic"] = bundle.premium_fit

    if bundle.hedonic is not None:
        metadata["hedonic"] = {
            **(_coefficient_block(bundle.hedonic.fit) or {}),
            "floor_premium": bundle.hedonic.floor_premium.as_dict(),
            "residual_sd_log_ppsf": bundle.hedonic.residual_sd,
            "bound_sd_multiple": bundle.hedonic.bound_sd_multiple,
        }
    return metadata


def model_dir(market: str, name: str = "current", root: Path | None = None) -> Path:
    """Directory a named bundle for `market` lives in.

    Both components are validated before they are joined: `market` reaches here
    from the URL and `name` from a request body, and what sits at the end of
    this path is a pickle that `load_bundle` will execute the constructors of.
    """
    base = root if root is not None else _BACKEND_ROOT / "data" / "processed"
    safe_market = validate_market_id(market)
    if not _BUNDLE_NAME.match(str(name)):
        raise SchemaError(
            f"Invalid bundle name {name!r}. A bundle name is lowercase letters, "
            "digits, hyphens, and underscores — it names a directory holding a "
            "pickle, so it is never a path."
        )
    return base / safe_market / "models" / str(name)


def save_bundle(
    bundle: ModelBundle, *, name: str = "current", root: Path | None = None
) -> Path:
    """Write a fitted bundle and its metadata to disk.

    Returns:
        The directory written to.
    """
    target = model_dir(bundle.provenance.market, name, root)
    target.mkdir(parents=True, exist_ok=True)

    with (target / _MODEL_PICKLE).open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (target / _METADATA_JSON).open("w", encoding="utf-8") as handle:
        json.dump(build_metadata(bundle), handle, indent=2, default=str)

    logger.info(
        "Saved %s bundle to %s (beta_price=%.4f, calibrated_on_real_data=%s)",
        bundle.provenance.market,
        target,
        bundle.beta_price,
        bundle.provenance.is_calibrated_on_real_data,
    )
    return target


def load_bundle(
    market: str, *, name: str = "current", root: Path | None = None
) -> ModelBundle:
    """Load a saved bundle.

    Raises:
        SchemaError: when no bundle exists at that path, or the pickle cannot be
            read under the current library versions.
    """
    target = model_dir(market, name, root)
    path = target / _MODEL_PICKLE
    if not path.exists():
        raise SchemaError(
            f"No fitted model at {path}. Fit and save one before asking for a "
            "price recommendation."
        )
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - unpickling raises almost anything
        raise SchemaError(
            f"Could not load the model bundle at {path}. Pickles do not survive "
            f"library upgrades; refit. Underlying error: {exc}"
        ) from exc
    if not isinstance(bundle, ModelBundle):
        raise SchemaError(f"{path} does not contain a ModelBundle")
    return bundle


def load_metadata(
    market: str, *, name: str = "current", root: Path | None = None
) -> dict[str, Any]:
    """Read a bundle's metadata without unpickling the models."""
    path = model_dir(market, name, root) / _METADATA_JSON
    if not path.exists():
        raise SchemaError(f"No model metadata at {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
