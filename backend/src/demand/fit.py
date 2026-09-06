"""Fit the demand system and persist a bundle — the calibration-gate entry point.

`PROJECT_BRIEF.md` §5 names this module as the command to run once real MLS
data is ready. It fits, diagnoses, and saves. It does **not** flip
`is_calibrated_on_real_data` to true: that flag is read from
`config/{market}.yaml` and stays false until a human edits it after inspecting
the fit. A successful fit is not evidence the numbers are ready for a client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.config import MarketConfig, load_market_config
from src.data.features import DEMAND_COVARIATES, build_features
from src.data.ingest_mls import ingest_mls
from src.data.normalize import normalize_mls
from src.data.synth import generate_synthetic_mls
from src.demand.diagnostics import (
    DiagnosticsReport,
    _targets_real_data,
    format_diagnostics_report,
    run_diagnostics,
)
from src.demand.hedonic import fit_hedonic
from src.demand.logistic import LogisticDemandModel
from src.demand.registry import REAL_MLS, SYNTHETIC, ModelBundle, Provenance, save_bundle
from src.demand.survival import (
    CONTROLLED_CATEGORICALS,
    CoxDemandModel,
    available_covariates,
)
from src.exceptions import IdentificationError, SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

DatasetName = Literal["synthetic", "mls"]


@dataclass
class FitRequest:
    """Parameters for a demand fit. Shared by the CLI and the API."""

    dataset: DatasetName
    market: str = "miami"
    calibration_gate: bool = False
    assert_calibrated: bool = False
    n_listings: int = 6000
    seed: int = 42
    controls: bool = True
    building_fe: bool = False
    horizon_days: int | None = None
    name: str = "current"
    directory: Path | None = None
    files: list[Path] | None = None


@dataclass
class FitOutcome:
    """Saved bundle plus the diagnostics a caller must read first."""

    bundle: ModelBundle
    report: DiagnosticsReport
    path: Path
    is_calibrated_on_real_data: bool
    config_flag: bool

    def as_dict(self) -> dict[str, Any]:
        beta = self.bundle.cox_result.beta_price
        return {
            "path": str(self.path),
            "status": self.report.status,
            "identification_report": self.report.identification.as_dict(),
            "diagnostics": self.report.as_dict(),
            "coefficients": {
                name: c.as_dict() for name, c in self.bundle.cox_result.coefficients.items()
            },
            "beta_price": beta.as_dict(),
            "provenance": self.bundle.provenance.as_dict(),
            "is_calibrated_on_real_data": self.is_calibrated_on_real_data,
            "config_is_calibrated_on_real_data": self.config_flag,
        }


def fit_demand(
    request: FitRequest, config: MarketConfig | None = None
) -> FitOutcome:
    """Fit, diagnose, and save a bundle.

    Raises:
        SchemaError: refused (missing calibration gate, bad config flag) or
            unfittable input.
        IdentificationError: model identification failure during fit.
    """
    cfg = config or load_market_config(request.market)
    defaults = cfg.get("defaults") or {}
    horizon = int(request.horizon_days or defaults.get("horizon_days") or 180)
    files = request.files

    if request.dataset == "mls":
        if not request.calibration_gate:
            raise SchemaError(
                "REFUSED: dataset=mls would fit on the real MLS export. "
                "PROJECT_BRIEF §5 requires calibration_gate=true after a human "
                "decision. Use dataset=synthetic, or pass calibration_gate=true."
            )
        raw_dir = request.directory or (_BACKEND_ROOT / "data" / "raw" / "mls")
        ingested = ingest_mls(
            paths=files, market=request.market, raw_dir=raw_dir, config=cfg
        )
        built = build_features(ingested.frame, cfg)
        frame, premium_fit = built.frame, built.report.hedonic_premium
        source_paths = [str(p) for p in (files or [raw_dir])]
        rows = len(ingested.frame)
        planted = None
        data_source = REAL_MLS
    else:
        if files or request.directory:
            if _targets_real_data(request.directory, files) and not request.calibration_gate:
                raise SchemaError(
                    "REFUSED: path points at data/raw without calibration_gate=true."
                )
            ingested = ingest_mls(
                paths=files, market=request.market, raw_dir=request.directory, config=cfg
            )
            built = build_features(ingested.frame, cfg)
            frame, premium_fit = built.frame, built.report.hedonic_premium
            source_paths = [str(p) for p in (files or [request.directory])]
            rows = len(ingested.frame)
            planted = None
        else:
            synthetic = generate_synthetic_mls(
                n=request.n_listings,
                seed=request.seed,
                market=request.market,
                profile="rich",
            )
            built = build_features(
                normalize_mls(synthetic.frame, config=cfg).frame, cfg
            )
            frame, premium_fit = built.frame, built.report.hedonic_premium
            source_paths = [
                f"synthetic:seed={request.seed}:n={request.n_listings}"
            ]
            rows = len(frame)
            planted = {
                "true_beta_price": synthetic.truth.beta_price,
                "profile": "rich",
                "seed": request.seed,
            }
        data_source = SYNTHETIC

    calibrated_flag = bool(defaults.get("is_calibrated_on_real_data", False))
    if request.assert_calibrated:
        if request.dataset != "mls" or not request.calibration_gate:
            raise SchemaError(
                "REFUSED: assert_calibrated requires dataset=mls and "
                "calibration_gate=true. The flag is never set as a side effect "
                "of a synthetic fit."
            )
        if not calibrated_flag:
            raise SchemaError(
                "REFUSED: config defaults.is_calibrated_on_real_data is still "
                f"false. Edit config/{request.market}.yaml to true after "
                "inspecting β_price and its CI, then re-run with assert_calibrated."
            )
        is_calibrated = True
    else:
        is_calibrated = False

    covariates = (
        available_covariates(frame) if request.controls else tuple(DEMAND_COVARIATES)
    )
    categoricals = (
        CONTROLLED_CATEGORICALS if request.controls else ("submarket", "season")
    )
    cox = CoxDemandModel(
        covariates=covariates,
        categoricals=categoricals,
        building_fixed_effects=request.building_fe,
    )
    cox_result = cox.fit(frame)
    logit: LogisticDemandModel | None = LogisticDemandModel(
        horizon, covariates=covariates, categoricals=categoricals
    )
    logit_result = None
    logit_error: str | None = None
    try:
        logit_result = logit.fit(frame)
    except (SchemaError, IdentificationError) as exc:
        logit_error = str(exc)
        logit = None
    surface = fit_hedonic(frame, cfg)
    report = run_diagnostics(
        frame, cox_result, logistic_model=logit, logistic_error=logit_error
    )
    # Submarket medians at fit time — what optimize uses for rel_price_premium
    # when the client does not pass an explicit comps map.
    comps_at_fit: dict[str, float] = {}
    if "submarket" in frame.columns and "cell_median_ppsf" in frame.columns:
        for submarket, group in frame.groupby(frame["submarket"].astype(str), sort=False):
            median = pd.to_numeric(group["cell_median_ppsf"], errors="coerce").median()
            if pd.notna(median):
                comps_at_fit[str(submarket)] = float(median)
    provenance = Provenance(
        market=request.market,
        data_source=data_source,
        is_calibrated_on_real_data=is_calibrated,
        source_paths=source_paths,
        rows_ingested=rows,
        horizon_days=horizon,
        planted_truth={**(planted or {}), "comps_ppsf_by_submarket": comps_at_fit},
        notes=[
            "is_calibrated_on_real_data stays false until config is flipped "
            "and fit is re-run with assert_calibrated"
        ],
    )
    bundle = ModelBundle(
        provenance=provenance,
        cox=cox,
        cox_result=cox_result,
        diagnostics=report,
        logistic=logit,
        logistic_result=logit_result,
        hedonic=surface,
        premium_fit=premium_fit,
    )
    target = save_bundle(bundle, name=request.name)
    return FitOutcome(
        bundle=bundle,
        report=report,
        path=target,
        is_calibrated_on_real_data=is_calibrated,
        config_flag=calibrated_flag,
    )


def main(argv: list[str] | None = None) -> int:
    """Fit on `synthetic` or `mls`, save the bundle, print diagnostics.

    Exit codes: 0 clean, 2 warnings, 1 blocked / refused / unfittable.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit the demand model and save a provenance-tagged bundle"
    )
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "mls"),
        required=True,
        help="synthetic = generator + optional --file; mls = data/raw/mls/",
    )
    parser.add_argument("--market", default="miami")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--file", type=Path, action="append", default=None)
    parser.add_argument("--n-listings", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controls",
        action="store_true",
        default=True,
        help="hedonic controls (default on)",
    )
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--building-fe", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument(
        "--name",
        default="current",
        help="bundle name under data/processed/{market}/models/",
    )
    parser.add_argument(
        "--calibration-gate",
        action="store_true",
        help="required to fit --dataset mls (PROJECT_BRIEF §5)",
    )
    parser.add_argument(
        "--assert-calibrated",
        action="store_true",
        help=(
            "set is_calibrated_on_real_data from config (must already be true in "
            "yaml). Refused unless --dataset mls and --calibration-gate."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        outcome = fit_demand(
            FitRequest(
                dataset=args.dataset,
                market=args.market,
                calibration_gate=args.calibration_gate,
                assert_calibrated=args.assert_calibrated,
                n_listings=args.n_listings,
                seed=args.seed,
                controls=bool(args.controls) and not args.no_controls,
                building_fe=args.building_fe,
                horizon_days=args.horizon_days,
                name=args.name,
                directory=args.dir,
                files=args.file,
            )
        )
    except (SchemaError, IdentificationError) as exc:
        print(f"FAIL: {exc}")
        return 1

    print(format_diagnostics_report(outcome.report))
    print()
    print(f"Saved bundle → {outcome.path}")
    print(
        f"is_calibrated_on_real_data={outcome.is_calibrated_on_real_data}  "
        f"(config default was {outcome.config_flag})"
    )
    print(
        "Read first: censored count, IQR(rel_price_premium), β_price with 95% CI. "
        "If the CI includes zero, stop — do not tune the model."
    )
    if outcome.report.status == "FAILED":
        return 1
    return 2 if outcome.report.status == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
