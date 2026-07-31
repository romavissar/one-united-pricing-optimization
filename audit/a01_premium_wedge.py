"""A01 — Why does `realized`-basis recovery attenuate when it should be exact?

Under `hazard_basis="realized"` the synthetic generator drives the sale hazard
with the *same* `rel_price_premium` the estimator regresses on. Recovery should
therefore be exact up to sampling noise. It is not: at a planted -0.4 the fit
returns about -0.31.

This script measures the wedge between the premium the generator used and the
premium the feature pipeline hands the estimator, and then refits on the
generator's own premium to confirm the wedge is the cause.

Run: .venv/bin/python ../audit/a01_premium_wedge.py   (from backend/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import build_features  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.survival import CoxDemandModel  # noqa: E402
from src.data.features import DEMAND_COVARIATES  # noqa: E402

N = 6000
SEED = 13


def main() -> int:
    config = load_market_config("miami")
    print(f"config submarkets: {len(config['submarkets'])}")

    for true_beta in (-0.4, -1.6, -3.0):
        synth = generate_synthetic_mls(
            n=N, seed=SEED, true_beta_price=true_beta, hazard_basis="realized"
        )
        frame = build_features(
            normalize_mls(synth.frame, config=config).frame, config
        ).frame

        # Align the generator's latent truth onto the feature frame by mls_number.
        latent = synth.latent.set_index("mls_number")
        merged = frame.set_index(frame["mls_number"].astype(str))
        planted = latent["rel_price_premium"].reindex(merged.index)
        pipeline = merged["rel_price_premium"]

        both = pd.DataFrame({"planted": planted, "pipeline": pipeline}).dropna()
        diff = both["pipeline"] - both["planted"]

        print(f"\n=== planted beta_price = {true_beta:+.2f} ===")
        print(f"  rows with both premiums: {len(both)} of {len(frame)}")
        print(f"  median_basis counts: "
              f"{frame['median_basis'].value_counts().to_dict()}")
        print(f"  corr(planted, pipeline)      : {both['planted'].corr(both['pipeline']):.6f}")
        print(f"  sd(planted)                  : {both['planted'].std():.6f}")
        print(f"  sd(pipeline)                 : {both['pipeline'].std():.6f}")
        print(f"  sd(pipeline - planted)       : {diff.std():.6f}")
        print(f"  max|pipeline - planted|      : {diff.abs().max():.6f}")
        print(f"  rows differing > 1e-12       : {int((diff.abs() > 1e-12).sum())}")

        # Classical errors-in-variables attenuation factor implied by the wedge.
        var_planted = float(both["planted"].var())
        var_wedge = float(diff.var())
        if var_planted > 0:
            attenuation = var_planted / (var_planted + var_wedge)
            print(f"  implied attenuation factor   : {attenuation:.4f}"
                  f"  -> predicts beta ~ {true_beta * attenuation:+.4f}")

        # Fit as-is.
        model = CoxDemandModel(covariates=DEMAND_COVARIATES)
        fitted = model.fit(frame)
        print(f"  fit on pipeline premium      : {fitted.beta_price.value:+.4f}")

        # Refit substituting the generator's own premium. If this recovers the
        # planted value, the wedge is the whole story.
        swapped = frame.copy()
        swapped["rel_price_premium"] = planted.to_numpy()
        model2 = CoxDemandModel(covariates=DEMAND_COVARIATES)
        fitted2 = model2.fit(swapped)
        print(f"  fit on planted premium       : {fitted2.beta_price.value:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
