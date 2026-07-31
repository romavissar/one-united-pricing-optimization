"""A02 — Isolate the source of the systematic attenuation in Cox recovery.

A01 ruled out any wedge between the generator's `rel_price_premium` and the
pipeline's: they are bit-identical. Yet the fit returns roughly -0.31 / -1.49 /
-2.88 against planted -0.4 / -1.6 / -3.0 — a systematic shrink toward zero.

This script walks the candidate causes one at a time on the same data:

  1. baseline           — what the shipped model does
  2. premium only       — drop every other covariate
  3. + month dummies    — the DGP's month effect is unmodelled (only `season` is)
  4. unscaled design    — rule out the scale/unscale round-trip
  5. exact-duration     — rule out day-rounding ties
  6. Poisson MLE        — a correctly-specified parametric fit on the true DGP

Run from backend/:  .venv/bin/python ../audit/a02_recovery_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import DEMAND_COVARIATES, build_features  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.survival import CoxDemandModel  # noqa: E402

N = 6000
SEED = 13


def _cox_raw(df: pd.DataFrame, cols: list[str], duration: str, event: str) -> float:
    """Fit lifelines directly on named columns, no scaling, return beta_price."""
    fit_frame = df[[*cols, duration, event]].dropna().copy()
    fitter = CoxPHFitter()
    fitter.fit(fit_frame, duration_col=duration, event_col=event)
    return float(fitter.params_["rel_price_premium"])


def main() -> int:
    config = load_market_config("miami")

    for true_beta in (-0.4, -1.6, -3.0):
        synth = generate_synthetic_mls(
            n=N, seed=SEED, true_beta_price=true_beta, hazard_basis="realized"
        )
        frame = build_features(
            normalize_mls(synth.frame, config=config).frame, config
        ).frame
        latent = synth.latent.set_index("mls_number")
        frame = frame.set_index(frame["mls_number"].astype(str))
        frame["exact_duration"] = latent["duration_days"].reindex(frame.index)
        frame["latent_event"] = latent["event_sold"].reindex(frame.index)
        frame["time_to_sale"] = latent["time_to_sale_days"].reindex(frame.index)
        frame["censor_time"] = latent["censor_time_days"].reindex(frame.index)

        print(f"\n{'=' * 66}\nplanted beta_price = {true_beta:+.2f}   n={len(frame)}")
        print(f"{'=' * 66}")

        # 1. shipped model
        shipped = CoxDemandModel(covariates=DEMAND_COVARIATES).fit(frame)
        print(f"  1. shipped model                    {shipped.beta_price.value:+.4f}"
              f"   (n={shipped.n_observations}, events={shipped.n_events})")

        # 2. premium only, raw lifelines
        b2 = _cox_raw(frame, ["rel_price_premium"], "duration_days", "event_sold")
        print(f"  2. premium only (raw lifelines)     {b2:+.4f}")

        # 3. premium + submarket + month dummies
        d3 = pd.get_dummies(
            frame[["rel_price_premium", "submarket", "list_month",
                   "duration_days", "event_sold"]].dropna(),
            columns=["submarket", "list_month"], drop_first=True, dtype=float,
        )
        b3 = _cox_raw(
            d3, [c for c in d3.columns if c not in ("duration_days", "event_sold")],
            "duration_days", "event_sold",
        )
        print(f"  3. + submarket + month dummies      {b3:+.4f}")

        # 4. premium + submarket only (no month)
        d4 = pd.get_dummies(
            frame[["rel_price_premium", "submarket", "duration_days", "event_sold"]].dropna(),
            columns=["submarket"], drop_first=True, dtype=float,
        )
        b4 = _cox_raw(
            d4, [c for c in d4.columns if c not in ("duration_days", "event_sold")],
            "duration_days", "event_sold",
        )
        print(f"  4. + submarket only                 {b4:+.4f}")

        # 5. exact (unrounded) durations, premium only
        exact = frame[["rel_price_premium", "time_to_sale", "censor_time"]].dropna().copy()
        exact["dur"] = np.minimum(exact["time_to_sale"], exact["censor_time"])
        exact["ev"] = (exact["time_to_sale"] <= exact["censor_time"]).astype(int)
        b5 = _cox_raw(exact, ["rel_price_premium"], "dur", "ev")
        print(f"  5. exact unrounded durations        {b5:+.4f}")

        # 6. correctly specified exponential MLE on the true DGP.
        #    hazard is constant, so the MLE of beta solves a Poisson regression
        #    of event on log(exposure) offset + covariates.
        import statsmodels.api as sm

        pois = frame[["rel_price_premium", "duration_days", "event_sold"]].dropna().copy()
        X = sm.add_constant(pois[["rel_price_premium"]].astype(float))
        b6 = float(
            sm.GLM(
                pois["event_sold"].astype(float), X,
                family=sm.families.Poisson(),
                offset=np.log(pois["duration_days"].astype(float)),
            ).fit().params["rel_price_premium"]
        )
        print(f"  6. exponential/Poisson MLE          {b6:+.4f}")

        # 7. Poisson MLE with exact durations
        pois2 = exact.copy()
        X2 = sm.add_constant(pois2[["rel_price_premium"]].astype(float))
        b7 = float(
            sm.GLM(
                pois2["ev"].astype(float), X2,
                family=sm.families.Poisson(),
                offset=np.log(pois2["dur"].astype(float)),
            ).fit().params["rel_price_premium"]
        )
        print(f"  7. Poisson MLE, exact durations     {b7:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
