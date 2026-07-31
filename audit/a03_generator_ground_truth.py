"""A03 — Does the synthetic generator actually plant the beta it claims?

A02 showed every estimator (Cox with ties, Cox on exact durations, and a
correctly-specified exponential MLE) agrees on ~-0.30 for a planted -0.4. That
rules out estimator code and points at the data-generating process itself.

Step 1 validates the estimator against a hand-rolled exponential simulation
with a known beta, using the same estimator call. If that recovers cleanly, the
estimator is sound and the generator is the suspect.

Step 2 reproduces the generator's own hazard arithmetic on its own latent frame
and asks whether the realized log-hazard really is `beta * rel_price_premium`.

Run from backend/:  .venv/bin/python ../audit/a03_generator_ground_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.data.synth import generate_synthetic_mls  # noqa: E402


def exponential_mle(event: np.ndarray, exposure: np.ndarray, x: np.ndarray) -> float:
    X = sm.add_constant(pd.DataFrame({"x": x}))
    model = sm.GLM(
        event.astype(float), X,
        family=sm.families.Poisson(),
        offset=np.log(exposure.astype(float)),
    ).fit()
    return float(model.params["x"])


def step1_estimator_sanity() -> None:
    """Hand-rolled exponential data with a known beta, same estimator."""
    print("STEP 1 — estimator sanity on hand-rolled exponential data")
    rng = np.random.default_rng(7)
    n = 60_000
    for beta in (-0.4, -1.6, -3.0):
        x = rng.normal(0.0, 0.21, size=n)          # same sd as the real premium
        rate = 0.004 * np.exp(beta * x)
        t = rng.exponential(1.0 / rate)
        c = np.minimum(rng.exponential(360.0, size=n), 180.0)
        event = (t <= c).astype(int)
        dur = np.minimum(t, c)
        got = exponential_mle(event, dur, x)
        print(f"   planted {beta:+.2f}  ->  recovered {got:+.4f}   "
              f"({abs(got - beta) / abs(beta):.1%} off)")


def step2_generator_truth() -> None:
    """Re-derive the generator's own log-hazard from its own latent frame."""
    print("\nSTEP 2 — the generator's realized hazard vs its planted beta")
    for beta in (-0.4, -1.6, -3.0):
        synth = generate_synthetic_mls(
            n=60_000, seed=13, true_beta_price=beta, hazard_basis="realized"
        )
        lat = synth.latent
        t = lat["time_to_sale_days"].to_numpy()
        c = lat["censor_time_days"].to_numpy()
        prem = lat["rel_price_premium"].to_numpy()
        signal = lat["hazard_price_signal"].to_numpy()

        event = (t <= c).astype(int)
        dur = np.minimum(t, c)

        on_premium = exponential_mle(event, dur, prem)
        on_signal = exponential_mle(event, dur, signal)

        # Uncensored: the cleanest possible read of the DGP.
        uncens = exponential_mle(np.ones_like(t, dtype=int), t, prem)

        print(f"\n   planted beta = {beta:+.2f}   (n=60000)")
        print(f"     MLE on rel_price_premium, censored    {on_premium:+.4f}")
        print(f"     MLE on hazard_price_signal, censored  {on_signal:+.4f}")
        print(f"     MLE on rel_price_premium, UNCENSORED  {uncens:+.4f}")
        print(f"     corr(premium, signal)                 "
              f"{np.corrcoef(prem, signal)[0, 1]:.6f}")
        print(f"     max|premium - signal|                 "
              f"{np.abs(prem - signal).max():.3e}")

        # Direct check: is log(1/mean time) linear in premium with slope beta?
        # Regress log(t) on premium; for an exponential, E[log T] = -log(rate) - gamma,
        # so the slope of log(t) on premium is exactly -beta.
        slope = np.polyfit(prem, np.log(t), 1)[0]
        print(f"     slope of log(time_to_sale) on premium {slope:+.4f}"
              f"   (should be {-beta:+.4f})")


if __name__ == "__main__":
    step1_estimator_sanity()
    step2_generator_truth()
