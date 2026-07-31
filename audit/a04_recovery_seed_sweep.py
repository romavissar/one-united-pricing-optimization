"""A04 — Is the failing recovery test measuring bias, or is it mis-calibrated?

A03 showed the generator plants what it claims (n=60,000 recovers -0.399 for a
planted -0.4) and the estimator is unbiased on hand-rolled data. So the n=6,000
shortfall is either a small-sample bias or ordinary sampling noise that the
test's fixed 20% relative tolerance cannot accommodate.

This distinguishes them. Across many seeds at n=6,000 it reports the mean
recovered beta (bias), the spread (noise), the mean fitted standard error, and
the share of seeds the shipped test would fail. An unbiased estimator whose
20%-of-beta tolerance is smaller than ~2 standard errors will fail a large
fraction of seeds *by construction*.

Run from backend/:  .venv/bin/python ../audit/a04_recovery_seed_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import DEMAND_COVARIATES, build_features  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.survival import CoxDemandModel  # noqa: E402

N = 6000
SEEDS = list(range(1, 25))
TOLERANCE = 0.20


def main() -> int:
    config = load_market_config("miami")
    print(f"n={N} per fit, {len(SEEDS)} seeds, shipped tolerance {TOLERANCE:.0%}\n")

    for true_beta in (-0.4, -1.6, -3.0):
        betas, ses, covered = [], [], []
        for seed in SEEDS:
            synth = generate_synthetic_mls(
                n=N, seed=seed, true_beta_price=true_beta, hazard_basis="realized"
            )
            frame = build_features(
                normalize_mls(synth.frame, config=config).frame, config
            ).frame
            fit = CoxDemandModel(covariates=DEMAND_COVARIATES).fit(frame)
            b = fit.beta_price
            betas.append(b.value)
            ses.append(b.std_error)
            covered.append(b.ci_low <= true_beta <= b.ci_high)

        betas_a = np.array(betas)
        ses_a = np.array(ses)
        rel_err = np.abs(betas_a - true_beta) / abs(true_beta)
        fails = rel_err > TOLERANCE
        bias = betas_a.mean() - true_beta
        # t-statistic for "is the mean recovered value different from planted"
        t_bias = bias / (betas_a.std(ddof=1) / np.sqrt(len(betas_a)))

        print(f"planted {true_beta:+.2f}")
        print(f"  mean recovered      {betas_a.mean():+.4f}   bias {bias:+.4f}"
              f"   (t = {t_bias:+.2f} across seeds)")
        print(f"  sd across seeds     {betas_a.std(ddof=1):.4f}")
        print(f"  mean fitted SE      {ses_a.mean():.4f}")
        print(f"  tolerance in SEs    {TOLERANCE * abs(true_beta) / ses_a.mean():.2f} SE")
        print(f"  95% CI covers truth {sum(covered)}/{len(SEEDS)}")
        print(f"  shipped test FAILS  {int(fails.sum())}/{len(SEEDS)} seeds"
              f"  ({fails.mean():.0%})\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
