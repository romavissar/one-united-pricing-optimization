"""A11 — Why does the building-fixed-effects recovery test now fall short?

`test_building_fixed_effects_recover_beta_when_quality_is_building_level` demands
that fixed effects recover >= 60% of a planted -1.6 when 100% of the quality
contamination is building-level. The README records -1.185 (74%). It now returns
-0.705 (44%).

Hypothesis: the synthetic generator builds `_BUILDINGS_PER_SUBMARKET` towers for
every submarket in `config/miami.yaml`. The config was extended from 8 to 14
submarkets to cover the real export's ZIPs, which raised the tower count from 96
to 168 while n stayed at 6,000 — so listings per building fell from ~62 to ~36,
and a Cox partial likelihood carrying one dummy per tower pays an
incidental-parameters cost that grows as each level thins.

If that is the cause, holding the submarket count at 8 should restore ~74%, and
the defect is a hidden coupling between market config and every synthetic
benchmark — including the numbers written down in the README.

Run from backend/:  .venv/bin/python ../audit/a11_building_fe_coupling.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import DEMAND_COVARIATES, build_features  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.survival import (  # noqa: E402
    CONTROLLED_CATEGORICALS, CONTROLLED_COVARIATES, CoxDemandModel,
)

PLANTED = -1.6
N = 6000
SEED = 13


def recover(config, n=N, seed=SEED):
    synth = generate_synthetic_mls(
        n=n, seed=seed, true_beta_price=PLANTED, hazard_basis="latent",
        building_quality_share=1.0, config=config,
    )
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    naive = CoxDemandModel(covariates=DEMAND_COVARIATES).fit(frame).beta_price.value
    controlled = CoxDemandModel(
        covariates=CONTROLLED_COVARIATES, categoricals=CONTROLLED_CATEGORICALS,
        building_fixed_effects=True,
    ).fit(frame)
    n_buildings = frame["building_name"].nunique()
    return naive, controlled.beta_price.value, n_buildings, controlled.n_observations


def main() -> int:
    full = load_market_config("miami")
    all_subs = list(full["submarkets"])
    print(f"config declares {len(all_subs)} submarkets: {all_subs}\n")

    print(f"{'submarkets':>11} {'towers':>7} {'n fitted':>9} {'listings/tower':>15} "
          f"{'naive':>8} {'with FE':>9} {'% of planted':>13}")
    for k in (8, 10, 12, 14):
        cfg = copy.deepcopy(full)
        cfg["submarkets"] = {s: full["submarkets"][s] for s in all_subs[:k]}
        naive, fe, towers, n_fit = recover(cfg)
        print(f"{k:>11} {towers:>7} {n_fit:>9} {n_fit / towers:>15.1f} "
              f"{naive:>8.4f} {fe:>9.4f} {abs(fe) / abs(PLANTED):>12.0%}")

    print("\nHolding submarkets at 8 but raising n, to separate 'more towers' from")
    print("'fewer listings per tower':")
    cfg8 = copy.deepcopy(full)
    cfg8["submarkets"] = {s: full["submarkets"][s] for s in all_subs[:8]}
    cfg14 = full
    for label, cfg, n in (("8 subs, n=6000", cfg8, 6000),
                          ("14 subs, n=6000", cfg14, 6000),
                          ("14 subs, n=10500", cfg14, 10500)):
        naive, fe, towers, n_fit = recover(cfg, n=n)
        print(f"  {label:<18} towers={towers:>4}  n={n_fit:>6}  "
              f"listings/tower={n_fit / towers:>5.1f}  FE beta={fe:+.4f}  "
              f"({abs(fe) / abs(PLANTED):.0%} of planted)")

    print("\nIf recovery tracks listings-per-tower rather than tower count, the")
    print("shortfall is incidental parameters, and the test is measuring the market")
    print("config rather than the estimator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
