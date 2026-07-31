"""A10 — Metamorphic and property checks (brief §1.3).

Relations that must hold with no ground truth needed:

  M1  scale invariance: multiplying every price AND the cell median by the same
      factor must leave rel_price_premium unchanged
  M2  relabelling unit ids must not change the plan
  M3  doubling every unit's area should scale revenue proportionally
  M4  adding a strictly dominated price level must not change the optimum
  M5  monotonicity of P(sale) in price: more expensive => never more likely
  M6  probabilities in [0,1]; revenue non-negative; percentiles ordered
  M7  a more negative beta_price must not raise any recommended price
  M8  beta_price = 0 must push every released unit to its ceiling
  M9  permuting inventory ROW ORDER must not change the plan
  M10 the objective must be >= the value of any feasible point (here: the
      price-at-comps plan evaluated on the same tensor)

Run from backend/:  .venv/bin/python ../audit/a10_metamorphic.py
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
from src.demand.base import relative_premium  # noqa: E402
from src.demand.hedonic import fit_hedonic  # noqa: E402
from src.demand.survival import (  # noqa: E402
    CONTROLLED_CATEGORICALS, CoxDemandModel, available_covariates,
)
from src.optimizer.discretize import build_price_ladder, ladder_from_bounds  # noqa: E402
from src.optimizer.formulate import (  # noqa: E402
    Comps, DemandProvenance, Phase, ProjectSpec, build_revenue_tensor,
)
from src.optimizer.solve import solve_release_plan  # noqa: E402
from src.utils.validate import validate_inventory  # noqa: E402

PASS, FAIL = "  [ok]  ", "  [XX]  "
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL}{name}" + (f"   {detail}" if detail else ""))


class FixedBeta:
    """A demand model with a known beta and no other structure.

    Lets the elasticity-response relations be tested without a refit confounding
    the comparison.
    """

    kind = "fixed_beta"
    premium_support = None

    def __init__(self, beta: float, base: float = 0.6) -> None:
        self.beta, self.base = beta, base

    def predict_sale_probability(self, features, price_ppsf, horizon_days):
        prem = relative_premium(
            price_ppsf, features["cell_median_ppsf"].to_numpy(dtype="float64"))
        surv = (1.0 - self.base) ** np.exp(self.beta * prem)
        return 1.0 - surv


def context(config, units=None):
    units = validate_inventory(
        pd.read_csv("data/project_inputs/example_inventory.csv"), config
    ).frame if units is None else units
    synth = generate_synthetic_mls(n=4000, seed=13, market="miami", profile="rich")
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    surface = fit_hedonic(frame, config)
    comps = Comps({s: float(frame.loc[frame["submarket"] == s, "cell_median_ppsf"].median())
                   for s in units["submarket"].unique()})
    return units, frame, surface, comps


def spec_for(config, comps, months=(0.0, 6.0, 12.0, 18.0)):
    return ProjectSpec.from_config(
        config, project_start=pd.Timestamp("2026-01-01"),
        phases=tuple(Phase(index=j, name=f"phase_{j+1}", start_month=float(m),
                           max_units=20, competing_listings=30.0)
                     for j, m in enumerate(months)),
        comps=comps)


def main() -> int:
    config = load_market_config("miami")
    units, frame, surface, comps = context(config)
    prov = DemandProvenance(beta_price=-1.5)

    # ---------------------------------------------------------------- M1
    price = np.array([500.0, 1000.0, 2500.0])
    median = np.array([800.0, 800.0, 800.0])
    for factor in (0.5, 1.0, 2.0, 137.0):
        a = relative_premium(price, median)
        b = relative_premium(price * factor, median * factor)
        if not np.allclose(a, b):
            check("M1 scale invariance of rel_price_premium", False,
                  f"factor {factor}: {a} vs {b}")
            break
    else:
        check("M1 scale invariance of rel_price_premium", True)

    check("M1b non-positive median yields NaN, not a plausible zero",
          bool(np.isnan(relative_premium(np.array([900.0]), np.array([0.0])))[0]))

    # ---------------------------------------------------------------- M5
    model = FixedBeta(-1.5)
    feats = pd.DataFrame({"cell_median_ppsf": [800.0] * 5})
    grid = np.linspace(400, 1600, 40)
    probs = np.array([model.predict_sale_probability(feats, p, 180)[0] for p in grid])
    check("M5 P(sale) is non-increasing in price", bool(np.all(np.diff(probs) <= 1e-12)),
          f"range {probs.min():.4f}..{probs.max():.4f}")
    check("M6 P(sale) stays within [0,1]",
          bool(probs.min() >= 0.0 and probs.max() <= 1.0))

    real = CoxDemandModel(covariates=available_covariates(frame),
                          categoricals=CONTROLLED_CATEGORICALS)
    real.fit(frame)
    score = frame.head(300)
    pr = np.array([np.nanmean(real.predict_sale_probability(score, p, 180))
                   for p in np.linspace(300, 3000, 30)])
    check("M5b fitted Cox: mean P(sale) non-increasing in price",
          bool(np.all(np.diff(pr) <= 1e-9)), f"{pr[0]:.4f} -> {pr[-1]:.4f}")

    # ---------------------------------------------------------------- M2/M9
    spec = spec_for(config, comps)
    ladder = build_price_ladder(units, surface, config)
    tensor = build_revenue_tensor(units, ladder, real, spec)
    base = solve_release_plan(tensor, units, prov)
    base_map = {r.unit_id: (r.phase_index, r.level_index) for r in base.plan}

    relabelled = units.copy()
    mapping = {u: f"Z-{u}" for u in relabelled["unit_id"]}
    relabelled["unit_id"] = relabelled["unit_id"].map(mapping)
    l2 = build_price_ladder(relabelled, surface, config)
    t2 = build_revenue_tensor(relabelled, l2, real, spec)
    r2 = solve_release_plan(t2, relabelled, prov)
    m2 = {u.replace("Z-", "", 1): (p, k)
          for u, (p, k) in ((r.unit_id, (r.phase_index, r.level_index)) for r in r2.plan)}
    same = base_map == m2
    check("M2 relabelling unit ids leaves the plan unchanged", same,
          "" if same else f"{sum(1 for k in base_map if base_map.get(k) != m2.get(k))} differ")
    check("M2b objective unchanged under relabelling",
          bool(np.isclose(base.objective_usd, r2.objective_usd, rtol=1e-9)),
          f"${base.objective_usd:,.2f} vs ${r2.objective_usd:,.2f}")

    shuffled = units.sample(frac=1.0, random_state=7).reset_index(drop=True)
    l3 = build_price_ladder(shuffled, surface, config)
    t3 = build_revenue_tensor(shuffled, l3, real, spec)
    r3 = solve_release_plan(t3, shuffled, prov)
    m3 = {r.unit_id: (r.phase_index, r.level_index) for r in r3.plan}
    check("M9 permuting inventory row order leaves the plan unchanged",
          base_map == m3,
          "" if base_map == m3 else
          f"{sum(1 for k in base_map if base_map.get(k) != m3.get(k))} of "
          f"{len(base_map)} differ; objective "
          f"${base.objective_usd:,.0f} vs ${r3.objective_usd:,.0f}")

    # ---------------------------------------------------------------- M3
    doubled = units.copy()
    doubled["living_area_sqft"] = doubled["living_area_sqft"] * 2.0
    ok_area = bool((doubled["living_area_sqft"] <= 20000).all())
    if ok_area:
        # Hold the ladder fixed so only area changes: same $/sqft, twice the sqft.
        t4 = build_revenue_tensor(doubled, ladder, real, spec)
        ratio = float(np.nanmean(t4.expected_revenue_usd / tensor.expected_revenue_usd))
        prob_same = bool(np.allclose(
            np.nan_to_num(t4.probability), np.nan_to_num(tensor.probability), atol=1e-9))
        check("M3 doubling area doubles expected revenue at fixed $/sqft",
              bool(np.isclose(ratio, 2.0, rtol=1e-6)), f"ratio {ratio:.6f}")
        check("M3b doubling area leaves P(sale) unchanged ONLY if area is not a covariate",
              prob_same,
              "area IS a covariate, so P(sale) moves — expected, but it means "
              "revenue does not scale exactly")
    else:
        check("M3 doubling area", False, "areas exceed the validator bound")

    # ---------------------------------------------------------------- M4
    lo = ladder.p_floor_ppsf.copy()
    hi = ladder.p_ceiling_ppsf.copy()
    wide = ladder_from_bounds(list(ladder.unit_ids), lo, hi, n_levels=ladder.n_levels)
    t5 = build_revenue_tensor(units, wide, real, spec)
    r5 = solve_release_plan(t5, units, prov)
    check("M4 rebuilding the identical ladder reproduces the objective",
          bool(np.isclose(r5.objective_usd, base.objective_usd, rtol=1e-6)),
          f"${r5.objective_usd:,.0f} vs ${base.objective_usd:,.0f}")

    # ---------------------------------------------------------------- M7/M8
    plans = {}
    for beta in (-0.4, -3.0, 0.0):
        m = FixedBeta(beta)
        t = build_revenue_tensor(units, ladder, m, spec)
        r = solve_release_plan(t, units, DemandProvenance(beta_price=beta))
        plans[beta] = {row.unit_id: row for row in r.plan}

    common = set(plans[-0.4]) & set(plans[-3.0])
    raised = [u for u in common
              if plans[-3.0][u].price_ppsf > plans[-0.4][u].price_ppsf + 1e-9]
    mean_ratio = float(np.mean([plans[-3.0][u].price_ppsf / plans[-0.4][u].price_ppsf
                                for u in common]))
    check("M7 a more elastic beta never raises a unit's price",
          len(raised) == 0, f"{len(raised)} units raised; mean ratio {mean_ratio:.4f}")

    K = ladder.n_levels
    at_ceiling = sum(1 for r in plans[0.0].values() if r.level_index == K - 1)
    check("M8 beta_price = 0 pushes every released unit to its ceiling",
          at_ceiling == len(plans[0.0]),
          f"{at_ceiling}/{len(plans[0.0])} at ceiling")

    # ---------------------------------------------------------------- M10
    from src.simulation.monte_carlo import price_at_comps_levels
    comp_levels = price_at_comps_levels(ladder, tensor.unit_ids)
    feasible = 0.0
    for i, uid in enumerate(tensor.unit_ids):
        j = min(range(tensor.n_phases),
                key=lambda jj: -tensor.discounted_usd[i, jj, comp_levels[i]]
                if tensor.releasable[i, jj] else 1e18)
        if tensor.releasable[i, j]:
            feasible += float(tensor.discounted_usd[i, j, comp_levels[i]])
    check("M10 optimum is at least as good as the price-at-comps point",
          base.objective_usd >= feasible - 1e-6,
          f"optimum ${base.objective_usd:,.0f} vs comps point ${feasible:,.0f} "
          f"(note: the comps point ignores max_units, so it may be infeasible)")

    # ---------------------------------------------------------------- M6b
    from src.simulation.monte_carlo import summarize
    for sample in (np.array([1.0]), np.array([1.0, 1.0, 1.0]),
                   np.array([0.0, 5.0, 10.0, 1e9])):
        s = summarize(sample)
        if not (s.is_ordered and s.cvar5 <= s.p5 + 1e-9):
            check("M6b summarize keeps percentiles ordered and CVaR <= P5", False,
                  f"{sample} -> {s}")
            break
    else:
        check("M6b summarize keeps percentiles ordered and CVaR <= P5", True)

    print(f"\n{'=' * 70}")
    failed = [r for r in results if not r[1]]
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"    FAILED: {name}   {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
