"""A12 — Re-verification: the four numbers of brief §9.2, after the fix pass.

Reports, against the repaired system:

  1. beta_price estimate, SE, 95% CI            (the system's foundation)
  2. % of (submarket x month) cells with n >= 8 (signal or noise)
  3. % of units at the price ceiling            (economics or extrapolation)
  4. total unmodelled probability mass          (partial sum or total)

1 and 2 are computed on the real Miami export, because that is what the
coefficient would be calibrated on. 3 and 4 are computed on the example
inventory against a synthetic-data fit, because the real fit is not cleared for
producing a plan and the calibration gate is not this audit's to open.

Run from backend/:  .venv/bin/python ../audit/a12_headline_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import DEMAND_COVARIATES, build_features  # noqa: E402
from src.data.ingest_mls import ingest_mls  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.diagnostics import run_diagnostics  # noqa: E402
from src.demand.hedonic import fit_hedonic  # noqa: E402
from src.demand.survival import (  # noqa: E402
    CONTROLLED_CATEGORICALS, CoxDemandModel, available_covariates,
)
from src.optimizer.discretize import build_price_ladder  # noqa: E402
from src.optimizer.formulate import (  # noqa: E402
    Comps, DemandProvenance, Phase, ProjectSpec, build_revenue_tensor,
)
from src.optimizer.solve import solve_release_plan  # noqa: E402
from src.simulation.monte_carlo import simulate_plan  # noqa: E402
from src.simulation.scenarios import ScenarioSpec  # noqa: E402
from src.utils.validate import validate_inventory  # noqa: E402


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    config = load_market_config("miami")

    # ---------------------------------------------------------- 1 and 2
    rule("REAL EXPORT — numbers 1 and 2")
    ingested = ingest_mls(market="miami")  # 15 quarterly files
    feats = build_features(ingested.frame, config).frame

    model = CoxDemandModel(covariates=DEMAND_COVARIATES)
    fit = model.fit(feats)
    b = fit.beta_price
    print(f"\n[1] beta_price (naive covariate set)")
    print(f"      estimate {b.value:+.4f}   SE {b.std_error:.4f}   "
          f"95% CI [{b.ci_low:+.4f}, {b.ci_high:+.4f}]")
    print(f"      {'excludes' if b.excludes_zero else 'COVERS'} zero   "
          f"n={fit.n_observations}  events={fit.n_events}")

    controlled = CoxDemandModel(covariates=available_covariates(feats),
                                categoricals=CONTROLLED_CATEGORICALS)
    cfit = controlled.fit(feats)
    cb = cfit.beta_price
    print(f"\n    with hedonic controls")
    print(f"      estimate {cb.value:+.4f}   SE {cb.std_error:.4f}   "
          f"95% CI [{cb.ci_low:+.4f}, {cb.ci_high:+.4f}]")
    print(f"      n={cfit.n_observations}  events={cfit.n_events}")

    report = run_diagnostics(feats, cfit)
    print(f"\n    diagnostics status: {report.status}")
    print(f"    quality_explained_share: "
          f"{report.identification.quality_explained_share:.1%}")
    print(f"    brief gate is [-0.5, -3.0]: "
          f"{'INSIDE' if 0.5 <= abs(cb.value) <= 3.0 else 'OUTSIDE — too weak'}")

    keyed = feats[feats["submarket"].notna() & feats["list_month"].notna()]
    sizes = keyed.groupby(["submarket", "list_month"]).size()
    dense = int((sizes >= 8).sum())
    rows_dense = int(sizes[sizes >= 8].sum())
    print(f"\n[2] (submarket x month) cell density")
    print(f"      cells total            {len(sizes)}")
    print(f"      cells with n >= 8      {dense}  ({dense / len(sizes):.1%} of cells)")
    print(f"      rows in those cells    {rows_dense}  "
          f"({rows_dense / len(keyed):.1%} of keyed rows)")
    print(f"      median cell size       {sizes.median():.0f}")
    rel = feats["rel_price_premium"].dropna()
    print(f"      rel_price_premium: n={len(rel)}  IQR {rel.quantile(.75)-rel.quantile(.25):.4f}"
          f"  p1 {rel.quantile(.01):+.3f}  p99 {rel.quantile(.99):+.3f}"
          f"  max {rel.max():+.3f}")

    # ---------------------------------------------------------- 3 and 4
    rule("EXAMPLE INVENTORY, SYNTHETIC FIT — numbers 3 and 4")
    units = validate_inventory(
        pd.read_csv("data/project_inputs/example_inventory.csv"), config).frame
    synth = generate_synthetic_mls(n=4000, seed=13, market="miami", profile="rich")
    sframe = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    surface = fit_hedonic(sframe, config)
    ladder = build_price_ladder(units, surface, config)
    smodel = CoxDemandModel(covariates=available_covariates(sframe),
                            categoricals=CONTROLLED_CATEGORICALS)
    sfit = smodel.fit(sframe)
    spec = ProjectSpec.from_config(
        config, project_start=pd.Timestamp("2026-01-01"),
        phases=tuple(Phase(index=j, name=f"phase_{j+1}", start_month=float(m),
                           max_units=20, competing_listings=30.0)
                     for j, m in enumerate((0.0, 6.0, 12.0, 18.0))),
        comps=Comps({s: float(sframe.loc[sframe["submarket"] == s,
                                         "cell_median_ppsf"].median())
                     for s in units["submarket"].unique()}))
    tensor = build_revenue_tensor(units, ladder, smodel, spec)
    prov = DemandProvenance(demand_model=sfit.model_kind,
                            beta_price=sfit.beta_price.value,
                            beta_price_se=sfit.beta_price.std_error,
                            beta_price_ci95=(sfit.beta_price.ci_low, sfit.beta_price.ci_high))
    result = solve_release_plan(tensor, units, prov)

    K = tensor.n_levels
    levels = np.array([r.level_index for r in result.plan])
    ceiling = int((levels == K - 1).sum())
    floor = int((levels == 0).sum())
    print(f"\n[3] price-ladder position of released units  (K = {K} levels)")
    print(f"      at ceiling  {ceiling} / {len(levels)}  ({ceiling/len(levels):.1%})")
    print(f"      at floor    {floor} / {len(levels)}  ({floor/len(levels):.1%})")
    print(f"      mean level index {levels.mean():.2f} of {K-1}")
    print(f"      synthetic beta_price {sfit.beta_price.value:+.4f}")

    print(f"\n[4] unmodelled probability mass")
    print(f"      units released              {result.units_released}")
    print(f"      SUM(1 - D_i)                {result.unmodelled_probability_mass:.2f} units"
          f"  ({result.unmodelled_probability_mass / result.units_released:.1%})")
    print(f"      gross ask if all sold       ${result.gross_ask_usd:,.0f}")
    print(f"      expected revenue            ${result.expected_revenue_usd:,.0f}")
    print(f"      never modelled              ${result.unmodelled_revenue_usd:,.0f}"
          f"  ({result.unmodelled_revenue_usd / result.gross_ask_usd:.1%} of ask)")
    print(f"      discounted objective        ${result.objective_usd:,.0f}")

    # ---------------------------------------------------------- downstream
    rule("DOWNSTREAM RE-VERIFICATION")
    print(f"  extrapolation: {result.extrapolation.message}")
    print(f"  monotonicity violations: ", end="")
    from src.optimizer.solve import check_monotone
    print(len(check_monotone(result, units)))

    hz = {n: float(c.value) for n, c in sfit.coefficients.items()}
    sspec = ScenarioSpec(beta_price_mean=sfit.beta_price.value,
                         beta_price_se=sfit.beta_price.std_error,
                         absorption_log_hazard_sd=0.15, comps_drift_sd=0.03,
                         completion_delay_months_sd=1.0)
    dist = simulate_plan(result, tensor, units, sspec, n_draws=10_000, seed=13,
                         presale_lead_months=spec.presale_lead_months,
                         phase_dates=spec.phase_dates(), hazard_coefficients=hz,
                         ladder=ladder)
    d = dist.discounted_usd
    print(f"  revenue P5 ${d.p5:,.0f}  P50 ${d.p50:,.0f}  P95 ${d.p95:,.0f}  "
          f"CVaR@5% ${d.cvar5:,.0f}")
    print(f"  percentiles ordered: {d.is_ordered}   CVaR <= P5: {d.cvar5 <= d.p5}")
    up = dist.uplift_vs_baseline_usd
    print(f"  paired uplift vs pricing at comps: P50 ${up.p50:,.0f}   "
          f"90% band ${up.p5:,.0f} to ${up.p95:,.0f}")
    print(f"  P(plan beats baseline) = {dist.prob_beats_baseline:.1%}")
    if up.p5 <= 0 <= up.p95:
        print("  -> the uplift band CONTAINS ZERO: the gain is not distinguishable")
        print("     from repricing at comps, even under the model's own demand curve.")

    print(f"\n  provenance is_calibrated_on_real_data: "
          f"{result.provenance['is_calibrated_on_real_data']}")
    print(f"  caveats attached to the plan: {len(result.caveats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
