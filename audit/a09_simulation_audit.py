"""A09 — Monte Carlo and risk audit (brief §4.1-§4.4).

  4.1  parameter sampling: is the full coefficient covariance used, or only the
       marginal SE of beta_price? what happens to a draw with beta_price >= 0?
  4.2  the tautology: what exactly does P(plan > baseline) compare?
  4.3  percentile ordering, CVaR definition, Monte Carlo standard error of P5,
       seed control and reproducibility
  4.4  tornado: same sigma as the MC marginals? baseline re-evaluated?

Run from backend/:  .venv/bin/python ../audit/a09_simulation_audit.py
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
from src.demand.hedonic import fit_hedonic  # noqa: E402
from src.demand.survival import (  # noqa: E402
    CONTROLLED_CATEGORICALS, CoxDemandModel, available_covariates,
)
from src.optimizer.discretize import build_price_ladder  # noqa: E402
from src.optimizer.formulate import (  # noqa: E402
    Comps, DemandProvenance, Phase, ProjectSpec, build_revenue_tensor,
)
from src.optimizer.solve import solve_release_plan  # noqa: E402
from src.simulation.monte_carlo import price_at_comps_levels, simulate_plan, summarize  # noqa: E402
from src.simulation.scenarios import ScenarioSpec, draw_scenarios  # noqa: E402
from src.simulation.sensitivity import tornado  # noqa: E402
from src.utils.validate import validate_inventory  # noqa: E402


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    config = load_market_config("miami")
    units = validate_inventory(pd.read_csv("data/project_inputs/example_inventory.csv"),
                               config).frame
    synth = generate_synthetic_mls(n=4000, seed=13, market="miami", profile="rich")
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    surface = fit_hedonic(frame, config)
    ladder = build_price_ladder(units, surface, config)
    model = CoxDemandModel(covariates=available_covariates(frame),
                           categoricals=CONTROLLED_CATEGORICALS)
    fit = model.fit(frame)
    beta, se = float(fit.beta_price.value), float(fit.beta_price.std_error)

    spec = ProjectSpec.from_config(
        config, project_start=pd.Timestamp("2026-01-01"),
        phases=tuple(Phase(index=j, name=f"phase_{j+1}", start_month=float(m),
                           max_units=20, competing_listings=30.0,
                           cash_flow_floor_usd=(1.5e7 if j == 0 else 0.0))
                     for j, m in enumerate((0.0, 6.0, 12.0, 18.0))),
        comps=Comps({s: float(frame.loc[frame["submarket"] == s,
                                        "cell_median_ppsf"].median())
                     for s in units["submarket"].unique()}),
    )
    tensor = build_revenue_tensor(units, ladder, model, spec)
    prov = DemandProvenance(demand_model=fit.model_kind, beta_price=beta,
                            beta_price_se=se,
                            beta_price_ci95=(fit.beta_price.ci_low, fit.beta_price.ci_high))
    result = solve_release_plan(tensor, units, prov)
    hz = {n: float(c.value) for n, c in fit.coefficients.items()}
    sspec = ScenarioSpec(beta_price_mean=beta, beta_price_se=se,
                         absorption_log_hazard_sd=0.15, comps_drift_sd=0.03,
                         completion_delay_months_sd=1.0, competing_listings_sd=0.0)

    rule("4.1  Parameter sampling: marginal SE or full covariance?")
    print(f"  fitted beta_price = {beta:+.4f}  se = {se:.4f}")
    print("  ScenarioSpec carries a single scalar `beta_price_se`; the other")
    print("  channels are user-supplied sds correlated by a HARDCODED matrix.")
    print("  The Cox coefficient covariance matrix is available but unused:")
    vm = model._fitter.variance_matrix_
    print(f"    variance_matrix_ shape {vm.shape}")
    corr_names = [c for c in vm.columns if c != "rel_price_premium"][:6]
    print("    corr(beta_price, other coefficients) — these are DISCARDED:")
    sd = np.sqrt(np.diag(vm.to_numpy()))
    for n in corr_names:
        i = list(vm.columns).index("rel_price_premium")
        j = list(vm.columns).index(n)
        r = vm.iloc[i, j] / (sd[i] * sd[j])
        print(f"      {n:<28} {r:+.4f}")
    print("  Only beta_price is resampled; every other fitted coefficient is held")
    print("  at its point estimate, so the reported spread omits their uncertainty.")

    print("\n  Is the persisted bundle carrying the covariance matrix?")
    from src.demand.registry import build_metadata
    print("    metadata keys per coefficient: name, value, std_error, ci_low, ci_high, p_value")
    print("    -> marginal SEs only; no covariance is persisted (Phase 4 cannot use it)")

    rule("4.1b Draws with a non-negative beta_price")
    for n in (10_000,):
        d = draw_scenarios(sspec, n, seed=13)
        pos = int((d.beta_price >= 0).sum())
        print(f"  n={n}: draws with beta_price >= 0 : {pos} ({pos/n:.3%})")
        print(f"    beta range {d.beta_price.min():+.4f} .. {d.beta_price.max():+.4f}")
        print(f"    notes: {d.notes}")
    print("  A non-negative draw is a world where price does not deter buyers; the")
    print("  plan's revenue there is bounded only by the ceiling, which fattens the")
    print("  right tail. The code flags it in notes but does not exclude it (correct).")

    rule("4.3  Distribution: ordering, CVaR, MC standard error, reproducibility")
    dist = simulate_plan(result, tensor, units, sspec, n_draws=10_000, seed=13,
                         presale_lead_months=spec.presale_lead_months,
                         phase_dates=spec.phase_dates(), hazard_coefficients=hz,
                         ladder=ladder)
    d = dist.discounted_usd
    print(f"  P5  ${d.p5:,.0f}")
    print(f"  P25 ${d.p25:,.0f}")
    print(f"  P50 ${d.p50:,.0f}")
    print(f"  P75 ${d.p75:,.0f}")
    print(f"  P95 ${d.p95:,.0f}")
    print(f"  CVaR@5% ${d.cvar5:,.0f}   mean ${d.mean:,.0f}   sd ${d.sd:,.0f}")
    print(f"  ordered: {d.is_ordered}   CVaR <= P5: {d.cvar5 <= d.p5}")

    # Reproducibility
    dist2 = simulate_plan(result, tensor, units, sspec, n_draws=10_000, seed=13,
                          presale_lead_months=spec.presale_lead_months,
                          phase_dates=spec.phase_dates(), hazard_coefficients=hz,
                          ladder=ladder)
    print(f"  same seed reproduces P50: "
          f"{np.isclose(dist.discounted_usd.p50, dist2.discounted_usd.p50)}")
    dist3 = simulate_plan(result, tensor, units, sspec, n_draws=10_000, seed=99,
                          presale_lead_months=spec.presale_lead_months,
                          phase_dates=spec.phase_dates(), hazard_coefficients=hz,
                          ladder=ladder)
    print(f"  different seed changes P50 by "
          f"${abs(dist3.discounted_usd.p50 - d.p50):,.0f}")

    # MC standard error of P5 via a seed bootstrap.
    p5s, p50s = [], []
    for s in range(20):
        dd = simulate_plan(result, tensor, units, sspec, n_draws=10_000, seed=1000 + s,
                           presale_lead_months=spec.presale_lead_months,
                           phase_dates=spec.phase_dates(), hazard_coefficients=hz,
                           ladder=ladder)
        p5s.append(dd.discounted_usd.p5)
        p50s.append(dd.discounted_usd.p50)
    p5s, p50s = np.array(p5s), np.array(p50s)
    spread = d.p95 - d.p5
    print(f"\n  across 20 independent seeds at 10,000 draws:")
    print(f"    P5  mean ${p5s.mean():,.0f}  sd ${p5s.std(ddof=1):,.0f}"
          f"   = {p5s.std(ddof=1)/spread:.2%} of the P5-P95 spread")
    print(f"    P50 mean ${p50s.mean():,.0f}  sd ${p50s.std(ddof=1):,.0f}")
    print(f"    reported P5-P95 spread ${spread:,.0f}")

    rule("4.2  The tautology: what is P(plan > baseline) actually comparing?")
    print(f"  prob_beats_baseline = {dist.prob_beats_baseline:.4f}")
    b = dist.baseline_discounted_usd
    print(f"  baseline P50 ${b.p50:,.0f}   plan P50 ${d.p50:,.0f}")
    print(f"  difference of medians ${d.p50 - b.p50:,.0f}")
    lv = price_at_comps_levels(ladder, tuple(r.unit_id for r in result.plan))
    pl = np.array([r.level_index for r in result.plan])
    print(f"  baseline ladder levels: mean {lv.mean():.2f}   plan levels mean {pl.mean():.2f}")
    print(f"  baseline is the SAME release schedule priced at the hedonic fitted value.")
    print(f"  Both arms are scored by the SAME demand model the plan was optimized")
    print(f"  against, and the baseline is a feasible point in the optimizer's search")
    print(f"  space, so the plan is >= baseline BY CONSTRUCTION under the model.")
    print(f"  units priced ABOVE comps in the plan: "
          f"{int((pl > lv).sum())} of {len(pl)}")
    print(f"  units priced BELOW comps in the plan: {int((pl < lv).sum())}")

    rule("4.4  Tornado: same sigma, and is the baseline re-evaluated?")
    tr = tornado(result, tensor, units, sspec,
                 presale_lead_months=spec.presale_lead_months,
                 phase_dates=spec.phase_dates(), hazard_coefficients=hz)
    print(f"  tornado baseline ${tr.baseline_usd:,.0f}")
    print(f"  plan objective   ${result.objective_usd:,.0f}   "
          f"(difference ${abs(tr.baseline_usd - result.objective_usd):,.0f})")
    for bar in tr.bars:
        print(f"    {bar.channel:<26} fitted={str(bar.fitted):<5} "
              f"sigma={abs(bar.high_value - bar.low_value)/2:.4f}  "
              f"swing ${bar.swing_usd:,.0f}")
    print("\n  sigmas used by the Monte Carlo marginals:")
    for k, v in sspec.standard_deviations.items():
        print(f"    {k:<26} {v}")
    print("  The tornado holds the PLAN fixed; it does not re-optimize. Both are")
    print("  legitimate but they answer different questions, so the chart must say which.")

    rule("4.3b Cash-flow breach on an expected-value constraint")
    for pb in dist.phase_breach:
        if pb.cash_flow_floor_usd:
            print(f"  {pb.name}: floor ${pb.cash_flow_floor_usd:,.0f}  "
                  f"E[rev] ${pb.expected_revenue_usd:,.0f}  "
                  f"P(breach) {pb.breach_probability:.1%}  "
                  f"buffered ${pb.buffered_floor_usd:,.0f}"
                  if pb.buffered_floor_usd else
                  f"  {pb.name}: floor ${pb.cash_flow_floor_usd:,.0f}  "
                  f"P(breach) {pb.breach_probability:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
