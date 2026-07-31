"""A08 — Optimizer audit: objective, units, timing, unsold mass, corners, MILP mechanics.

Covers brief §3.1-§3.5 plus a differential reimplementation of the objective:
a slow, obvious, loop-based recomputation from the extracted plan, compared
against what PuLP reports.

Run from backend/:  .venv/bin/python ../audit/a08_optimizer_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import build_features  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.hedonic import fit_hedonic  # noqa: E402
from src.demand.survival import (  # noqa: E402
    CONTROLLED_CATEGORICALS, CoxDemandModel, available_covariates,
)
from src.optimizer.constraints import ConstraintSet, comparable_groups  # noqa: E402
from src.optimizer.discretize import build_price_ladder  # noqa: E402
from src.optimizer.formulate import (  # noqa: E402
    Comps, DemandProvenance, Phase, ProjectSpec, build_revenue_tensor,
)
from src.optimizer.solve import (  # noqa: E402
    SolveStatus, build_program, check_monotone, solve_release_plan,
)
from src.utils.npv import discount_factor  # noqa: E402
from src.utils.validate import validate_inventory  # noqa: E402

INVENTORY = Path("data/project_inputs/example_inventory.csv")


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def build(config, beta_override=None, phase_months=(0.0, 6.0, 12.0, 18.0)):
    units = validate_inventory(pd.read_csv(INVENTORY), config).frame
    synth = generate_synthetic_mls(n=4000, seed=13, market="miami", profile="rich")
    frame = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    surface = fit_hedonic(frame, config)
    ladder = build_price_ladder(units, surface, config)
    model = CoxDemandModel(covariates=available_covariates(frame),
                           categoricals=CONTROLLED_CATEGORICALS)
    fit = model.fit(frame)
    spec = ProjectSpec.from_config(
        config,
        project_start=pd.Timestamp("2026-01-01"),
        phases=tuple(Phase(index=j, name=f"phase_{j+1}", start_month=float(m),
                           max_units=20, competing_listings=30.0)
                     for j, m in enumerate(phase_months)),
        comps=Comps({s: float(frame.loc[frame["submarket"] == s,
                                        "cell_median_ppsf"].median())
                     for s in units["submarket"].unique()}),
    )
    tensor = build_revenue_tensor(units, ladder, model, spec)
    return units, frame, surface, ladder, model, fit, spec, tensor


def main() -> int:
    config = load_market_config("miami")
    units, frame, surface, ladder, model, fit, spec, tensor = build(config)
    prov = DemandProvenance(demand_model=fit.model_kind,
                            beta_price=fit.beta_price.value,
                            beta_price_se=fit.beta_price.std_error,
                            beta_price_ci95=(fit.beta_price.ci_low, fit.beta_price.ci_high))
    result = solve_release_plan(tensor, units, prov)

    rule("3.1  Objective: differential reimplementation from the extracted plan")
    print(f"  status {result.status.value}   units {result.units_released}/{tensor.n_units}")
    print(f"  PuLP objective            ${result.objective_usd:,.2f}")
    manual = 0.0
    for row in result.plan:
        i = tensor.unit_ids.index(row.unit_id)
        j, k = row.phase_index, row.level_index
        p = float(tensor.price_ppsf[i, k])
        a = float(tensor.area_sqft[i])
        d = float(tensor.probability[i, j, k])
        delta = discount_factor(spec.phases[j].start_month, spec.discount_rate_annual)
        manual += p * a * d * delta
    print(f"  loop-based recomputation  ${manual:,.2f}")
    print(f"  difference                ${abs(manual - result.objective_usd):,.6f}")
    print(f"  sum of row fields         "
          f"${sum(r.discounted_expected_revenue_usd for r in result.plan):,.2f}")

    rule("3.1b Unit consistency and order of magnitude")
    gdv = float((units["living_area_sqft"] * ladder.predicted_ppsf.mean()).sum())
    print(f"  inventory area            {units['living_area_sqft'].sum():,.0f} sqft")
    print(f"  mean predicted $/sqft     ${ladder.predicted_ppsf.mean():,.0f}")
    print(f"  notional GDV at comps     ${gdv:,.0f}")
    print(f"  undiscounted expected rev ${result.expected_revenue_usd:,.0f}")
    print(f"  ratio to GDV              {result.expected_revenue_usd / gdv:.3f}"
          f"   (must be < 1: it is GDV x P(sale))")
    if not (0.01 < result.expected_revenue_usd / gdv < 1.5):
        print("  !! ORDER OF MAGNITUDE LOOKS WRONG (sqft/sqm or per-area/total confusion)")
    ratio_sqm = result.expected_revenue_usd / gdv * 10.7639
    print(f"  ratio if an sqft/sqm swap had happened would be {ratio_sqm:.1f}")

    rule("3.1c Discount factor units")
    for m in (0.0, 6.0, 12.0, 18.0):
        print(f"    month {m:>5.1f}  delta = {discount_factor(m, 0.12):.6f}"
              f"   (annual 12% -> 1/1.12^(m/12))")
    print(f"  tensor discount_factor: {np.round(tensor.discount_factor, 6).tolist()}")
    expected = [discount_factor(p.start_month, spec.discount_rate_annual) for p in spec.phases]
    print(f"  independently computed : {np.round(expected, 6).tolist()}")
    print(f"  match: {np.allclose(tensor.discount_factor, expected)}")

    rule("3.1d Timing mismatch: D is P(sale within T), delta discounts to release")
    print(f"  horizon_days = {spec.horizon_days} ({spec.horizon_days/30.4375:.1f} months)")
    print("  Revenue from a unit released at phase j actually arrives spread over")
    print("  [t_j, t_j + T]. The objective discounts all of it to t_j, so late cash is")
    print("  under-discounted. Size the error: discounting instead to the midpoint")
    mid_extra = spec.horizon_days / 30.4375 / 2.0
    tot_at_release = sum(r.discounted_expected_revenue_usd for r in result.plan)
    tot_at_mid = 0.0
    for row in result.plan:
        m = spec.phases[row.phase_index].start_month + mid_extra
        tot_at_mid += row.expected_revenue_usd * discount_factor(m, spec.discount_rate_annual)
    print(f"    discounted to release   ${tot_at_release:,.0f}")
    print(f"    discounted to midpoint  ${tot_at_mid:,.0f}")
    print(f"    overstatement           ${tot_at_release - tot_at_mid:,.0f}"
          f"  ({(tot_at_release / tot_at_mid - 1):.2%})")

    rule("3.2  The unsold-unit problem: unmodelled probability mass")
    probs = np.array([r.sale_probability for r in result.plan])
    unmodelled = float((1.0 - probs).sum())
    lost = float(sum(r.expected_revenue_usd * (1 - r.sale_probability) / r.sale_probability
                     for r in result.plan if r.sale_probability > 0))
    gross = float(sum(r.total_price_usd for r in result.plan))
    print(f"  units released            {len(probs)}")
    print(f"  mean P(sale)              {probs.mean():.4f}")
    print(f"  min / max P(sale)         {probs.min():.4f} / {probs.max():.4f}")
    print(f"  SUM(1 - P) unmodelled     {unmodelled:.2f} units"
          f"  ({unmodelled / len(probs):.1%} of the released stack)")
    print(f"  gross ask if all sold     ${gross:,.0f}")
    print(f"  expected revenue          ${result.expected_revenue_usd:,.0f}")
    print(f"  value never modelled      ${gross - result.expected_revenue_usd:,.0f}"
          f"  ({1 - result.expected_revenue_usd / gross:.1%} of ask)")
    print("  In the MILP the unsold share simply vanishes: C1 releases a unit at most")
    print("  once and there is no carry-over, so expected revenue is a partial sum.")

    rule("3.4  Corner solutions: where on its ladder did each unit land?")
    K = tensor.n_levels
    levels = np.array([r.level_index for r in result.plan])
    at_ceiling = int((levels == K - 1).sum())
    at_floor = int((levels == 0).sum())
    print(f"  ladder levels K = {K}")
    print(f"  at ceiling (k=K-1)  {at_ceiling:>3d} / {len(levels)}  "
          f"({at_ceiling / len(levels):.1%})")
    print(f"  at floor   (k=0)    {at_floor:>3d} / {len(levels)}  "
          f"({at_floor / len(levels):.1%})")
    print(f"  level histogram: {np.bincount(levels, minlength=K).tolist()}")
    print(f"  mean level index    {levels.mean():.2f} of {K-1}")

    rule("3.4b Price ladder vs the support of the estimated data")
    sup = tensor.premium_support
    prem = tensor.rel_price_premium
    print(f"  fitted support: low {sup.low:+.4f}  p1 {sup.p1:+.4f}  "
          f"p99 {sup.p99:+.4f}  high {sup.high:+.4f}")
    print(f"  ladder grid premiums: min {np.nanmin(prem):+.4f}  max {np.nanmax(prem):+.4f}")
    outside = int(sup.outside(prem[np.isfinite(prem)]).sum())
    total = int(np.isfinite(prem).sum())
    print(f"  grid points outside the fitted range : {outside} of {total}"
          f"  ({outside/total:.2%})")
    tail = int(sup.in_tail(prem[np.isfinite(prem)]).sum())
    print(f"  grid points in the thin tails        : {tail} of {total} ({tail/total:.2%})")
    print(f"  extrapolation report on the PLAN     : {result.extrapolation.message}")

    rule("3.5  MILP mechanics")
    program = build_program(tensor, units, ConstraintSet())
    print(f"  binaries              {len(program.y):,}")
    print(f"  constraints           {len(program.problem.constraints):,}")
    names = [v.name for v in program.problem.variables()]
    print(f"  variables             {len(names):,}   distinct names {len(set(names)):,}")
    if len(names) != len(set(names)):
        print("  !! DUPLICATE VARIABLE NAMES")
    cnames = list(program.problem.constraints.keys())
    print(f"  constraint names      {len(cnames):,}   distinct {len(set(cnames)):,}")
    if len(cnames) != len(set(cnames)):
        print("  !! DUPLICATE CONSTRAINT NAMES")

    print("\n  integer tolerance on the extracted solution:")
    vals = np.array([(v.value() or 0.0) for v in program.y.values()])
    program.problem.solve(pulp.PULP_CBC_CMD(msg=False))
    vals = np.array([(v.value() or 0.0) for v in program.y.values()])
    frac = vals[(vals > 1e-6) & (vals < 1 - 1e-6)]
    print(f"    values strictly between 0 and 1 : {len(frac)}")
    if len(frac):
        print(f"    worst fractional value        : {frac.max():.10f}")
    per_unit = np.zeros(tensor.n_units)
    for (i, j, k), v in program.y.items():
        per_unit[i] += (v.value() or 0.0)
    print(f"    max SUM_j SUM_k y[i,j,k]        : {per_unit.max():.10f}"
          f"   (must be <= 1)")
    print(f"    units with sum > 1 + 1e-6       : {int((per_unit > 1 + 1e-6).sum())}")

    print("\n  monotonicity verified on the extracted plan:")
    viol = check_monotone(result, units)
    print(f"    violations: {len(viol)}")
    for v in viol[:5]:
        print(f"      {v}")

    rule("3.5b Solver status handling on a deliberately infeasible instance")
    huge = ProjectSpec.from_config(
        config, project_start=pd.Timestamp("2026-01-01"),
        phases=tuple(Phase(index=j, name=f"p{j}", start_month=float(m),
                           cash_flow_floor_usd=1e12, max_units=20,
                           competing_listings=30.0)
                     for j, m in enumerate((0.0, 6.0, 12.0, 18.0))),
        comps=spec.comps,
    )
    t2 = build_revenue_tensor(units, ladder, model, huge)
    r2 = solve_release_plan(t2, units, prov)
    print(f"  status            {r2.status.value}")
    print(f"  plan rows         {len(r2.plan)}")
    print(f"  infeasibility set {r2.infeasibility is not None}")
    if r2.infeasibility:
        print(f"  binding family    {r2.infeasibility.binding_family}")
        print(f"  tested            {r2.infeasibility.tested}")

    rule("3.5c Degenerate inputs")
    for label, kw in (("one phase", dict(phase_months=(0.0,))),):
        u2, *_rest, s2, t3 = build(config, **kw)
        r3 = solve_release_plan(t3, u2, prov)
        print(f"  {label}: status={r3.status.value} units={r3.units_released} "
              f"objective=${r3.objective_usd:,.0f} monotone_viol="
              f"{len(check_monotone(r3, u2))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
