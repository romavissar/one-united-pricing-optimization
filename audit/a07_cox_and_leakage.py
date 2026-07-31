"""A07 — Cox internals, leakage, and the survival-probability conversion (§2.1, §2.4, §2.5).

Checks, in order:

  1. the exact feature matrix at fit time — column names off the array, not the docstring
  2. which price and area fields the premium is actually built from
  3. the survival conversion: does `predict_sale_probability` agree with a
     hand-rolled S0(t)^exp(beta'x) using *centered* covariates, and with
     lifelines' own `predict_survival_function`?
  4. extrapolation of the baseline hazard past the last observed event time
  5. proportional-hazards test on rel_price_premium
  6. collinearity: condition number and VIFs
  7. leakage: is any post-outcome column reachable from the design matrix?
  8. the list_date reconstruction identity, per status and per terminal-date choice

Run from backend/:  .venv/bin/python ../audit/a07_cox_and_leakage.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import DEMAND_COVARIATES, build_features  # noqa: E402
from src.data.ingest_mls import ingest_mls  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402
from src.data.synth import generate_synthetic_mls  # noqa: E402
from src.demand.survival import CoxDemandModel  # noqa: E402

POST_OUTCOME = {
    "days_on_market", "cumulative_days_on_market", "close_price", "close_ppsf",
    "close_date", "pending_date", "off_market_date", "last_list_price",
    "price_cut_pct", "sold_to_list_ratio", "duration_days", "event_sold",
    "duration_source", "status",
}


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    config = load_market_config("miami")

    # Synthetic frame — clean case where truth is known, used for 3 and 4.
    synth = generate_synthetic_mls(n=6000, seed=13, true_beta_price=-1.6,
                                   hazard_basis="realized")
    sframe = build_features(normalize_mls(synth.frame, config=config).frame, config).frame
    model = CoxDemandModel(covariates=DEMAND_COVARIATES)
    fit = model.fit(sframe)

    rule("1  The feature matrix as fitted (column names off the array)")
    X = fit.design.X
    print(f"  shape {X.shape}   rows_in {fit.design.rows_in} -> rows_used {fit.design.rows_used}")
    print(f"  columns ({X.shape[1]}):")
    for c in X.columns:
        print(f"    {c}")
    print(f"\n  scale factors applied: "
          f"{ {k: round(v, 4) for k, v in list(fit.design.scale.items())[:6]} }")
    print(f"  dropped_constant : {fit.design.dropped_constant}")
    print(f"  dropped_dependent: {fit.design.dropped_dependent}")

    rule("2  Leakage: is any post-outcome column in the design?")
    leaked = sorted({c.split("=")[0] for c in X.columns} & POST_OUTCOME)
    print(f"  post-outcome columns present in design: {leaked or 'NONE'}")
    print(f"  covariates requested: {list(fit.covariates)}")
    inv = sorted(set(DEMAND_COVARIATES) & POST_OUTCOME)
    print(f"  post-outcome names in DEMAND_COVARIATES: {inv or 'NONE'}")
    print("\n  inventory_competition definition check:")
    print("    counts other listings ENTERING the same (submarket, month).")
    print("    Fixed at listing time -> not a function of any realized duration.")
    print("    It is still not point-in-time available to a seller on day 0 of the")
    print("    month (the month is not over yet). Relevant to a backtest, if one existed.")

    rule("3  Survival conversion: does predict_sale_probability match lifelines?")
    horizon = 180
    score = sframe.loc[fit.design.index].head(200).copy()
    got = model.predict_sale_probability(score, score["list_ppsf"].to_numpy(), horizon)
    # lifelines' own path, via the model's design transform
    from src.demand.base import relative_premium, transform_to_design
    scored = score.copy()
    scored["rel_price_premium"] = relative_premium(
        score["list_ppsf"].to_numpy(), score["cell_median_ppsf"].to_numpy())
    matrix, usable = transform_to_design(scored, model._design)
    ref = 1.0 - model._fitter.predict_survival_function(
        matrix, times=[float(horizon)]).to_numpy().ravel()
    a = got[np.flatnonzero(usable.to_numpy())]
    print(f"  rows compared          : {len(a)}")
    print(f"  max abs difference     : {np.nanmax(np.abs(a - ref)):.3e}")

    # Hand-rolled: S(t|x) = S0(t) ** exp(beta'(x - xbar)). lifelines centers.
    beta = model._fitter.params_.to_numpy()
    xbar = model._fitter._norm_mean.to_numpy()
    bh = model._fitter.baseline_survival_
    s0 = float(np.interp(horizon, bh.index.to_numpy(),
                         bh.iloc[:, 0].to_numpy()))
    eta_centered = (matrix.to_numpy() - xbar) @ beta
    manual_centered = 1.0 - s0 ** np.exp(eta_centered)
    eta_raw = matrix.to_numpy() @ beta
    manual_raw = 1.0 - s0 ** np.exp(eta_raw)
    print(f"  hand-rolled CENTERED   : max abs diff vs lifelines "
          f"{np.nanmax(np.abs(manual_centered - ref)):.3e}   <- should be ~0")
    print(f"  hand-rolled UNCENTERED : max abs diff vs lifelines "
          f"{np.nanmax(np.abs(manual_raw - ref)):.3e}   <- the classic bug, if used")
    print(f"  mean P(sale) reported  : {np.nanmean(a):.4f}")
    print(f"  mean P(sale) uncentered: {np.nanmean(manual_raw):.4f}")
    print("  -> the shipped path delegates to lifelines, so centering is handled.")

    rule("4  Baseline hazard beyond the last observed event time")
    bh_index = model._fitter.baseline_survival_.index.to_numpy()
    print(f"  baseline survival defined on t in [{bh_index.min():.0f}, {bh_index.max():.0f}] days")
    print(f"  max observed duration in fit : {sframe.loc[fit.design.index, 'duration_days'].max():.0f}")
    for t in (30, 180, 365, 720, 1200, 3650):
        s = model._fitter.predict_survival_function(
            matrix.head(1), times=[float(t)]).to_numpy().ravel()[0]
        flag = "" if t <= bh_index.max() else "   <- BEYOND SUPPORT"
        print(f"    P(sale by {t:>5d}d) = {1 - s:.6f}{flag}")
    print("  lifelines holds the last value past the final event time, so the hazard")
    print("  silently flattens to zero. A horizon beyond the data returns a number.")

    rule("5  Proportional hazards test (Schoenfeld)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = model._fitter.check_assumptions(
                model._fitter._central_values if False else None, show_plots=False)
        except Exception:
            res = None
    # check_assumptions needs the training frame; rebuild it.
    fit_frame = model._design.X.copy()
    fit_frame["duration_days"] = sframe.loc[model._design.index, "duration_days"].to_numpy()
    fit_frame["event_sold"] = sframe.loc[model._design.index, "event_sold"].astype(int).to_numpy()
    from lifelines.statistics import proportional_hazard_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = proportional_hazard_test(model._fitter, fit_frame, time_transform="rank")
    summ = ph.summary.sort_values("p")
    print("  worst 8 covariates by p-value:")
    print(summ.head(8).to_string())
    if "rel_price_premium" in summ.index:
        row = summ.loc["rel_price_premium"]
        print(f"\n  rel_price_premium: test_stat={float(row['test_statistic']):.3f}  "
              f"p={float(row['p']):.4f}  -> "
              f"{'VIOLATES PH' if float(row['p']) < 0.05 else 'no PH violation detected'}")

    rule("6  Collinearity of the design matrix")
    M = X.to_numpy(dtype=float)
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"  condition number: {sv.max() / sv.min():.1f}")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    vifs = []
    for i, name in enumerate(Xc.columns):
        if name == "const":
            continue
        try:
            vifs.append((name, variance_inflation_factor(Xc.to_numpy(), i)))
        except Exception:
            vifs.append((name, float("nan")))
    for name, v in sorted(vifs, key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))[:8]:
        print(f"    VIF {v:>10.2f}  {name}")

    # ---------------------------------------------------------------- real data
    rule("7  The same, on the REAL export")
    real = build_features(ingest_mls(market="miami").frame, config).frame
    rmodel = CoxDemandModel(covariates=DEMAND_COVARIATES)
    try:
        rfit = rmodel.fit(real)
        b = rfit.beta_price
        print(f"  beta_price = {b.value:+.4f}  se {b.std_error:.4f}  "
              f"95% CI [{b.ci_low:+.4f}, {b.ci_high:+.4f}]  "
              f"{'excludes' if b.excludes_zero else 'COVERS'} zero")
        print(f"  n = {rfit.n_observations}  events = {rfit.n_events}  "
              f"concordance = {rfit.fit_stats['concordance']:.4f}")
        print(f"  rows_in {rfit.design.rows_in} -> rows_used {rfit.design.rows_used} "
              f"({rfit.design.retention:.1%} retained)")
        print(f"  premium support seen at fit: "
              f"{rmodel.premium_support.as_dict() if rmodel.premium_support else None}")
    except Exception as exc:
        print(f"  fit failed: {type(exc).__name__}: {exc}")

    rule("8  list_date reconstruction identity, by terminal-date choice")
    raw = pd.read_csv("data/raw/mls/PRICING_MODEL_export.csv", dtype=str)
    norm = normalize_mls(raw, config=config).frame
    dom = pd.to_numeric(norm["days_on_market"], errors="coerce")
    for label, col in (("pending_date", "pending_date"),
                       ("close_date", "close_date"),
                       ("off_market_date", "off_market_date")):
        implied = norm[col] - pd.to_timedelta(dom, unit="D")
        err = (implied - norm["list_date"]).dt.days
        ok = err.notna()
        if not ok.any():
            continue
        print(f"\n  list_date == {label} - DOM")
        for status, g in norm.assign(err=err).groupby("status"):
            e = g["err"].dropna()
            if len(e) < 5:
                continue
            print(f"    {status:<10} n={len(e):>5d}  exact={float((e == 0).mean()):6.2%}  "
                  f"|err|<=1 {float((e.abs() <= 1).mean()):6.2%}  median {e.median():+.0f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
