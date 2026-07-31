"""A06 — What population is the export actually a sample of? (brief §2.3, §2.6)

Three things A05 turned up that need pinning down precisely, because each one
changes what `beta_price` is an estimate *of*:

  1. Every terminal date in the file falls in a ~9.5-month window while list
     dates span 27 months. If true, the export is a sample of listings that
     *went off market* inside a calendar window, not a sample of listings that
     *started* in one. That is stock sampling on the outcome.
  2. The export is sorted by Status, and it is capped at exactly 5,000 rows
     against a search that returned "5000+". A cap applied to a status-sorted
     result truncates on the outcome variable.
  3. `rel_price_premium` has sd ~1.9 on a variable whose IQR is ~0.6. Something
     is generating extreme premiums; they sit directly on the coefficient.

Also tests whether the missing `list_date` on PENDING/WITHDRAWN rows can be
reconstructed from the identity `list_date = terminal_date - days_on_market`,
validated on the statuses where all three are present.

Run from backend/:  .venv/bin/python ../audit/a06_sampling_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.features import build_features  # noqa: E402
from src.data.ingest_mls import ingest_mls  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402

RAW = Path("data/raw/mls/PRICING_MODEL_export.csv")


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    config = load_market_config("miami")
    raw = pd.read_csv(RAW, dtype=str)
    norm = normalize_mls(raw, config=config).frame

    # ------------------------------------------------------------------ 1
    rule("1  What window is the sample drawn on?")
    terminal = norm[["close_date", "pending_date", "off_market_date"]].max(axis=1)
    norm = norm.assign(terminal=terminal)
    print(f"  list_date      : {norm['list_date'].min().date()} .. "
          f"{norm['list_date'].max().date()}   "
          f"(span {(norm['list_date'].max() - norm['list_date'].min()).days} days)")
    print(f"  terminal date  : {terminal.min().date()} .. {terminal.max().date()}   "
          f"(span {(terminal.max() - terminal.min()).days} days)")
    print(f"  rows with a terminal date : {int(terminal.notna().sum())} of {len(norm)}")
    print(f"  status ACTIVE             : {int((norm['status'] == 'ACTIVE').sum())}")

    print("\n  terminal date by status:")
    for status, grp in norm.groupby("status"):
        t = grp["terminal"].dropna()
        if len(t):
            print(f"    {status:<10} n={len(grp):>5d}  {t.min().date()} .. {t.max().date()}")

    print("\n  Every spell in the file is COMPLETE (it went off market). There are no")
    print("  listings still on the market at the export date, so a slow-selling unit")
    print("  that had not yet terminated is absent from the sample entirely.")

    # Observable-duration ceiling implied by the window.
    w0, w1 = terminal.min(), terminal.max()
    have_ld = norm["list_date"].notna() & norm["terminal"].notna()
    sub = norm.loc[have_ld].copy()
    sub["max_observable"] = (w1 - sub["list_date"]).dt.days
    sub["min_observable"] = (w0 - sub["list_date"]).dt.days.clip(lower=0)
    sub["dur"] = (sub["terminal"] - sub["list_date"]).dt.days
    print(f"\n  For a listing started on date L, only durations in "
          f"[max(0, W0-L), W1-L] are observable.")
    print(f"  W0 = {w0.date()}, W1 = {w1.date()}")
    started_before_w0 = int((sub["list_date"] < w0).sum())
    print(f"  listings started before W0 : {started_before_w0} "
          f"({started_before_w0 / len(sub):.1%})")
    print(f"    their minimum possible observed duration is "
          f"{sub.loc[sub['list_date'] < w0, 'min_observable'].min():.0f} .. "
          f"{sub.loc[sub['list_date'] < w0, 'min_observable'].max():.0f} days")
    print(f"    -> a 2024 listing can only appear here if it lasted "
          f"{sub.loc[sub['list_date'] < w0, 'min_observable'].max():.0f}+ days")
    print(f"  observed duration by start cohort:")
    sub["cohort"] = sub["list_date"].dt.to_period("Q").astype(str)
    for coh, g in sub.groupby("cohort"):
        print(f"    {coh}  n={len(g):>5d}  median dur {g['dur'].median():>6.0f}  "
              f"event rate {g['event_sold'].mean():.3f}")

    # ------------------------------------------------------------------ 2
    rule("2  Export sort order and the 5,000-row cap")
    print("  Status sequence through the file (run-length encoded):")
    s = raw["Status"].to_numpy()
    runs, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            runs.append((s[start], start, i - 1, i - start))
            start = i
    for label, a, b, n in runs:
        print(f"    rows {a:>5d}-{b:<5d}  {label:<20} n={n}")
    print(f"\n  distinct runs: {len(runs)}  -> the export is sorted by Status"
          if len(runs) <= 10 else f"\n  {len(runs)} runs — not a clean status sort")
    print("  The search returned '5000+' and the file holds exactly 5000 rows, so the")
    print("  cap truncated whichever status sorts LAST. That is truncation on the")
    print("  outcome variable, not a random subsample.")
    last_label, _, _, last_n = runs[-1]
    print(f"  last status block: {last_label} with n={last_n} — the truncated one")

    # ------------------------------------------------------------------ 3
    rule("3  rel_price_premium tails")
    feats = build_features(ingest_mls(market="miami").frame, config).frame
    rel = feats["rel_price_premium"].dropna()
    print(f"  n={len(rel)}  mean={rel.mean():+.4f}  sd={rel.std():.4f}")
    for q in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
        print(f"    q{q:<6} {rel.quantile(q):+10.4f}")
    print(f"  IQR = {rel.quantile(0.75) - rel.quantile(0.25):.4f}")
    print(f"  min = {rel.min():+.4f}   max = {rel.max():+.4f}")
    extreme = feats.loc[feats["rel_price_premium"] > 5]
    print(f"\n  rows with premium > +500%: {len(extreme)}")
    if len(extreme):
        cols = ["mls_number", "original_list_price", "living_area_sqft",
                "list_ppsf", "cell_median_ppsf", "rel_price_premium", "submarket"]
        print(extreme[cols].head(12).to_string(index=False))
    tiny = feats.loc[feats["living_area_sqft"] < 400]
    print(f"\n  rows with living_area_sqft < 400: {len(tiny)}")
    if len(tiny):
        print(tiny[["mls_number", "original_list_price", "living_area_sqft",
                    "list_ppsf"]].head(10).to_string(index=False))
    print(f"\n  list_ppsf: min {feats['list_ppsf'].min():,.0f}  "
          f"max {feats['list_ppsf'].max():,.0f}  "
          f"median {feats['list_ppsf'].median():,.0f}")

    # ------------------------------------------------------------------ 4
    rule("4  Can the missing list_date be reconstructed?  list_date = terminal - DOM")
    check = norm[norm["list_date"].notna() & norm["terminal"].notna()
                 & norm["days_on_market"].notna()].copy()
    check["implied"] = check["terminal"] - pd.to_timedelta(
        check["days_on_market"].astype(float), unit="D")
    check["err_days"] = (check["implied"] - check["list_date"]).dt.days
    print(f"  rows where all three are present : {len(check)}")
    print(f"  exact match (0 days)             : {int((check['err_days'] == 0).sum())} "
          f"({(check['err_days'] == 0).mean():.2%})")
    print(f"  within +/-1 day                  : {int((check['err_days'].abs() <= 1).sum())} "
          f"({(check['err_days'].abs() <= 1).mean():.2%})")
    print(f"  within +/-3 days                 : {int((check['err_days'].abs() <= 3).sum())} "
          f"({(check['err_days'].abs() <= 3).mean():.2%})")
    print(f"  err quantiles: {check['err_days'].quantile([0.01,0.25,0.5,0.75,0.99]).to_dict()}")

    print("\n  by status (where the identity can be checked):")
    for status, g in check.groupby("status"):
        print(f"    {status:<10} n={len(g):>5d}  exact={(g['err_days'] == 0).mean():6.2%}  "
              f"median err={g['err_days'].median():+.0f}d")

    print("\n  Rows that NEED reconstruction (no list_date):")
    need = norm[norm["list_date"].isna()]
    print(f"    total {len(need)}")
    for status, g in need.groupby("status"):
        have_terminal = int(g["terminal"].notna().sum())
        have_dom = int(g["days_on_market"].notna().sum())
        both = int((g["terminal"].notna() & g["days_on_market"].notna()).sum())
        print(f"    {status:<10} n={len(g):>4d}  terminal={have_terminal:>4d}  "
              f"dom={have_dom:>4d}  both={both:>4d}  <- reconstructable: {both}")

    # Also check CDOM, which spans relists and may be the wrong clock.
    print("\n  DOM vs CDOM on rows needing reconstruction:")
    if "cumulative_days_on_market" in need.columns:
        d = pd.to_numeric(need["days_on_market"], errors="coerce")
        c = pd.to_numeric(need["cumulative_days_on_market"], errors="coerce")
        print(f"    DOM  median {d.median():.0f}   CDOM median {c.median():.0f}   "
              f"equal in {int((d == c).sum())} of {len(need)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
