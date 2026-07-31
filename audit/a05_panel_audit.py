"""A05 — Survival panel, denominator, and sample-selection audit (brief §2.2, §2.3, §2.6, §6.1).

Everything here runs on the real Miami export. It answers, with counts:

  * how thin the (submarket x month) cells are, and what fell back or fell out
  * whether the focal listing is inside its own cell median
  * where the 13.2% of rows with no List Date go, and whether that loss is
    selective on the outcome
  * how durations were actually constructed, and how many non-positive spells
    were silently rerouted rather than reported
  * administrative censoring at the export edge
  * relists sharing a unit_key
  * whether the 5,000-row export cap is correlated with price or outcome

Run from backend/:  .venv/bin/python ../audit/a05_panel_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import load_market_config  # noqa: E402
from src.data.clean import clean_mls  # noqa: E402
from src.data.features import build_features  # noqa: E402
from src.data.ingest_mls import ingest_mls  # noqa: E402
from src.data.normalize import normalize_mls  # noqa: E402

RAW = Path("data/raw/mls/PRICING_MODEL_export.csv")


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> int:
    config = load_market_config("miami")
    ingested = ingest_mls(market="miami")
    frame = ingested.frame
    feats = build_features(frame, config).frame

    # ---------------------------------------------------------------- 2.2
    rule("2.2  The denominator: (submarket x month) cell sizes")
    keyed = feats[feats["submarket"].notna() & feats["list_month"].notna()]
    sizes = keyed.groupby(["submarket", "list_month"]).size()
    print(f"  rows with both submarket and list_month : {len(keyed)} of {len(feats)}")
    print(f"  distinct cells                          : {len(sizes)}")
    for threshold in (1, 2, 4, 8, 16, 32):
        n_cells = int((sizes >= threshold).sum())
        n_rows = int(sizes[sizes >= threshold].sum())
        print(f"    cells with n >= {threshold:<3d}: {n_cells:>4d} "
              f"({n_cells / len(sizes):5.1%} of cells)   "
              f"rows {n_rows:>5d} ({n_rows / len(keyed):5.1%})")
    print(f"  median cell size                        : {sizes.median():.0f}")
    print(f"  cells with n < 8                        : {int((sizes < 8).sum())}"
          f"  holding {int(sizes[sizes < 8].sum())} rows")

    print("\n  median_basis actually used:")
    for basis, n in feats["median_basis"].value_counts(dropna=False).items():
        print(f"    {basis:<22} {n:>5d}  ({n / len(feats):5.1%})")

    # Sampling variance of the cell median -> errors-in-variables attenuation.
    rule("2.2b Is the cell median signal or noise? (attenuation from a noisy denominator)")
    ppsf = feats["list_ppsf"]
    within_var = []
    for (_sub, _mon), grp in keyed.groupby(["submarket", "list_month"]):
        vals = grp["list_ppsf"].dropna()
        if len(vals) >= 4:
            # var of the sample median ~ 1 / (4 n f(m)^2); estimate f via IQR-based
            # normal-reference density at the median.
            iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
            if iqr > 0:
                sd_hat = iqr / 1.349
                var_median = (np.pi / 2.0) * sd_hat**2 / len(vals)
                within_var.append((len(vals), var_median, float(vals.median())))
    if within_var:
        n_arr = np.array([w[0] for w in within_var])
        varm = np.array([w[1] for w in within_var])
        med = np.array([w[2] for w in within_var])
        # premium = ppsf/median - 1 ; d(premium)/d(median) ~ -ppsf/median^2 ~ -1/median
        prem_noise_var = float(np.mean(varm / med**2))
        prem_var = float(feats["rel_price_premium"].var())
        print(f"  mean sampling var of cell median (in premium units): {prem_noise_var:.6f}")
        print(f"  observed var of rel_price_premium                  : {prem_var:.6f}")
        if prem_var > 0:
            atten = 1.0 - prem_noise_var / prem_var
            print(f"  implied attenuation factor from denominator noise  : {atten:.4f}")
            print(f"    -> a true beta of -1.0 would be measured as {-atten:.4f}")

    rule("2.2c Is the focal listing inside its own cell median? (leave-one-out)")
    # Recompute rel_price_premium leaving each row out of its own cell median.
    loo_prem = np.full(len(feats), np.nan)
    idx_pos = {ix: p for p, ix in enumerate(feats.index)}
    for (_sub, _mon), grp in keyed.groupby(["submarket", "list_month"]):
        vals = grp["list_ppsf"].dropna()
        if len(vals) < 8:
            continue
        arr = vals.to_numpy()
        for ix, v in vals.items():
            others = arr[arr != v] if (arr == v).sum() == 1 else np.delete(
                arr, np.where(arr == v)[0][0]
            )
            if len(others):
                loo_prem[idx_pos[ix]] = v / np.median(others) - 1.0
    incl = feats["rel_price_premium"].to_numpy()
    both = np.isfinite(loo_prem) & np.isfinite(incl)
    print(f"  rows compared            : {int(both.sum())}")
    print(f"  mean(inclusive - LOO)    : {np.mean(incl[both] - loo_prem[both]):+.6f}")
    print(f"  sd(inclusive - LOO)      : {np.std(incl[both] - loo_prem[both]):.6f}")
    print(f"  corr                     : {np.corrcoef(incl[both], loo_prem[both])[0,1]:.6f}")
    print(f"  sd(inclusive) {np.std(incl[both]):.6f}   sd(LOO) {np.std(loo_prem[both]):.6f}")

    # ---------------------------------------------------------------- 2.3
    rule("2.3  The 13.2% with no List Date — where do they go, and is the loss selective?")
    missing_ld = frame["list_date"].isna()
    print(f"  rows with no list_date : {int(missing_ld.sum())} "
          f"({missing_ld.mean():.2%})")
    print("\n  status distribution, by whether list_date is present:")
    ct = pd.crosstab(frame["status"], missing_ld, normalize="columns")
    counts = pd.crosstab(frame["status"], missing_ld)
    for status in counts.index:
        have = counts.loc[status, False] if False in counts.columns else 0
        lack = counts.loc[status, True] if True in counts.columns else 0
        print(f"    {status:<12} have={have:>5d} ({ct.loc[status, False]:6.2%})   "
              f"missing={lack:>5d} ({ct.loc[status, True]:6.2%})")
    chi2, p, _, _ = stats.chi2_contingency(counts.values)
    print(f"\n  chi-square test of independence (status vs list_date missing):")
    print(f"    chi2 = {chi2:.1f}   p = {p:.3e}"
          f"   -> {'NOT random' if p < 0.05 else 'consistent with random'}")

    ev_have = frame.loc[~missing_ld, "event_sold"].mean()
    ev_lack = frame.loc[missing_ld, "event_sold"].mean()
    print(f"\n  event rate WITH list_date    : {ev_have:.4f}")
    print(f"  event rate WITHOUT list_date : {ev_lack:.4f}")
    print(f"  difference                   : {ev_lack - ev_have:+.4f}")

    for col in ("original_list_price", "list_ppsf", "living_area_sqft"):
        a = frame.loc[~missing_ld, col].dropna()
        b = frame.loc[missing_ld, col].dropna()
        if len(a) > 5 and len(b) > 5:
            t, pv = stats.ttest_ind(a, b, equal_var=False)
            print(f"  {col:<22} have med={a.median():>10,.0f}  "
                  f"missing med={b.median():>10,.0f}   Welch p={pv:.3e}")

    print("\n  where those rows end up in the demand fit:")
    prem_null = feats["rel_price_premium"].isna()
    print(f"    rel_price_premium null overall : {int(prem_null.sum())} "
          f"({prem_null.mean():.2%})")
    print(f"    of which no list_date          : "
          f"{int((prem_null & feats['list_date'].isna()).sum())}")
    print(f"    of which no submarket          : "
          f"{int((prem_null & feats['submarket'].isna()).sum())}")

    rule("2.3b Duration construction: sources, non-positive spells, silent reroutes")
    print("  duration_source counts:")
    for src, n in frame["duration_source"].value_counts(dropna=False).items():
        print(f"    {str(src):<18} {n:>5d}")

    # Recompute the raw differences per status to find spells the rule silently skipped.
    raw = pd.read_csv(RAW, dtype=str)
    norm = normalize_mls(raw, config=config).frame
    ld = norm["list_date"]
    checks = {
        "pending_date - list_date": norm["pending_date"] - ld,
        "close_date - list_date": norm["close_date"] - ld,
        "off_market_date - list_date": norm["off_market_date"] - ld,
    }
    for name, delta in checks.items():
        d = delta.dt.days.dropna()
        neg = int((d < 0).sum())
        zero = int((d == 0).sum())
        print(f"  {name:<30} n={len(d):>5d}  negative={neg:>4d}  zero={zero:>4d}"
              f"  max={int(d.max()) if len(d) else 0}")

    # How many rows HAVE a list_date and a terminal date but still fell back to DOM?
    has_ld = norm["list_date"].notna()
    fell_to_dom = norm["duration_source"] == "days_on_market"
    print(f"\n  rows using days_on_market fallback            : {int(fell_to_dom.sum())}")
    print(f"    of which DO have a list_date                : "
          f"{int((fell_to_dom & has_ld).sum())}   <- silent reroute of a bad spell")
    print(f"  rows with NO duration at all                  : "
          f"{int(norm['duration_days'].isna().sum())}")

    rule("2.3c Administrative censoring at the export edge")
    print(f"  list_date range      : {norm['list_date'].min()}  ..  {norm['list_date'].max()}")
    print(f"  off_market range     : {norm['off_market_date'].min()}  ..  "
          f"{norm['off_market_date'].max()}")
    print(f"  close_date range     : {norm['close_date'].min()}  ..  {norm['close_date'].max()}")
    print(f"  ACTIVE listings      : {int((norm['status'] == 'ACTIVE').sum())}")
    print(f"  max duration_days    : {norm['duration_days'].max()}")
    late = norm[norm["list_date"] > (norm["list_date"].max() - pd.Timedelta(days=180))]
    if len(late):
        print(f"  listed in final 180d : {len(late)}  event rate "
              f"{late['event_sold'].mean():.4f}  median duration "
              f"{late['duration_days'].median():.0f}")
        early = norm[norm["list_date"] <= (norm["list_date"].max() - pd.Timedelta(days=180))]
        print(f"  listed earlier       : {len(early)}  event rate "
              f"{early['event_sold'].mean():.4f}  median duration "
              f"{early['duration_days'].median():.0f}")

    rule("2.3d Relists: duplicate unit_key")
    uk = frame["unit_key"]
    dup = uk[uk.astype(str).str.len() > 0].value_counts()
    repeats = dup[dup > 1]
    print(f"  distinct unit_key        : {uk.nunique()}")
    print(f"  unit_keys appearing >1   : {len(repeats)}  covering "
          f"{int(repeats.sum())} rows")
    print(f"  largest repeat group     : {int(repeats.max()) if len(repeats) else 0}")
    print(f"  unit_keys starting 'na|' : "
          f"{int(uk.astype(str).str.startswith('na|').sum())}  "
          f"<- building_name was pd.NA, street fallback never fired")
    print(f"  empty unit_key           : {int((uk.astype(str) == '').sum())}")

    # ---------------------------------------------------------------- 2.6
    rule("2.6  The 5,000-row export cap: is the truncation selective?")
    raw_ordered = pd.read_csv(RAW, dtype=str)
    n = len(raw_ordered)
    first, last = raw_ordered.iloc[: n // 5], raw_ordered.iloc[-(n // 5):]
    print(f"  comparing first {len(first)} vs last {len(last)} rows as exported")
    print("\n  status mix:")
    fs = first["Status"].value_counts(normalize=True)
    ls = last["Status"].value_counts(normalize=True)
    for status in sorted(set(fs.index) | set(ls.index)):
        print(f"    {status:<18} first={fs.get(status, 0):6.2%}   last={ls.get(status, 0):6.2%}")
    ct2 = pd.DataFrame({
        "first": first["Status"].value_counts(),
        "last": last["Status"].value_counts(),
    }).fillna(0)
    chi2b, pb, _, _ = stats.chi2_contingency(ct2.values)
    print(f"    chi2 = {chi2b:.1f}  p = {pb:.3e}"
          f"  -> {'DIFFERENT' if pb < 0.05 else 'consistent'}")

    fp = pd.to_numeric(first["Original List Price"].str.replace(r"[$,]", "", regex=True),
                       errors="coerce").dropna()
    lp = pd.to_numeric(last["Original List Price"].str.replace(r"[$,]", "", regex=True),
                       errors="coerce").dropna()
    t2, p2 = stats.ttest_ind(fp, lp, equal_var=False)
    ks, pks = stats.ks_2samp(fp, lp)
    print(f"\n  original list price: first median ${fp.median():,.0f}   "
          f"last median ${lp.median():,.0f}")
    print(f"    Welch p={p2:.3e}   KS p={pks:.3e}")

    fd = pd.to_datetime(first["List Date"], format="%m/%d/%Y", errors="coerce").dropna()
    ldt = pd.to_datetime(last["List Date"], format="%m/%d/%Y", errors="coerce").dropna()
    print(f"\n  list date: first median {fd.median()}   last median {ldt.median()}")
    print(f"  ML# first {first['ML#'].iloc[0]} .. last {last['ML#'].iloc[-1]}")
    mlf = pd.to_numeric(first["ML#"], errors="coerce").dropna()
    mll = pd.to_numeric(last["ML#"], errors="coerce").dropna()
    if len(mlf) and len(mll):
        print(f"  ML# numeric: first median {mlf.median():,.0f}  last median {mll.median():,.0f}")
        print("  -> export appears sorted by "
              f"{'ML#' if abs(mlf.median()-mll.median()) > mlf.std() else 'something else'}")

    # ---------------------------------------------------------------- 6.1
    rule("6.1  Row-count waterfall and silent losses")
    print(f"  raw rows read                      : {len(raw_ordered)}")
    print(f"  after normalize                    : {len(norm)}")
    cleaned, creport = clean_mls(norm, config)
    print(f"  after clean                        : {len(cleaned)}")
    for reason, k in creport.dropped_by_reason.items():
        print(f"    dropped {k:>4d}  {reason}")
    print(f"  with a usable rel_price_premium    : "
          f"{int(feats['rel_price_premium'].notna().sum())}")
    covars = ["rel_price_premium", "log_floor", "living_area_sqft", "beds",
              "hoa_per_sqft", "is_new_construction", "submarket", "season",
              "inventory_competition"]
    present = [c for c in covars if c in feats.columns]
    complete = feats[present].notna().all(axis=1)
    print(f"  complete on all Phase-3 covariates : {int(complete.sum())}"
          f"  ({complete.mean():.1%})")
    print("\n  null count per covariate:")
    for c in present:
        print(f"    {c:<24} {int(feats[c].isna().sum()):>5d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
