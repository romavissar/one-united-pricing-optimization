"""R01 — Profile the 15 quarterly exports before touching the ingest.

Verifies every claim in the remediation brief against the files themselves,
because a claim about data is a hypothesis until it is measured:

  * 48,206 rows, uniform 48-column schema, no duplicate ML#
  * every list date inside its own quarter
  * List Date null for exactly the six live statuses, 0% elsewhere
  * DOM and Status Change Date 100% filled on those rows
  * Unit View fill rate and its level vocabulary
  * Association Fee vs Maintenance Charge/Month overlap and ratio
  * REO / Short Sale distributions
  * Q1-2023 left-truncation at the window edge

Run from backend/:  .venv/bin/python ../audit/r01_profile_new_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/mls")
OLD = Path("data/raw/mls_old/PRICING_MODEL_export.csv")


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def quarter_of(name: str) -> pd.Period | None:
    stem = name.lstrip("~").replace(".csv", "")
    try:
        q, y = stem.split("-")
        return pd.Period(f"{y}Q{q[1]}", freq="Q")
    except Exception:
        return None


def main() -> int:
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() == ".csv")
    frames = {}
    for p in files:
        frames[p.name] = pd.read_csv(p, dtype=str, keep_default_na=True)

    rule("1  Files, rows, schema")
    total = 0
    headers = set()
    for name, df in frames.items():
        total += len(df)
        headers.add(tuple(df.columns))
        print(f"  {name:<16} {len(df):>6} rows  {len(df.columns)} cols  "
              f"quarter={quarter_of(name)}")
    print(f"\n  total rows      : {total}")
    print(f"  distinct headers: {len(headers)}  -> "
          f"{'UNIFORM' if len(headers) == 1 else 'MISMATCH'}")

    everything = pd.concat(frames.values(), ignore_index=True)
    print(f"  concatenated    : {len(everything)}")
    ml = everything["ML#"]
    print(f"  distinct ML#    : {ml.nunique()}   duplicates: "
          f"{int(ml.duplicated().sum())}")
    if ml.duplicated().any():
        dupes = ml[ml.duplicated(keep=False)].value_counts().head(10)
        print(f"    top duplicated: {dupes.to_dict()}")

    rule("2  List Date: coverage inside its own quarter, and who is missing it")
    for name, df in frames.items():
        q = quarter_of(name)
        ld = pd.to_datetime(df["List Date"], format="%m/%d/%Y", errors="coerce")
        have = ld.notna()
        if q is None:
            continue
        inside = ld.dt.to_period("Q") == q
        outside = int((have & ~inside).sum())
        print(f"  {name:<16} listdate {int(have.sum()):>6}/{len(df):<6} "
              f"({have.mean():6.1%})  outside its quarter: {outside}")

    rule("3  List Date null by status (the blocking issue)")
    ld_all = pd.to_datetime(everything["List Date"], format="%m/%d/%Y", errors="coerce")
    tab = pd.crosstab(everything["Status"], ld_all.isna())
    tab.columns = ["has_list_date", "missing_list_date"]
    tab["missing_pct"] = tab["missing_list_date"] / tab.sum(axis=1)
    print(tab.sort_values("missing_pct", ascending=False).to_string())
    missing = ld_all.isna()
    print(f"\n  total missing: {int(missing.sum())}")
    print(f"  statuses 100% missing : "
          f"{sorted(tab.index[tab['missing_pct'] == 1.0].tolist())}")
    print(f"  statuses 0% missing   : "
          f"{sorted(tab.index[tab['missing_pct'] == 0.0].tolist())}")

    print("\n  recovery inputs on the missing rows:")
    sub = everything[missing]
    for col in ("DOM", "CDOM (Days on Market)", "Status Change Date",
                "Off Market Date", "Pending Date", "Closing Date"):
        filled = sub[col].notna().mean() if col in sub.columns else float("nan")
        print(f"    {col:<26} {filled:6.1%} filled")

    rule("4  Status distribution and event/censor split")
    vc = everything["Status"].value_counts(dropna=False)
    for s, n in vc.items():
        print(f"  {str(s):<26} {n:>6}  ({n/len(everything):5.1%})")

    rule("5  Unit View — the amenity the old export lacked")
    uv = everything["Unit View"]
    print(f"  fill rate: {uv.notna().mean():.1%}  ({int(uv.notna().sum())} rows)")
    print(f"  distinct raw values: {uv.nunique()}")
    print("\n  top 25 raw values:")
    for v, n in uv.value_counts().head(25).items():
        print(f"    {n:>6}  {v}")

    rule("6  Association Fee vs Maintenance Charge/Month")
    def money(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    af, mc = money(everything["Association Fee"]), money(everything["Maintenance Charge/Month"])
    both = af.notna() & mc.notna() & (mc > 0)
    print(f"  Association Fee filled        : {af.notna().mean():.1%}")
    print(f"  Maintenance Charge/Month      : {mc.notna().mean():.1%}")
    print(f"  overlap (both, mc>0)          : {int(both.sum())} rows")
    if both.any():
        ratio = (af[both] / mc[both])
        print(f"  ratio AF/MC  median {ratio.median():.4f}  "
              f"mean {ratio.mean():.4f}  p10 {ratio.quantile(.1):.3f}  "
              f"p90 {ratio.quantile(.9):.3f}")
        print(f"  median AF ${af[both].median():,.0f}   "
              f"median MC ${mc[both].median():,.0f}")
        print(f"  share with ratio in [0.95, 1.05]: "
              f"{float(ratio.between(0.95, 1.05).mean()):.1%}")
        print(f"  share with ratio near 12 (annual AF): "
              f"{float(ratio.between(11, 13).mean()):.1%}")

    rule("7  REO / Short Sale — the distressed filters")
    for col in ("REO", "Short Sale"):
        vc2 = everything[col].value_counts(dropna=False)
        print(f"  {col}: {vc2.to_dict()}")

    rule("8  New fields worth bringing into the hedonic")
    for col in ("Restrictions", "Terms Considered", "Parking Description",
                "Occupancy Information", "Minimum # of Days for Lease",
                "Furnished Info (List)", "Special Assessment YN", "Amenities",
                "Waterfront Description", "Type of Association", "Property Type",
                "Unit Number", "Tax Amount", "Special Information"):
        s = everything[col]
        print(f"  {col:<30} fill {s.notna().mean():6.1%}  distinct {s.nunique():>5}"
              f"   e.g. {str(s.dropna().iloc[0])[:44] if s.notna().any() else '-'}")

    rule("9  Q1-2023 left truncation at the window edge")
    q12023 = frames.get("Q1-2023.csv")
    if q12023 is not None:
        ld = pd.to_datetime(q12023["List Date"], format="%m/%d/%Y", errors="coerce")
        print(f"  Q1-2023 rows: {len(q12023)}   list dates "
              f"{ld.min()} .. {ld.max()}")
        print(f"  A listing begun in 2022 and still live into 2023 has no row here,")
        print(f"  because the quarter file is keyed on a list date inside the quarter.")
        for name in ("Q1-2023.csv", "Q2-2023.csv", "Q3-2023.csv", "Q4-2023.csv",
                     "Q1-2024.csv"):
            d = frames[name]
            st = d["Status"].value_counts(normalize=True)
            closed = st.get("Closed", 0.0)
            print(f"    {name:<14} n={len(d):>5}  Closed share {closed:5.1%}")

    rule("10  Area and price sanity, incl. SqFt = 0")
    sq = pd.to_numeric(everything["SqFt Liv Area"].astype(str)
                       .str.replace(r"[,]", "", regex=True), errors="coerce")
    olp = money(everything["Original List Price"])
    print(f"  SqFt Liv Area: filled {sq.notna().mean():.1%}  "
          f"zero {int((sq == 0).sum())}  negative {int((sq < 0).sum())}")
    print(f"    min {sq.min()}  p1 {sq.quantile(.01)}  median {sq.median()}  "
          f"p99 {sq.quantile(.99)}  max {sq.max()}")
    print(f"  Original List Price: filled {olp.notna().mean():.1%}  "
          f"zero {int((olp == 0).sum())}  min {olp.min()}  max {olp.max()}")
    ppsf = olp / sq.replace(0, np.nan)
    print(f"  implied $/sqft: min {ppsf.min():,.0f}  p01 {ppsf.quantile(.01):,.0f}  "
          f"median {ppsf.median():,.0f}  p999 {ppsf.quantile(.999):,.0f}  "
          f"max {ppsf.max():,.0f}")
    print(f"    rows > $20,000/sqft: {int((ppsf > 20000).sum())}")
    print(f"    rows < $100/sqft   : {int((ppsf < 100).sum())}")

    rule("11  Floor vs total floors")
    uf = pd.to_numeric(everything["Unit Floor Location"], errors="coerce")
    tf = pd.to_numeric(everything["Total Floors In Building"], errors="coerce")
    bad = uf.notna() & tf.notna() & (uf > tf)
    print(f"  Unit Floor Location filled : {uf.notna().mean():.1%}")
    print(f"  Total Floors filled        : {tf.notna().mean():.1%}")
    print(f"  floor > total_floors       : {int(bad.sum())} rows "
          f"({bad.mean():.2%})")
    if bad.any():
        print(f"    of those, total_floors == 0 or 1: "
              f"{int((bad & tf.le(1)).sum())}")
        print(f"    excess distribution: "
              f"{(uf[bad] - tf[bad]).describe()[['min','50%','max']].to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
