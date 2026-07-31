"""R02 — Which identity recovers `list_date` on the live statuses?

The first attempt anchored on Status Change Date and 66% of the derived dates
landed outside their file's own quarter, which means the anchor is wrong. This
tests every candidate anchor against the one hard constraint available: each
quarterly file is a query for listings whose list date falls inside that
quarter, so a derived date outside it is arithmetic that did not work.

Candidates, per status group:
    list_date = <anchor> - DOM      for anchor in {status change, off market,
                                    pending, close, file export date}
    list_date = <anchor> - CDOM     (CDOM spans relists, so it should be worse)

Also validates each candidate on the three statuses where List Date IS
reported, which is the only place the answer is known.

Run from backend/:  .venv/bin/python ../audit/r02_list_date_identity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/mls")
_Q = re.compile(r"(?i)^~?\s*Q([1-4])[-_ ]?(\d{4})")


def quarter(name: str) -> str | None:
    m = _Q.match(name)
    return f"{m.group(2)}Q{m.group(1)}" if m else None


def main() -> int:
    parts = []
    for p in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(p, dtype=str)
        d["_file"] = p.name
        d["_quarter"] = quarter(p.name)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # Use the production coercion: List Date is US m/d/Y, every other date
    # column in this export is ISO, some with a time component. Hardcoding one
    # format silently NaTs the others.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from src.data.normalize import coerce_date

    def d8(s):
        return pd.to_datetime(df[s].map(coerce_date), errors="coerce")
    ld = d8("List Date")
    anchors = {
        "status_change": d8("Status Change Date"),
        "off_market": d8("Off Market Date"),
        "pending": d8("Pending Date"),
        "closing": d8("Closing Date"),
        "expiration": d8("Expiration Date"),
    }
    dom = pd.to_numeric(df["DOM"], errors="coerce")
    cdom = pd.to_numeric(df["CDOM (Days on Market)"], errors="coerce")

    print("=" * 78)
    print("A  Validate each anchor where List Date IS reported")
    print("=" * 78)
    known = ld.notna()
    for name, anchor in anchors.items():
        for label, days in (("DOM", dom), ("CDOM", cdom)):
            ok = known & anchor.notna() & days.notna()
            if ok.sum() < 100:
                continue
            err = ((anchor - pd.to_timedelta(days, unit="D")) - ld).dt.days[ok]
            print(f"  {name:<15} - {label:<5} n={int(ok.sum()):>6}  "
                  f"exact {float((err == 0).mean()):6.1%}  "
                  f"|err|<=2 {float((err.abs() <= 2).mean()):6.1%}  "
                  f"median {err.median():+.0f}d")

    print()
    print("=" * 78)
    print("B  Snapshot date per file (max date seen anywhere in the file)")
    print("=" * 78)
    all_dates = pd.concat(list(anchors.values()) + [ld], axis=1)
    df["_snapshot"] = all_dates.max(axis=1)
    snap = df.groupby("_file")["_snapshot"].max()
    for f, s in snap.items():
        print(f"  {f:<16} {s.date() if pd.notna(s) else '-'}")
    global_snap = df["_snapshot"].max()
    print(f"  GLOBAL max: {global_snap.date()}")

    print()
    print("=" * 78)
    print("C  On the rows that NEED recovery: which candidate lands in-quarter?")
    print("=" * 78)
    need = ld.isna()
    print(f"  rows needing recovery: {int(need.sum())}")
    qp = pd.PeriodIndex(df["_quarter"].astype("string"), freq="Q")

    file_snap = df["_file"].map(snap)
    candidates = {
        f"{n} - DOM": anchors[n] - pd.to_timedelta(dom, unit="D") for n in anchors
    }
    candidates.update(
        {f"{n} - CDOM": anchors[n] - pd.to_timedelta(cdom, unit="D") for n in anchors}
    )
    candidates["file snapshot - DOM"] = file_snap - pd.to_timedelta(dom, unit="D")
    candidates["file snapshot - CDOM"] = file_snap - pd.to_timedelta(cdom, unit="D")
    candidates["global snapshot - DOM"] = global_snap - pd.to_timedelta(dom, unit="D")

    rows = []
    for name, cand in candidates.items():
        avail = need & cand.notna()
        inq = avail & (cand.dt.to_period("Q") == qp)
        rows.append((name, int(avail.sum()), int(inq.sum()),
                     float(inq.sum() / max(int(need.sum()), 1))))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'candidate':<26} {'available':>10} {'in-quarter':>11} {'coverage':>9}")
    for name, a, i, cov in rows:
        print(f"  {name:<26} {a:>10} {i:>11} {cov:>8.1%}")

    print()
    print("=" * 78)
    print("D  Per status: best candidate")
    print("=" * 78)
    for status, idx in df[need].groupby("Status").groups.items():
        sub = df.index.isin(idx)
        best = []
        for name, cand in candidates.items():
            inq = sub & cand.notna().to_numpy() & (
                cand.dt.to_period("Q") == qp).to_numpy()
            best.append((name, int(inq.sum())))
        best.sort(key=lambda r: -r[1])
        n = int(sub.sum())
        top = ", ".join(f"{nm} {c}/{n}" for nm, c in best[:3])
        print(f"  {status:<24} n={n:>5}   {top}")

    print()
    print("=" * 78)
    print("E  Why status_change - DOM fails: what IS Status Change Date?")
    print("=" * 78)
    for status in ("Active", "Pending", "Withdrawn", "Closed", "Cancelled"):
        m = df["Status"] == status
        sc = anchors["status_change"][m]
        print(f"  {status:<14} status_change {sc.min().date() if sc.notna().any() else '-'}"
              f" .. {sc.max().date() if sc.notna().any() else '-'}"
              f"   DOM median {dom[m].median():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
