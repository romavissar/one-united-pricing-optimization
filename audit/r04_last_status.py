"""R04 — What is `Last Status`, and is it authoritative for outcome coding?

`Last Status` is 87.7% filled on the quarterly export and is mapped nowhere: it
falls into `_unmapped` with no documented semantics. Before it can be ignored
on purpose it has to be established what it holds.

The question that matters: does `Status` or `Last Status` describe the outcome
the survival model should code? If they disagree on rows that matter — a
listing whose `Status` says Cancelled but whose `Last Status` says Closed — then
the event coding is reading the wrong column.

Cross-tabs `Status` x `Last Status` and checks each combination against the hard
evidence of what actually happened: whether a Closing Date exists.

Run from backend/:  .venv/bin/python ../audit/r04_last_status.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    frames = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob("data/raw/mls/*.csv"))]
    df = pd.concat(frames, ignore_index=True)

    status = df["Status"].astype("string")
    last = df["Last Status"].astype("string")
    closed = df["Closing Date"].notna()
    pending = df["Pending Date"].notna()

    print("=" * 74)
    print("1  Fill and vocabulary")
    print("=" * 74)
    print(f"  rows              : {len(df)}")
    print(f"  Last Status filled: {last.notna().mean():.1%}")
    print(f"  Status values     : {sorted(status.dropna().unique())}")
    print(f"  Last Status values: {sorted(last.dropna().unique())}")

    print()
    print("=" * 74)
    print("2  Status x Last Status")
    print("=" * 74)
    tab = pd.crosstab(status, last.fillna("<blank>"))
    print(tab.to_string())

    print()
    print("=" * 74)
    print("3  Do they ever disagree about a SALE?")
    print("=" * 74)
    sold_by_status = status.eq("Closed")
    sold_by_last = last.eq("Closed")
    print(f"  Status == Closed                : {int(sold_by_status.sum())}")
    print(f"  Last Status == Closed           : {int(sold_by_last.sum())}")
    print(f"  Closing Date present            : {int(closed.sum())}")
    print()
    print(f"  Status=Closed but no Closing Date      : "
          f"{int((sold_by_status & ~closed).sum())}")
    print(f"  Closing Date but Status != Closed      : "
          f"{int((closed & ~sold_by_status).sum())}")
    print(f"  LastStatus=Closed but Status != Closed : "
          f"{int((sold_by_last & ~sold_by_status).sum())}")
    if int((sold_by_last & ~sold_by_status).sum()):
        rows = df.loc[sold_by_last & ~sold_by_status,
                      ["Status", "Last Status", "Closing Date", "Sale Price", "DOM"]]
        print("\n  sample of the disagreement:")
        print(rows.head(10).to_string(index=False))

    print()
    print("=" * 74)
    print("4  Which column agrees with the hard evidence (a Closing Date)?")
    print("=" * 74)
    for label, series in (("Status", status), ("Last Status", last)):
        says_sold = series.eq("Closed")
        tp = int((says_sold & closed).sum())
        fp = int((says_sold & ~closed).sum())
        fn = int((~says_sold & closed).sum())
        print(f"  {label:<12} agrees with Closing Date on "
              f"{tp}/{int(closed.sum())} sales; claims {fp} sales with no closing "
              f"date; misses {fn}")

    print()
    print("=" * 74)
    print("5  Is Last Status just the pre-terminal state? (the live statuses)")
    print("=" * 74)
    live = status.isin(["Active", "Active With Contract", "Pending", "Coming Soon"])
    print(f"  live rows: {int(live.sum())}")
    print(f"  their Last Status distribution:")
    for value, n in last[live].fillna("<blank>").value_counts().head(8).items():
        print(f"    {str(value):<24} {n}")
    print()
    print(f"  terminated rows: {int((~live).sum())}")
    print(f"  their Last Status distribution:")
    for value, n in last[~live].fillna("<blank>").value_counts().head(8).items():
        print(f"    {str(value):<24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
