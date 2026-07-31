"""R03 — Decompose the change in beta_price: new sample vs respecification.

Item 1 of the remediation brief. Fits all four combinations of

    data     in {old 5,000-row export, new 15-quarter re-pull}
    variable in {cell-median premium (old), hedonic residual (new)}

so the movement in `beta_price` can be attributed rather than asserted. Without
this the two changes are confounded and any improvement could be claimed for
either.

Both variables are put on the same scale before fitting — the residual is
exponentiated to a ratio-minus-one, matching the cell-median convention — so
the four coefficients are directly comparable and the decomposition is not an
artefact of one arm being in log points.

Run from backend/:  .venv/bin/python ../audit/r03_beta_decomposition.py
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
from src.demand.survival import CoxDemandModel  # noqa: E402

NEW_DIR = Path("data/raw/mls")
OLD_FILE = Path("data/raw/mls_old/PRICING_MODEL_export.csv")


def fit(frame: pd.DataFrame, variable: str, label: str) -> dict:
    """Fit the Cox with `variable` substituted in as the price regressor."""
    work = frame.copy()
    if variable not in work.columns:
        return {"label": label, "error": f"{variable} absent"}
    work["rel_price_premium"] = pd.to_numeric(work[variable], errors="coerce")
    usable = work["rel_price_premium"].notna()
    if int(usable.sum()) < 200:
        return {"label": label, "error": f"only {int(usable.sum())} usable rows"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = CoxDemandModel(covariates=DEMAND_COVARIATES).fit(work)
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        return {"label": label, "error": f"{type(exc).__name__}: {exc}"}
    beta = result.beta_price
    return {
        "label": label,
        "beta": beta.value,
        "se": beta.std_error,
        "lo": beta.ci_low,
        "hi": beta.ci_high,
        "n": result.n_observations,
        "events": result.n_events,
        "sd_x": float(work.loc[result.design.index, "rel_price_premium"].std()),
    }


def show(row: dict) -> None:
    if "error" in row:
        print(f"  {row['label']:<34} FAILED: {row['error']}")
        return
    print(
        f"  {row['label']:<34} {row['beta']:+.4f}  se {row['se']:.4f}  "
        f"CI [{row['lo']:+.4f}, {row['hi']:+.4f}]  n={row['n']:>6} "
        f"events={row['events']:>6}  sd(x)={row['sd_x']:.3f}"
    )


def main() -> int:
    config = load_market_config("miami")

    print("Building feature frames …")
    new = build_features(ingest_mls(raw_dir=NEW_DIR, market="miami").frame, config).frame
    old = build_features(
        ingest_mls(paths=[OLD_FILE], market="miami").frame, config
    ).frame
    print(f"  new: {len(new)} rows    old: {len(old)} rows")

    cells = {
        ("old data", "old variable"): (old, "cell_median_premium"),
        ("old data", "new variable"): (old, "hedonic_price_premium_ratio"),
        ("new data", "old variable"): (new, "cell_median_premium"),
        ("new data", "new variable"): (new, "hedonic_price_premium_ratio"),
    }

    print(f"\n{'=' * 78}\n2x2: beta_price by data source and variable definition\n{'=' * 78}")
    results: dict[tuple[str, str], dict] = {}
    for key, (frame, variable) in cells.items():
        row = fit(frame, variable, f"{key[0]} x {key[1]}")
        results[key] = row
        show(row)

    print(f"\n{'=' * 78}\nDecomposition\n{'=' * 78}")
    def b(k):
        r = results[k]
        return r.get("beta")

    oo, on = b(("old data", "old variable")), b(("old data", "new variable"))
    no, nn = b(("new data", "old variable")), b(("new data", "new variable"))
    if None in (oo, on, no, nn):
        print("  one or more cells failed; decomposition not available")
        return 1

    print(f"  starting point   (old data, old variable) : {oo:+.4f}")
    print(f"  ending point     (new data, new variable) : {nn:+.4f}")
    print(f"  total change                              : {nn - oo:+.4f}")
    print()
    print("  Path A — respecify first, then re-sample:")
    print(f"    respecification, holding data fixed     : {on - oo:+.4f}")
    print(f"    new sample, holding variable fixed      : {nn - on:+.4f}")
    print("  Path B — re-sample first, then respecify:")
    print(f"    new sample, holding variable fixed      : {no - oo:+.4f}")
    print(f"    respecification, holding data fixed     : {nn - no:+.4f}")
    print()
    print("  These decompositions are path-dependent — the two orders assign the")
    print("  interaction differently — so both are reported rather than one being")
    print("  presented as the answer. The interaction itself is:")
    print(f"    {(nn - no) - (on - oo):+.4f}")
    print()
    avg_resp = 0.5 * ((on - oo) + (nn - no))
    avg_samp = 0.5 * ((no - oo) + (nn - on))
    print(f"  Shapley-style average attribution:")
    print(f"    respecification : {avg_resp:+.4f}  ({avg_resp / (nn - oo):.0%} of the total)"
          if abs(nn - oo) > 1e-9 else f"    respecification : {avg_resp:+.4f}")
    print(f"    new sample      : {avg_samp:+.4f}  ({avg_samp / (nn - oo):.0%} of the total)"
          if abs(nn - oo) > 1e-9 else f"    new sample      : {avg_samp:+.4f}")

    print(f"\n{'=' * 78}\nWhat the respecification removed\n{'=' * 78}")
    for name, frame in (("old", old), ("new", new)):
        cm = pd.to_numeric(frame.get("cell_median_premium"), errors="coerce")
        hp = pd.to_numeric(frame.get("hedonic_price_premium_ratio"), errors="coerce")
        both = cm.notna() & hp.notna()
        if not both.any():
            continue
        print(f"  {name} data, n={int(both.sum())}")
        print(f"    sd(cell-median premium)  {cm[both].std():.4f}")
        print(f"    sd(hedonic residual)     {hp[both].std():.4f}")
        print(f"    corr between them        {cm[both].corr(hp[both]):.4f}")
        print(f"    variance ratio           {hp[both].var() / cm[both].var():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
