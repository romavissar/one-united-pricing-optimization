# Calibration gate — ready for human review

Phases 0–5 are built and verified end-to-end on **synthetic** data. Fitting on
the real MLS export is deliberately not automated (`PROJECT_BRIEF.md` §5 /
`AGENTS.md` §6). This checklist is the handoff.

---

## 1. Where to place the real files

```
backend/data/raw/mls/
```

Put the broker export(s) there (CSV/XLSX). The ingest CLI defaults to that
directory when no `--file` / `--dir` is given.

---

## 2. Ingest and inspect

```bash
cd backend && source .venv/bin/activate
python -m src.data.ingest_mls --inspect
```

Read the printed report: header mapping, status mix, submarket coverage, floor
parse rates, date formats. Fix aliases in `src/data/normalize.py` if required
fields are still missing — do not impute.

---

## 3. Fit (gated)

```bash
python -m src.demand.fit --dataset mls --calibration-gate --controls --building-fe
```

`--calibration-gate` is required. Without it the command refuses to touch
`data/raw/`. A successful fit **does not** set `is_calibrated_on_real_data`
to true.

Equivalent diagnostics-only path (no bundle save):

```bash
python -m src.demand.diagnostics --inspect --controls --building-fe --calibration-gate
```

---

## 4. Three numbers to read first

1. **Count of censored (non-`SOLD`) records** — right-censoring is what makes
   the Cox model the primary estimator. If almost everything is `SOLD`, the
   export is selected on outcome.
2. **IQR of `rel_price_premium`** — identification needs within-cell price
   variation. A near-zero IQR means there is nothing for `β_price` to be
   identified from.
3. **`β_price` with its 95% CI** — the number this system exists to estimate.

These appear in the fit / diagnostics report and in the saved bundle metadata
at `backend/data/processed/{market}/models/current/metadata.json`.

---

## 5. Decision rule

**If the 95% CI for `β_price` includes zero, elasticity is not identified —
stop and fix the data. Do not tune the model until the sign flips.**

A positive point estimate is an unconditional `FAILED` (`AGENTS.md` §3). The
honest output is that this export cannot answer the pricing question yet.

---

## 6. Clearing the illustrative banner

`is_calibrated_on_real_data` defaults to `false` in `backend/config/miami.yaml`:

```yaml
defaults:
  is_calibrated_on_real_data: false
```

Flipping it is a **single, explicit config change**, not a side effect of a
successful fit. After you have inspected the three numbers above and the CI
excludes zero:

1. Set `is_calibrated_on_real_data: true` in `config/miami.yaml`.
2. Re-run:

```bash
python -m src.demand.fit --dataset mls --calibration-gate --controls --building-fe --assert-calibrated
```

`--assert-calibrated` refuses unless the config flag is already true and the
dataset is `mls`. Synthetic fits can never set the flag.

---

## What has been verified on synthetic data

| Phase | Acceptance |
|---|---|
| 0–2 | Scaffolding, ingest, features |
| 1.5 | Synthetic generator with planted `β_price` |
| 3 | Cox / logistic / hedonic + loud failure on `β ≥ 0` |
| 4 | MILP optimizer, constraints, extrapolation guard |
| 5 | 10k Monte Carlo draws &lt; 60s; distribution widens with SE(β̂); tornado; LP shadow prices; optional LHS re-solve |
| 6 | API routes; provenance on optimize/simulate; mls fit requires `calibration_gate` |
| 7 | React UI; full synthetic loop; non-dismissible illustrative banner |

Fitting real MLS still requires this checklist — the API will 403
`dataset=mls` without `calibration_gate: true`.
