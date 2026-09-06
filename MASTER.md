# MASTER.md — Complete system explanation

This is the long-form map of the **Pricing & Release Optimizer**: what it is for,
how every layer works, how the pieces connect, where numbers come from, and where
the system can lie to you. Read `PROJECT_BRIEF.md` and `AGENTS.md` first for the
short version; this file is the deep cut.

**Last aligned with the codebase after Phases 0–7** (synthetic end-to-end path
complete; real MLS fit still behind the calibration gate).

---

## Table of contents

1. [Prime directive](#1-prime-directive)
2. [What the product outputs](#2-what-the-product-outputs)
3. [Architecture overview](#3-architecture-overview)
4. [Numerical conventions (read before touching numbers)](#4-numerical-conventions)
5. [Hard rules (violations are bugs)](#5-hard-rules-violations-are-bugs)
6. [Repository map](#6-repository-map)
7. [End-to-end data flow](#7-end-to-end-data-flow)
8. [Phase 0 — Scaffolding](#8-phase-0--scaffolding)
9. [Phase 1 — MLS ingest and normalization](#9-phase-1--mls-ingest-and-normalization)
10. [Phase 1.5 — Synthetic data](#10-phase-15--synthetic-data)
11. [Phase 2 — Feature engineering](#11-phase-2--feature-engineering)
12. [Phase 3 — Demand models](#12-phase-3--demand-models)
13. [Phase 4 — Optimizer (MILP)](#13-phase-4--optimizer-milp)
14. [Phase 5 — Monte Carlo and sensitivity](#14-phase-5--monte-carlo-and-sensitivity)
15. [Phase 6 — API](#15-phase-6--api)
16. [Phase 7 — Frontend](#16-phase-7--frontend)
17. [Config and markets](#17-config-and-markets)
18. [Provenance and the calibration gate](#18-provenance-and-the-calibration-gate)
19. [Inventory fields (developer upload)](#19-inventory-fields-developer-upload)
20. [Potential bugs and failure modes](#20-potential-bugs-and-failure-modes)
21. [Known modeling simplifications (do not “fix” silently)](#21-known-modeling-simplifications)
22. [Engineering debt and stub surfaces](#22-engineering-debt-and-stub-surfaces)
23. [Testing strategy](#23-testing-strategy)
24. [Phase acceptance criteria (brief)](#24-phase-acceptance-criteria-brief)
25. [How to run everything](#25-how-to-run-everything)
26. [Glossary](#26-glossary)

---

## 1. Prime directive

This project exists to estimate one number:

> **`β_price`** — own-price elasticity of demand, as the coefficient on
> `rel_price_premium` (or its hedonic-residual successor) in a survival /
> discrete-choice model of time-to-sale.

Every module must do one of three things:

1. **Help estimate** `β_price`,
2. **Consume** it (optimizer, simulation, UI), or
3. **Report honestly** about its uncertainty / non-identification.

Anything else is out of scope.

### The failure mode to design against

A dashboard that prints confident prices which are really just:

> “Charge the maximum on every unit’s price ladder.”

That happens when `β_price ≈ 0` (or positive): the objective
`price × area × P(sell)` becomes monotone in price, CBC pushes every binary to
the ceiling, and the output looks “optimal.” The system must make that failure
**loud** (diagnostics `FAILED`, ceiling-share caveats, uncalibrated banner), not
pretty.

---

## 2. What the product outputs

Given a **project inventory** (unit floor, size, view, type, cost basis,
completion date, submarket):

1. A **$/sqft price** for every releasable unit,
2. A **phase** in which to release it,
3. A **revenue distribution** under uncertainty (not only a point NPV),
4. **Sensitivity** (tornado, cash-flow duals),
5. Always with a **`provenance`** block stating whether demand was fitted on
   real MLS or synthetic data.

Subject to business constraints: cash-flow floors, max units/phase, construction
gates, monotone price paths for comparable units, type diversity, cost+margin vs
comps bounds.

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (React + Vite + Tailwind + Plotly + TanStack)                 │
│  Upload → Fit/Optimize/Simulate/Sensitivity → Heatmap, table, charts    │
│  Non-dismissible banner if is_calibrated_on_real_data === false         │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ REST + SSE (/api/*)
┌───────────────────────────────┴─────────────────────────────────────────┐
│  API (FastAPI)                                                          │
│  validate, fit, optimize, simulate, sensitivity, demand/curve            │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────────┐
│  SIMULATION                                                             │
│  scenarios (correlated shocks) → monte_carlo (fixed plan) → sensitivity │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ needs R[i,j,k] and a plan
┌───────────────────────────────┴─────────────────────────────────────────┐
│  OPTIMIZER (PuLP / CBC MILP)                                            │
│  y[i,j,k] ∈ {0,1}: unit i, phase j, price level k                       │
│  max Σ discounted expected revenue                                      │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ needs D(price, features) → P(sell)
┌───────────────────────────────┴─────────────────────────────────────────┐
│  DEMAND                                                                 │
│  Cox PH (primary) + logistic cross-check + hedonic surface (bounds)     │
│  diagnostics fail loudly if β_price ≥ 0                                 │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ needs rel_price_premium + covariates
┌───────────────────────────────┴─────────────────────────────────────────┐
│  FEATURES + (optional) hedonic residual premium                         │
└───────────────────────────────▲─────────────────────────────────────────┘
                                │ needs clean canonical MLS
┌───────────────────────────────┴─────────────────────────────────────────┐
│  DATA                                                                   │
│  ingest → normalize → clean → synth (for verification)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Locked stack** (do not replace): Python 3.11+, FastAPI, PuLP/CBC, lifelines,
statsmodels, sklearn, pandas/numpy/scipy, React 18, Vite, Tailwind, Plotly,
TanStack Table, Papa Parse, SheetJS. No Streamlit, Next.js, Docker, Postgres,
Redis, Electron.

---

## 4. Numerical conventions

| Quantity | Convention |
|---|---|
| Price (internal) | **$/sqft**, USD |
| Total price | Only at presentation: `price_ppsf × living_area_sqft` |
| Area | **sqft** (Miami); Bucharest seam uses sqm — never mix in one run |
| Discount rate | **Annual decimal** (`0.12` = 12%). Factor `1/(1+r)^(t_months/12)` |
| Elasticity regressor | `rel_price_premium` = ratio − 1 (`0.10` = 10% above cell median / reference) |
| Duration | **Days** internally |
| Probability | `[0, 1]` until display |
| HOA in models | Prefer `hoa_per_sqft` ($/sqft/month), derived from `hoa_monthly` |

**Most likely silent bug in the entire repo:** treating a total USD price as
`$/sqft`, or an annual HOA as monthly. Validation and HOA sanity bands exist
specifically for this.

---

## 5. Hard rules (violations are bugs)

From `AGENTS.md`:

1. **Never hardcode an estimated parameter** (`β_price`, floor premiums α/γ,
   view premiums, absorption rates). Config holds priors/bounds, not answers.
   Literals like `-1.6` belong only in tests, synthetic generators, or
   documented sanity comments.
2. **Never silently impute.** Missing → null + `*_source` marker
   (`reported` / `parsed` / `missing`). Wrong floors corrupt hedonics → bounds →
   optimizer.
3. **Diagnostics fail loudly.** `β(rel_price_premium) ≥ 0` → hard `FAILED`. Do
   not retune the spec until the sign flips.
4. **Every optimize/simulate response includes `provenance`**, including
   `is_calibrated_on_real_data`. Frontend banner is non-dismissible when false.
5. **Do not add dependencies** outside the locked stack.
6. **Calibration gate:** do not autonomously fit real MLS; see
   `CALIBRATION_READY.md`.
7. **Build order:** phases with tests passing before the next phase.

---

## 6. Repository map

### Top-level docs

| Path | Role |
|---|---|
| `PROJECT_BRIEF.md` | Product, stack, phased build, acceptance criteria |
| `AGENTS.md` | Hard rules, conventions, known simplifications |
| `MLS_SCHEMA.md` | Canonical MLS fields, aliases, coercion, cleaning |
| `CALIBRATION_READY.md` | Human checklist before real-data fit |
| `README.md` | Quick start, CLIs, known gaps |
| `MASTER.md` | This file |
| `audit/` | Offline audit scripts + `AUDIT_REPORT.md` / `REMEDIATION_REPORT.md` |
| `.env.example` | Secrets surface (e.g. OpenRouter); not model parameters |

### Backend (`backend/`)

| Path | Role |
|---|---|
| `main.py` | FastAPI app, CORS, mounts routes, `/api/health` |
| `requirements.txt` | Locked Python deps |
| `config/miami.yaml` | Submarkets, views, defaults, filters, calibration flag |
| `config/bucharest.yaml` | Schema stub for secondary market |
| `src/config.py` | Load/validate YAML; ZIP → submarket |
| `src/exceptions.py` | `SchemaError`, `IdentificationError`, `InfeasibleModelError` |
| `src/data/ingest_mls.py` | Read raw exports, orchestrate normalize/clean, `--inspect` |
| `src/data/normalize.py` | Header aliases, types, floor/unit parse, status, dates |
| `src/data/clean.py` | Filter distressed/rentals, outliers, dedupe |
| `src/data/features.py` | `rel_price_premium`, competition, HOA psf, season, etc. |
| `src/data/premium.py` | Hedonic-residual premium (generated regressor) |
| `src/data/synth.py` | Synthetic MLS with planted `true_beta_price` |
| `src/data/store.py` | **Stub** — “Implemented in a later phase” (SQLite/Parquet helpers not built) |
| `src/demand/base.py` | Design matrix, `Coefficient`, `PremiumSupport` |
| `src/demand/survival.py` | Cox PH primary demand model |
| `src/demand/logistic.py` | Discrete P(sell within H) cross-check |
| `src/demand/hedonic.py` | OLS log close $/sqft → price bounds + floor premium |
| `src/demand/diagnostics.py` | Sign checks, AUC, concordance, calibration, CLI |
| `src/demand/registry.py` | Save/load model bundles + provenance JSON |
| `src/demand/fit.py` | Fit+save entry (CLI + API); calibration gate |
| `src/demand/bootstrap.py` | Bootstrap SEs for generated-regressor premium |
| `src/optimizer/discretize.py` | Per-unit price ladders (comps ∩ cost+margin) |
| `src/optimizer/formulate.py` | Phases, comps, revenue tensor `R[i,j,k]` |
| `src/optimizer/constraints.py` | Seven constraint families |
| `src/optimizer/solve.py` | MILP build/solve, extrapolation, CLI |
| `src/optimizer/crowding.py` | Post-solve cannibalization correction |
| `src/simulation/scenarios.py` | Correlated parameter draws (Cholesky, LHS) |
| `src/simulation/monte_carlo.py` | Fixed-plan re-eval, breach probs, re-solve mode |
| `src/simulation/sensitivity.py` | Tornado ±1σ, LP shadow prices |
| `src/utils/npv.py` | Discount factors (rejects percentage rates) |
| `src/utils/validate.py` | Inventory validation + scoring frame |
| `src/api/*` | Schemas, services, routes, error handlers |
| `tools/make_example_inventory.py` | 60-unit mock inventory |
| `tests/` | pytest suite per phase |

### Frontend (`frontend/`)

| Path | Role |
|---|---|
| `src/App.jsx` | Workspace shell, action buttons, layout |
| `src/context/ProjectContext.jsx` | All app state (no localStorage) |
| `src/api/client.js` | Fetch helpers + SSE simulate parser |
| `src/components/inputs/*` | Upload, constraints, macro sentiment |
| `src/components/outputs/*` | Plan table, revenue summary, risk table |
| `src/components/charts/*` | Heatmap, distribution, timeline, tornado, demand curve |
| `src/components/common/*` | Banner, empty/error/spinner/progress/buttons |
| `vite.config.js` | Dev proxy `/api` → `127.0.0.1:8000` |
| `tailwind.config.js` | Theme tokens: ink/paper/rule/signal/warn |

### Data dirs

| Path | Role |
|---|---|
| `backend/data/raw/mls/` | Real broker exports (gated) |
| `backend/data/synthetic/` | Generated MLS CSVs |
| `backend/data/processed/{market}/` | Reports, model bundles |
| `backend/data/project_inputs/` | `example_inventory.csv` |
| `backend/db/` | Intended SQLite location (often gitignored; unused while `store.py` is stub) |
| `data/` (repo root) | Extra/mirrored quarterly MLS CSVs — **not** the documented pipeline root; operators should use `backend/data/` |

---

## 7. End-to-end data flow

### Path A — Fit demand (synthetic or gated MLS)

```
raw CSV/XLSX
  → ingest_mls / normalize / clean
  → build_features (+ optional premium residual)
  → Cox.fit + logistic.fit + fit_hedonic
  → run_diagnostics  (FAILED if β≥0)
  → save_bundle (pickle + metadata.json)
       provenance.is_calibrated_on_real_data defaults FALSE
```

### Path B — Optimize a project

```
inventory CSV (developer)
  → validate_inventory
  → load ModelBundle (cox + hedonic)
  → build_price_ladder (comps band ∩ cost×(1+margin))
  → build_revenue_tensor
       for each unit i, phase j, level k:
         P = cox.predict_sale_probability(features_j, price_k)
         R = price × area × P × discount_factor[j]
  → solve_release_plan (MILP)
  → OptimizeResult + provenance + extrapolation report
  → optional crowding correction
```

### Path C — Simulate / sensitivity

```
plan + inventory + ScenarioSpec
  → draw correlated shocks
  → closed-form PH probability update
  → Bernoulli sales → revenue distribution, breach probs
  → tornado (±1σ expected revenue)
  → LP-relaxed duals on cash-flow floors
```

### Path D — UI loop

```
Fit synthetic → Upload example_inventory → set phases
  → Optimize → Simulate (SSE) → Sensitivity
  → click unit → Demand curve
Banner always on while uncalibrated
```

---

## 8. Phase 0 — Scaffolding

- FastAPI app with CORS for Vite (`localhost:5173`).
- `GET /api/health` → `{status, version}`.
- Frontend shell with theme CSS variables and fonts (Sora, IBM Plex Sans/Mono).
- Acceptance: uvicorn + `npm run dev` both up; health visible.

---

## 9. Phase 1 — MLS ingest and normalization

### Purpose

Map messy broker exports onto the **canonical schema** in `MLS_SCHEMA.md`,
coerce types, parse floors/units, assign submarkets from ZIP, clean junk rows,
and report quality — without inventing values.

### Key modules

**`normalize.py`**

- Alias table: e.g. `SqFt Liv Area` → `living_area_sqft`, `AssociationFee` →
  `hoa_monthly` (with header priority when multiple candidates exist).
- Money: strip `$`, commas; empty → null.
- Dates: try multiple formats; track which format won per column.
- Status: map to `SOLD|EXPIRED|WITHDRAWN|CANCELED|ACTIVE|PENDING`.
- Floor: from `unit_floor` or parse `unit_number` / address; reject vs
  `total_stories` when implausible; `floor_source` markers.
- HOA frequency: annual→/12, quarterly→/3, etc. Into monthly `hoa_monthly`.

**`clean.py`**

- Drop distressed sale types, non-condo types, price outliers per config filters.
- Dedupe listing episodes.
- Never impute floors or prices.

**`ingest_mls.py`**

- Orchestrates read → normalize → clean.
- `--inspect` report: header mapping, status mix, submarket coverage, cell
  counts, censoring, date formats, export-window / truncation warnings.

### Identification intuition

Elasticity needs, **within the same submarket-month**, different relative ask
prices and different outcomes — including **non-sales** (expired/withdrawn).
Sold-only samples select on the outcome and bias `β_price` toward zero.

---

## 10. Phase 1.5 — Synthetic data

### Purpose

Plant a known `true_beta_price`, generate listings, fit, and assert recovery.
This is the only way to know the estimator works before trusting real MLS.

### `synth.py` highlights

- Profiles: `rich` (full fields) vs `like_export` (matches real Miami field
  availability, censoring, status mix).
- Hazard bases:
  - **`realized`**: hazard driven by observed `rel_price_premium` (easy recovery).
  - **`latent`**: hazard driven by latent aggressiveness; observed premium
    contaminated by quality noise (honest bar; matches real export behavior).
- `building_quality_share`: fraction of fair-value noise at building level
  (building FE can absorb it).
- Returns `SyntheticMLS(frame, truth, latent)`.

### Trap

Under default `realized` hazard, Phase 3 acceptance can pass near-trivially.
Always also check `latent` / `like_export`.

---

## 11. Phase 2 — Feature engineering

### The identifying variable

Historically:

```
rel_price_premium = list_ppsf / cell_median_ppsf - 1
```

Cell = `(submarket, list_month)` with fallback ladder (e.g. quarterly /
submarket-only) when cells are thin (`min_cell_listings`).

**Problem:** cell median holds *where/when* fixed but not *what* (studio vs
penthouse). Quality then loads into the premium → classical measurement error →
attenuation of `β_price`.

### Hedonic residual premium (`premium.py`)

Newer path: regress `log(list_ppsf)` on observable unit traits + submarket +
month FE on **all listings** (not sold-only), then take the residual as the
premium. Rules:

- No post-listing / outcome variables on the RHS.
- No seller-motivation fields (those belong *in* the residual).
- Fit on all listings to avoid selecting on sale.

`bootstrap.py` resamples listings and refits hedonic→residual→Cox to get SEs
that respect the generated-regressor problem.

### Other features

| Feature | Meaning |
|---|---|
| `log_floor`, `log_living_area` | Scale controls |
| `hoa_per_sqft` | Monthly HOA / sqft; implausible band nulls with source |
| `is_new_construction` | Age / flag based |
| `inventory_competition` | New listings entering submarket-month (**pre-duration** count to avoid reverse causality) |
| `season` | Winter/spring/summer/fall from list month |

---

## 12. Phase 3 — Demand models

### 12.1 Cox proportional hazards (`survival.py`) — primary

- Outcome: time-to-sale with right-censoring (`EXPIRED`/`WITHDRAWN`/`CANCELED`
  and still-active as censored).
- Key coefficient: `β_price` on the premium.
- Optional building fixed effects (absorbs tower-constant quality; drops nested
  submarket dummies).
- `predict_sale_probability(features, price_ppsf, horizon_days)` for the
  optimizer.
- Stores `PremiumSupport` (range of premiums seen in fit) for extrapolation
  guards.

Under PH, changing the linear predictor by `Δη` updates survival as:

```
S_new = S_hat ** exp(Δη)
P_new = 1 - (1 - P_hat) ** exp(Δη)
```

Monte Carlo uses this closed form — no refit per draw.

### 12.2 Logistic (`logistic.py`) — cross-check

- Binary: sold within horizon `H` (default 180 days), with careful censoring
  rules for the label.
- Should agree in sign on `β_price`; failures surface as WARN in diagnostics.

### 12.3 Hedonic surface (`hedonic.py`) — price bounds, not elasticity

- OLS of `log(close_ppsf)` on unit features, **SOLD only**.
- Yields predicted $/sqft and a comps band:
  `exp(fit ± k × residual_sd)` with `k = hedonic_bound_sd_multiple`.
- Also estimates floor premium curve parameters (α, γ) — **fitted**, not
  config defaults.
- Feeds the optimizer’s comps half of the price ladder.

### 12.4 Design matrix (`base.py`)

- Numeric coercion, categorical expansion, optional scaling, FE columns.
- Drops linearly dependent columns (pivoted QR) so Cox/logistic can converge
  with building FE + rich categoricals.
- Non-numeric mistaken as numeric → `SchemaError` (don’t silent-coerce).

### 12.5 Diagnostics (`diagnostics.py`)

- Sign checks: `β_price < 0` unconditional FAIL if ≥ 0.
- Concordance / AUC floors, calibration deciles.
- Identification report: censored count, premium IQR, CI excludes zero?,
  quality-explained share of the premium.
- CLI refuses real `data/raw` without `--calibration-gate`.

### 12.6 Registry (`registry.py`)

- `ModelBundle`: cox (+ logistic, hedonic) + diagnostics + `Provenance`.
- `is_calibrated_on_real_data` cannot be true unless `data_source == real_mls`.
- Successful fit **never** auto-flips the flag.

---

## 13. Phase 4 — Optimizer (MILP)

### Why discretize

`price × P(price)` is nonlinear. Enumerate `K` prices per unit (default 15),
precompute revenue → linear objective in binaries `y[i,j,k]`.

### Price ladder (`discretize.py`)

For each unit:

```
comps_floor, comps_ceiling  ← hedonic surface (as_of pinned time controls)
cost_floor                  ← cost_basis_ppsf × (1 + min_margin_over_cost)
p_floor                     ← max(comps_floor, cost_floor)
p_ceiling                   ← comps_ceiling
```

If `cost_floor > comps_ceiling` → exclude (`cost_exceeds_comps`), do not pin a
fake one-point ladder.

`as_of_pricing_frame`: pin categorical time controls to a fitted period so
future phase dates don’t invent unseen quarter levels.

### Revenue tensor (`formulate.py`)

```
R[i,j,k] = price[i,k] × area[i] × P(sell | features_{i,j}, price[i,k]) × DF[j]
```

Also stores `rel_price_premium[i,j,k]`, `releasable[i,j]` (construction gate),
`discount_factor[j]`.

`Comps.require_coverage`: every inventory submarket needs a median $/sqft or
`SchemaError` (no silent drop).

### Constraints (`constraints.py`)

| Family | Intent |
|---|---|
| At most once | Σ_{j,k} y[i,j,k] ≤ 1 |
| Monotone price path | For `unit_type × floor_bucket`, later phases’ ladder **index** ≥ earlier (all phase pairs) |
| Cash-flow floor | Expected phase revenue ≥ CF_min (nominal by default) |
| Max units / phase | Sales capacity |
| Construction gate | No release before `completion − presale_lead` |
| Type diversity | Min counts of types if phase active |
| Price bounds | Built into ladder |

Monotonicity on **index** not raw $/sqft avoids size confounding.

### Solve (`solve.py`)

- PuLP maximize discounted expected revenue.
- Prefer `COIN_CMD` if system CBC present; else `PULP_CBC_CMD`.
- Infeasible → optional relaxation diagnostic (drop families in order).
- **ExtrapolationReport**: chosen premiums vs fitted support.
- Ceiling-share caveat if too many units at top of ladder (“charge the max”
  alarm).
- `DemandProvenance.as_response_block()` always attached.

### Crowding (`crowding.py`)

Post-solve scale: `1 − λ × (similar_released / buyer_pool)`. Does **not**
re-optimize staging. λ and pool are assumptions.

---

## 14. Phase 5 — Monte Carlo and sensitivity

### Scenarios (`scenarios.py`)

Channels:

| Channel | Fitted? | Meaning |
|---|---|---|
| `beta_price` | **Yes** — `Normal(β̂, SE(β̂))` | Elasticity uncertainty |
| `absorption` | No | Log-hazard market shift |
| `competing_listings` | Coeff fitted, shock size no | Inventory competition shift |
| `comps_drift` | No | Comp median moves; fixed ask → different premium |
| `completion_delay_months` | No | Construction slip (clipped ≥ 0) |

Correlated via correlation matrix + Cholesky. Non-PSD matrix → hard error.
Rejects `beta_price_mean ≥ 0` (don’t simulate broken identification).

Macro channel dispersions are DERIVED FROM DATA (`src/data/macro.py`, FRED+BLS),
not typed by the user: `comps_drift` from Case-Shiller Miami return volatility,
`absorption` from `−Δlog(median DOM)` volatility (hazard≈1/time-to-sale identity,
not a fitted rate→hazard coefficient), `competing_listings` from the active-
listing relative swing × the plan's assumed level. Correlations among these three
are estimated from history; `beta_price`'s stay documented priors. User override
is per-channel ("input custom"); offline → documented fallback, labelled in
provenance. Mortgage rate / CPI / unemployment are reported context.

### Monte Carlo (`monte_carlo.py`)

**Default:** fix the optimal plan; re-evaluate revenue under draws.

Two randomness layers:

1. Parameter draws,
2. Bernoulli sales given `P`.

Why both: covenants care about **realized** cash; parameter-only variance
understates spread on ~60 units.

Reports: P5/P25/P50/P75/P95, CVaR@5%, variance decomposition, phase breach
probs, buffered floor suggestion if breach > ~5%, `P(plan > price-at-comps)` on
**same schedule**.

**Optional:** LHS re-solve (~100 MILPs) for plan stability / regret — slow.

### Sensitivity (`sensitivity.py`)

- Tornado: ±1σ one channel at a time on **expected** discounted revenue.
- Shadow prices: LP relaxation (`y ∈ [0,1]`), dual on cash-flow floors;
  report cost = −π for max problems.

---

## 15. Phase 6 — API

| Method | Route | Notes |
|---|---|---|
| GET | `/api/health` | Liveness |
| GET | `/api/config/{market}` | YAML surface + calibration flag |
| POST | `/api/inventory/validate` | JSON rows or CSV/XLSX body; row errors, not 500 |
| POST | `/api/demand/fit` | `dataset=mls` requires `calibration_gate: true` → else 403 |
| GET | `/api/demand/current` | Bundle metadata |
| POST | `/api/optimize` | Plan + **provenance** |
| POST | `/api/simulate` | SSE progress + result + **provenance** |
| POST | `/api/sensitivity` | Tornado (+ shadow prices) |
| POST | `/api/demand/curve` | P(sell) vs ladder prices for one unit |

Domain exceptions → structured 4xx (`SchemaError`, `IdentificationError`,
`InfeasibleModelError`). Upload size/row caps in routes.

Fit-time submarket medians may be stored on provenance/planted_truth so
optimize can resolve comps when the client omits them.

---

## 16. Phase 7 — Frontend

### Design system (brief-mandated)

- Colors: `--ink`, `--paper`, `--rule`, `--signal`, `--warn`, `--muted`
- Fonts: Sora (display), IBM Plex Sans (UI), IBM Plex Mono + `tabular-nums`
  for prices
- Signature view: **building cross-section heatmap** (floor × stack, fill =
  recommended $/sqft)

### UX rules

- No `localStorage` / `sessionStorage`
- No default form submit
- Loading / empty / error states required
- Buttons: Optimize → Optimizing… → Optimized
- Non-dismissible calibration banner
- `prefers-reduced-motion` respected

### State

`ProjectContext` owns inventory, phases, plan, distribution, sensitivity,
selected unit, demand curve, busy flags, errors. Hooks under `hooks/` are thin
compat shims.

### Charts caveat

Revenue distribution Plotly chart **sketches** density from returned
percentiles (API does not stream full draws). Summary P5/P50/P95/CVaR numbers
are authoritative.

---

## 17. Config and markets

`backend/config/miami.yaml`:

- **submarkets** + ZIP lists (incl. extras added so real export coverage clears
  warnings)
- **view_categories** ordinal labels (premiums fitted, not fixed)
- **defaults:** discount rate, presale lead, horizon, ladder levels, crowding λ,
  MC draws, hedonic band width, min margin over cost, **`is_calibrated_on_real_data`**
- **filters:** min list price, exclude distressed, condo/co-op only

Bucharest: config seam only in this pass.

---

## 18. Provenance and the calibration gate

### Provenance block (required on optimize/simulate)

```json
{
  "demand_model": "cox_ph",
  "fitted_on": "synthetic",
  "is_calibrated_on_real_data": false,
  "beta_price": -1.58,
  "beta_price_se": 0.11,
  "beta_price_ci95": [-1.80, -1.36],
  "independence_assumption": true,
  "crowding_correction_applied": false,
  "warnings": ["Figures are illustrative: ..."]
}
```

### Clearing the banner (human only)

1. Inspect real MLS (`ingest --inspect`, fit with `--calibration-gate`).
2. Read: censored count, premium IQR, `β_price` + 95% CI.
3. If CI includes 0 → **stop**; fix data; do not tune until sign flips.
4. Set `defaults.is_calibrated_on_real_data: true` in YAML.
5. Re-fit with `--assert-calibrated` (API: `assert_calibrated: true`).

A successful fit alone never flips the flag.

---

## 19. Inventory fields (developer upload)

Required for validation / optimizer:

| Field | Meaning |
|---|---|
| `unit_id` | Unique id |
| `floor` | Integer floor |
| `living_area_sqft` | Interior sqft |
| `beds` | Bedroom count |
| `unit_type` | e.g. one_bed, two_bed, penthouse (monotonicity groups) |
| `submarket` | Must exist in comps / config |
| `completion_date` | Building complete; gates presale |
| `cost_basis_ppsf` | All-in developer cost **$/sqft** |

Optional: `baths`, `view`, `building`, `hoa_monthly` (monthly USD dues).

**`cost_basis_ppsf`:** not from MLS. Sets
`cost_floor = cost × (1+margin)`. If above comps ceiling → unit excluded.

**`hoa_monthly`:** monthly association fee USD; model uses `hoa_per_sqft`.

Mock generator: `python -m tools.make_example_inventory` → 60 units,
Marina Tower (Brickell) + Bayline North (Edgewater).

---

## 20. Potential bugs and failure modes

This section is the practical threat model.

### 20.1 Identification / demand

| Failure | Symptom | What to do |
|---|---|---|
| β ≥ 0 | Diagnostics `FAILED` | Do not optimize for clients; fix data/spec |
| CI includes 0 | Weak ID | Same — stop at gate |
| Sold-only / truncated expired | Attenuated β | Re-pull export; include actives; list-date sample |
| Export window selection | Slow overpriced listings missing | Warn in ingest; need list-date frame |
| 5000-row status-sorted cap | Truncates expired | Re-pull sorted by ML# |
| Cell median quality mix | Premium = quality | Use residual premium + controls + building FE |
| Endogenous inventory_competition | Spurious coeffs | Must use entry-based (pre-duration) definition |
| Collinearity / rank deficiency | Cox non-converge | Design matrix drops dependent cols; FE nests submarket |
| Generated regressor | Cox SE too small/wrong | Use bootstrap SEs |
| Positive β draws in MC | Worlds with no price deterrence | Scenario notes; large share ⇒ SE too big |

### 20.2 Units and money

| Failure | Symptom |
|---|---|
| Total price in `cost_basis_ppsf` | Absurd floors / validation fail |
| Annual HOA in monthly field | Implausible `hoa_per_sqft`; nulled or wrong amenity signal |
| Discount rate `12` instead of `0.12` | Future phases worthless; `SchemaError` in `npv.py` |
| Mixing sqft and sqm | Silent scale disaster |

### 20.3 Optimizer

| Failure | Symptom |
|---|---|
| β ≈ 0 | All units at ceiling; ceiling-share caveat |
| Missing comps submarket | Used to look like “unscorable”; now SchemaError |
| Cost > comps | Unit in `excluded_units` |
| Empty inventory `is_valid=True` | Fixed: empty is invalid |
| Monotone on $ not index | Size masquerades as discount (avoided by design) |
| Independence | Overstated multi-unit phase revenue |
| Expected CF floor | Passes in expectation, breaches in simulation |
| Extrapolation | Prices outside fitted premium support |
| Infeasible | Empty plan + relaxation diagnostic — not “release nothing” advice |

### 20.4 Simulation / API / UI

| Failure | Symptom |
|---|---|
| Simulate without ladder | No comps baseline comparison |
| MLS fit without gate | 403 REFUSED |
| Assert calibrated without YAML true | REFUSED |
| SSE without result event | Client error |
| Distribution chart vs summary | Chart is sketch; trust numeric percentiles |
| localStorage of prices | Forbidden — stale synthetic in a deck |

### 20.5 Process bugs (already hit in development)

- Fitting real MLS during Phase 3 by accident → calibration gate CLI flags.
- Quarterly median computed on biased subsample → fixed to use all usable rows.
- Penthouses without floor silently dropped from fit → now counted in reports.
- `view_description` all-null treated numeric → SchemaError path.
- Crowding provenance warnings list aliased/mutated → copy before mutate.
- PuLP deprecations (`PULP_CBC_CMD`, constraint dict access) — prefer
  `COIN_CMD` / `get_constraint_by_name`.

---

## 21. Known modeling simplifications

Documented in `AGENTS.md` / `README.md` — **do not silently remove**:

1. **Independence** of sale probabilities across units in a phase (crowding is
   post-hoc only).
2. **Expected-value** cash-flow constraints (use MC breach + buffer).
3. **No competitor reaction** (exogenous comps / competing listings).
4. **Static demand** over the sales horizon (no between-phase refit).
5. **Real export sampling** (off-market window, 5k cap) attenuates β in ways
   the sample cannot measure.
6. Macro channel dispersions are **derived from FRED+BLS volatility** via
   coefficient-free mappings (no fitted rate→hazard elasticity); the absorption
   channel uses the `hazard≈1/DOM` identity, not an estimated sensitivity.

---

## 22. Engineering debt and stub surfaces

From the live codebase inventory (and README known gaps), these are real but
deliberate or deferred — not silent TODOs inside critical paths:

| Item | Status |
|---|---|
| `src/data/store.py` | Stub; persistence is files under `data/processed/` + pickle bundles |
| Bucharest | Config schema only; do not run as a market |
| Optional LLM column-map | Mentioned in brief history; not wired |
| `normalize` performance | Some paths use slow `.iterrows()` — correctness first |
| Pickled model bundles | Fragile across library upgrades; `metadata.json` is the durable record |
| Weak Cox concordance on synth | Expected; coefficient can still be useful |
| Logistic on real export | Often singular → WARN, not silent pass |
| Repeat listings | Not used as within-unit identification strategy |
| View / sale_type gaps on Miami export | Limit controls; attenuation remains |
| Root `data/` vs `backend/data/` | Prefer `backend/data/` for all pipeline CLIs |
| `audit/` scripts | Offline investigation (premium wedge, sampling window, leakage, etc.); pair with `tests/test_audit_regressions.py` |

---

## 23. Testing strategy

- **Synthetic planted truth** is the gold standard (`test_demand_synthetic.py`).
- Optimizer: elasticity response (β soft vs hard changes prices);
  β=0 → all ceilings; infeasibility names binding family; extrapolation tests.
- Simulation: 10k draws &lt; 60s; distribution widens with SE(β); percentiles
  ordered; breach reporting; tornado; slow re-solve marked `@pytest.mark.slow`.
- API: happy path per route; bad inventory → row errors not 500; MLS without
  gate → 403; provenance present.
- Audit regressions: `test_audit_regressions.py` pins calibration flag,
  list-date recovery, sampling window, null handling, path traversal,
  metamorphic checks (F01–F21 style).
- Fixtures: `tests/fixtures/messy_mls_20.csv` for Phase 1.
- Run: `cd backend && pytest -m "not slow"`.

---

## 24. Phase acceptance criteria (brief)

| Phase | Accept (short) |
|---|---|
| **0** | `/api/health` OK; Vite shell can reach API |
| **1** | Messy + synthetic ingest → typed canonical frame; warn if zero non-`SOLD` |
| **1.5** | Synth passes normalize/clean; planted `true_beta_price`; status mix sane |
| **2** | On synth, smoke signal `corr(premium, event_sold) < 0` |
| **3** | Recover planted β on rich/realized; CI excludes 0; β≥0 → `FAILED` |
| **4** | ~60-unit solve Optimal quickly; elastic β lowers prices; β=0 → ceilings |
| **5** | 10k MC &lt; 60s; wider SE → wider revenue band; ordered percentiles |
| **6** | All routes; provenance on optimize/simulate; bad inventory ≠ 500 |
| **7** | Full synthetic UI loop; non-dismissible illustrative banner |
| **Gate** | Real MLS fit is human-only; CI including 0 → stop |

---

## 25. How to run everything

```bash
# Backend
cd backend && source .venv/bin/activate
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# CLIs (from backend/)
python -m src.data.synth --profile rich --n 6000 --seed 42 --beta -1.6
python -m src.data.ingest_mls --inspect --file data/synthetic/...
python -m src.data.features --inspect --file ...
python -m src.demand.diagnostics --inspect --controls --file ...
python -m src.demand.fit --dataset synthetic --controls
python -m src.optimizer.solve --inspect
python -m src.simulation.monte_carlo --inspect --tornado --shadow-prices
python -m tools.make_example_inventory

# Real MLS (human gate)
python -m src.data.ingest_mls --inspect
python -m src.demand.fit --dataset mls --calibration-gate --controls --building-fe
```

UI loop: **Fit synthetic** → drop `example_inventory.csv` → **Optimize** →
**Simulate** → **Sensitivity** → click heatmap/table for demand curve.

---

## 26. Glossary

| Term | Definition |
|---|---|
| `β_price` | Cox/logit coefficient on relative price premium; core elasticity |
| `rel_price_premium` | Ask vs reference − 1 (cell median or hedonic residual ratio) |
| Comps | Submarket (etc.) median $/sqft used for relative pricing |
| Hedonic surface | Model of fair $/sqft from sold comps → optimizer bounds |
| Price ladder | Discrete $/sqft grid per unit for the MILP |
| Phase | Release wave with start month and constraints |
| Construction gate | Cannot sell before completion − presale lead |
| Provenance | Metadata proving synthetic vs calibrated real fit |
| Censoring | Listing observed without sale by terminal/admin time |
| CVaR@5% | Mean of worst 5% of simulated revenues |
| LHS | Latin hypercube sampling for efficient scenario coverage |
| Crowding | Post-hoc correction for within-phase cannibalization |

---

## Appendix A — Mental model for `β_price`

Interpretation sketch (log-hazard): if `β_price = -1.6` and a unit is priced
10% above its reference (`rel = 0.10`), the sale hazard scales by
`exp(-1.6 × 0.10) ≈ 0.85` relative to a unit at the reference, holding other
covariates fixed.

If controls leave quality inside the premium, the fitted `|β|` is a **lower
bound** on true price sensitivity (attenuation). Building FE and residual
premiums exist to reduce that contamination — they do not create elasticity
from nothing.

---

## Appendix B — Objective in one line

```
max Σ_{i,j,k}  y[i,j,k] · price_{i,k} · area_i · P̂(sell | i,j,price_{i,k}) · DF_j
```

subject to the constraint families in §13, with `P̂` coming from the fitted Cox
model and `y` binary (or continuous in the LP relaxation for duals).

If `P̂` does not fall when `price` rises, this program’s optimum is the ceiling.
That is the whole product in miniature.

---

*End of MASTER.md. When behavior and this document disagree, trust the code and
tests, then update this file.*
