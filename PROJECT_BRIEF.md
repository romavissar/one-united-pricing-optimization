# PROJECT BRIEF — Dynamic Pricing & Release Optimizer

**Read this file first. Then read `AGENTS.md` and `MLS_SCHEMA.md` before writing any code.**

---

## 0. What you are building

A decision tool for luxury residential developers. Given a project's unit inventory
(each unit's floor, size, view, type), it outputs:

1. A **price for every unit**, and
2. A **phased release schedule** (which units go on sale in which phase),

chosen to maximize the discounted expected revenue of the whole project, subject to
real business constraints.

**Priority market: Miami, FL.** Currency USD, area in **sqft**, price quoted as
**$/sqft**. Bucharest/Romania is a secondary market — build the config seam for it,
but do not build out its data pipeline in this pass.

### The three-layer engine

```
  ┌────────────────────────────────────────────────────────────┐
  │  OPTIMIZER (MILP)                                          │
  │  Picks price + phase per unit to maximize NPV of revenue   │
  └───────────────────────────▲────────────────────────────────┘
                              │ needs a calibrated D()
  ┌───────────────────────────┴────────────────────────────────┐
  │  DEMAND MODEL                                              │
  │  D(price, features, macro) → P(sells within horizon)       │
  │  and/or hazard of selling at time t                        │
  └───────────────────────────▲────────────────────────────────┘
                              │ needs price/outcome contrast
  ┌───────────────────────────┴────────────────────────────────┐
  │  DATA LAYER                                                │
  │  MLS records: list price, sale price, DOM, status, features│
  └────────────────────────────────────────────────────────────┘
```

Then a **Monte Carlo** layer wraps the optimizer to report a revenue *distribution*
rather than a point estimate, and a **React frontend** presents it.

---

## 1. The one problem this project exists to solve

The optimizer's objective is, per unit:

```
expected_revenue = price × area × P(sells | price, features, macro)
```

Raising the price raises the first term and lowers the third. The optimum depends
entirely on **how fast `P(sells)` falls as price rises** — the own-price elasticity,
`β_price`.

**If `β_price` is not identified, the optimizer is worthless.** With no price
sensitivity in `D()`, the objective becomes monotonically increasing in price and the
solver will just push every unit to its price ceiling. That output is mathematically
"optimal" and practically absurd.

To identify `β_price` you need records where, **within the same submarket and the same
month**, comparable units were offered at **different relative prices** and you observe
**different outcomes** (sold fast / sold slow / never sold).

MLS data provides exactly this, because it carries:
- `original_list_price` **and** `close_price` (the market's pushback, per unit)
- `list_date`, `close_date`, `days_on_market` (speed of sale)
- **Expired / Withdrawn / Canceled** listings — units that were offered and **did not
  sell**. These are the negative outcomes. Without them the model only sees winners and
  concludes everything sells eventually.

Everything in this build serves that estimation.

---

## 2. Tech stack — locked, do not substitute

| Layer | Choice | Notes |
|---|---|---|
| Backend | **Python 3.11+, FastAPI** | `uvicorn` dev server |
| Optimizer | **PuLP** (CBC solver) primary; OR-Tools as fallback | Problem is a MILP after linearization |
| Demand model | **lifelines** (Cox PH), **scikit-learn** (logistic) | `statsmodels` for diagnostics |
| Numerics | numpy, scipy, pandas | |
| Storage | **SQLite** + **Parquet/CSV** flat files | No Postgres, no ORM server, no Docker |
| Frontend | **React 18 + Vite + Tailwind + Plotly.js** | |
| Tables | **TanStack Table v8** | |
| Uploads | react-dropzone + Papa Parse (CSV) + SheetJS (xlsx) | |
| Transport | REST/JSON over HTTP; SSE for long-run progress | No GraphQL, no WebSockets |

**Explicitly rejected:** Streamlit, Next.js, Electron, Docker, Postgres, Redis,
message queues, microservices. Do not introduce them.

---

## 3. Repository layout

Create exactly this. Do not invent parallel structures.

```
pricing-optimizer/
├── README.md
├── AGENTS.md
├── MLS_SCHEMA.md
├── .env.example
├── .gitignore
│
├── backend/
│   ├── main.py                       # FastAPI app + routes
│   ├── requirements.txt
│   ├── config/
│   │   ├── miami.yaml                # submarkets, view categories, bounds, tax/HOA rules
│   │   └── bucharest.yaml            # STUB in this pass (schema only, values TBD)
│   ├── src/
│   │   ├── config.py                 # load + validate market config
│   │   ├── data/
│   │   │   ├── ingest_mls.py         # read raw MLS export → normalized frame
│   │   │   ├── normalize.py          # column mapping, type coercion, unit parsing
│   │   │   ├── clean.py              # dedup, outliers, sale-type filter
│   │   │   ├── features.py           # derived features incl. relative price premium
│   │   │   ├── synth.py              # SYNTHETIC MLS generator (see §5.3)
│   │   │   └── store.py              # SQLite + Parquet read/write helpers
│   │   ├── demand/
│   │   │   ├── base.py               # DemandModel protocol / ABC
│   │   │   ├── survival.py           # Cox PH: hazard of sale vs relative price
│   │   │   ├── logistic.py           # P(sold within horizon) logistic
│   │   │   ├── hedonic.py            # floor / view / area price surface (OLS)
│   │   │   ├── diagnostics.py        # sign checks, AUC, concordance, calibration
│   │   │   └── registry.py           # save/load fitted models + metadata
│   │   ├── optimizer/
│   │   │   ├── formulate.py          # build the MILP
│   │   │   ├── discretize.py         # price ladder → binary y[i,j,k]
│   │   │   ├── constraints.py        # market-specific constraint builders
│   │   │   ├── solve.py              # solver interface + status handling
│   │   │   └── crowding.py           # optional within-phase cannibalization factor
│   │   ├── simulation/
│   │   │   ├── monte_carlo.py
│   │   │   ├── scenarios.py          # parameter distributions + correlation
│   │   │   └── sensitivity.py        # tornado, dual values
│   │   └── utils/
│   │       ├── npv.py
│   │       └── validate.py           # inventory schema validation
│   ├── data/
│   │   ├── raw/mls/                  # drop the real MLS export here (git-ignored)
│   │   ├── synthetic/                # generated synthetic MLS
│   │   ├── processed/miami/
│   │   └── project_inputs/
│   │       └── example_inventory.csv
│   ├── db/                           # sqlite file (git-ignored)
│   └── tests/
│       ├── test_normalize.py
│       ├── test_features.py
│       ├── test_demand_synthetic.py  # THE key test — see §5.4
│       ├── test_optimizer.py
│       ├── test_simulation.py
│       └── test_api.py
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── api/client.js
        ├── context/ProjectContext.jsx
        ├── hooks/{useOptimize.js,useSimulate.js,useInventoryUpload.js}
        ├── components/
        │   ├── inputs/{InventoryUpload.jsx,Constraints.jsx,MacroAssumptions.jsx}
        │   ├── outputs/{ReleasePlanTable.jsx,RevenueSummary.jsx,RiskTable.jsx}
        │   ├── charts/{BuildingHeatmap.jsx,RevenueDistribution.jsx,PhaseTimeline.jsx,SensitivityTornado.jsx,DemandCurve.jsx}
        │   └── common/{Spinner.jsx,ProgressBar.jsx,EmptyState.jsx,ErrorBoundary.jsx}
        └── utils/format.js
```

---

## 4. Build order

Work through these in order. **Each phase must satisfy its acceptance criteria before
you start the next.** Run the tests. Do not batch-build everything and test at the end.

### Phase 0 — Scaffolding

- Repo layout above; `.gitignore` covering `.env`, `db/`, `data/raw/`, `node_modules/`,
  `__pycache__/`, `*.parquet`.
- `backend/requirements.txt`, `frontend/package.json` with the locked stack.
- `.env.example` (see the separate file provided).
- `config/miami.yaml` populated per §6.
- FastAPI app boots with `GET /api/health` → `{"status":"ok","version":"0.1.0"}`.
- Vite dev server boots, proxies `/api` → `http://localhost:8000`.

**Accept:** `uvicorn main:app --reload` serves `/api/health`; `npm run dev` renders a
shell page that successfully fetches and displays the health payload.

---

### Phase 1 — MLS ingestion and normalization

Read `MLS_SCHEMA.md` in full before starting.

- `ingest_mls.py`: read a CSV/XLSX export from `data/raw/mls/`. Accept **multiple files**
  and concatenate (broker exports are often capped at 500–1000 rows, so the real data
  arrives in batches).
- `normalize.py`:
  - Map source columns → canonical names using the alias table in `MLS_SCHEMA.md`.
    Matching must be **case-insensitive and whitespace/punctuation-tolerant**
    (`"Orig List Price"`, `original_list_price`, `ORIGINAL LIST PRICE` all match).
  - Coerce types; parse dates to `datetime64`; strip `$` and `,` from money fields.
  - Parse `unit_number` out of the address when it is not its own column.
  - Derive `floor` from `unit_number` where a convention is detectable
    (`#2506` → floor 25, `#3101` → floor 31, `PH-2` → penthouse flag). **Record
    `floor_source` as `"parsed"` vs `"reported"` vs `"missing"`** — never silently
    invent a floor.
  - Map `status` to the canonical enum: `SOLD | EXPIRED | WITHDRAWN | CANCELED | ACTIVE`.
- `clean.py`:
  - Drop non-standard sale types (foreclosure, short sale, auction, REO).
  - Drop rentals/leases if present.
  - Deduplicate on `mls_number`; if a unit was relisted, keep each listing episode as a
    separate row but link them via `unit_key` (see below).
  - Outlier flags: `list_ppsf` beyond ±3σ of its submarket×quarter distribution →
    flag, do not drop silently. Log counts.
- `unit_key`: a stable identifier for the physical unit =
  `normalize(building_name or street_address) + "|" + unit_number`. Used to link relists.

**Accept:**
- Running ingestion on the synthetic file (Phase 1.5) and on a hand-made 20-row fixture
  with deliberately messy headers produces a frame with all canonical columns present
  and correctly typed.
- A validation report prints: rows in, rows out, rows dropped by reason, null rate per
  canonical column, status distribution, `floor_source` distribution.
- **Hard check:** the report must state the count of non-`SOLD` records. If that count
  is zero, print a loud warning that elasticity is **not identifiable** from this file.

---

### Phase 1.5 — Synthetic MLS generator

This is what lets you build and verify everything downstream before touching the real
file. Do not skip it.

`src/data/synth.py`:

```
generate_synthetic_mls(
    n=6000, seed=42,
    true_beta_price=-1.6,     # the ground-truth elasticity to recover
    horizon_days=180,
) -> pd.DataFrame
```

It must emit **exactly the canonical MLS schema** from `MLS_SCHEMA.md`, and it must
generate data via a known data-generating process so tests can check that the demand
model recovers the planted parameters:

1. Draw units: submarket, sqft, beds, baths, floor, year_built, hoa_monthly, view proxy.
2. Compute a "fair" $/sqft from a hedonic surface (submarket base × floor premium ×
   view premium × size effect) plus noise.
3. Draw a **listing aggressiveness** `a_i ~ Normal(0, 0.08)`: the seller's chosen list
   price is `fair_ppsf × (1 + a_i)`. This is the variation that identifies elasticity —
   it must be **independent of the unobserved noise in fair value** so the synthetic
   case is cleanly identified.
4. Time-to-sale from an exponential/Weibull hazard where
   `log(hazard) = base(submarket, month) + true_beta_price × relative_price_premium_i`.
5. If simulated time-to-sale ≤ `horizon_days` → `status = SOLD`, set `close_date`,
   and `close_price = list_price × (1 - concession(a_i))`. Otherwise →
   `status ∈ {EXPIRED, WITHDRAWN, CANCELED}` with a plausible split, `close_price = NaN`.
6. Spread `list_date` across 24 months so month×submarket cells are populated.

**Accept:** the generated frame passes the exact same `normalize` + `clean` + validation
path as a real export, with zero schema errors, and its status mix is roughly
55–75% `SOLD`.

---

### Phase 2 — Feature engineering

`src/data/features.py`. The critical feature is the identification variable:

```
list_ppsf              = original_list_price / living_area_sqft
submarket_month_median = median(list_ppsf) over (submarket, list_month) cell
rel_price_premium      = list_ppsf / submarket_month_median - 1
```

`rel_price_premium` is **the** covariate whose coefficient is `β_price`. Computing the
median **within submarket × list-month** is what holds the macro regime and location
fixed, so the remaining variation is the seller's pricing choice rather than calendar
drift. This is the whole reason the estimate is credible — do not compute the median
globally or over the pooled period.

Guard: require ≥ 8 listings in a `(submarket, list_month)` cell before using its median;
otherwise fall back to `(submarket, list_quarter)` and set `median_basis` accordingly.

Other features:

| Feature | Definition |
|---|---|
| `log_floor` | `ln(floor + 1)`, null where `floor_source == "missing"` |
| `floor_bucket` | low 1–5, mid 6–12, high 13–25, tower 26+ |
| `size_bucket` | quantile bucket of `living_area_sqft` within submarket |
| `bed_bath_ratio` | `baths / max(beds, 1)` |
| `hoa_per_sqft` | `hoa_monthly / living_area_sqft` |
| `is_new_construction` | from MLS flag; else `list_year - year_built <= 2` |
| `list_month`, `list_quarter`, `season` | from `list_date` |
| `duration_days` | `SOLD` → `close_date - list_date`; else `expiry-ish date - list_date`. Fall back to reported `days_on_market` when a terminal date is missing. |
| `event_sold` | `1` if `status == SOLD` else `0` |
| `sold_to_list_ratio` | `close_price / original_list_price` (SOLD only) |
| `price_cut_pct` | `last_list_price / original_list_price - 1` |
| `inventory_competition` | count of listings active in same submarket in that month |

**Accept:** on synthetic data, `rel_price_premium` has mean ≈ 0, and its correlation with
`event_sold` is **negative**. Print that correlation — it is your first smoke signal that
identification is present.

---

### Phase 3 — Demand model

Two estimators behind one interface (`demand/base.py`), because they answer different
questions and cross-check each other.

**3a. Survival (primary) — `demand/survival.py`**

Cox proportional hazards on time-to-sale.
- `duration_col = duration_days`, `event_col = event_sold`.
- `EXPIRED / WITHDRAWN / CANCELED` are **right-censored**: the listing was observed for
  its duration and no sale occurred. This is correct censoring, not a missing outcome.
- Covariates: `rel_price_premium`, `log_floor`, `living_area_sqft`, `beds`,
  `hoa_per_sqft`, `is_new_construction`, `submarket` (categorical), `season`,
  `inventory_competition`.
- Expose `predict_sale_probability(features, price, horizon_days) -> float` by
  evaluating the fitted survival function at `horizon_days`.

**3b. Logistic (secondary) — `demand/logistic.py`**

`P(sold within H days)` with the same covariates. Faster to reason about, and its
coefficient sign is an easy sanity check.

**3c. Hedonic surface — `demand/hedonic.py`**

OLS of `log(close_ppsf)` on unit features (SOLD records only). This gives the
**relative** price structure — what a floor is worth, what a view is worth — which is
separate from and complementary to elasticity. It also produces the per-unit price
bounds the optimizer needs.

Fit the floor premium as `premium(floor) = α · ln(floor + 1) + γ` **with α and γ
estimated from data**. Do not hardcode values. (For reference, a Bucharest-calibrated
illustrative pair is α ≈ 0.058, γ ≈ -0.038; Miami towers are much taller and the curve
flattens differently, so treat these purely as sanity-range hints, not defaults.)

**3d. Diagnostics — `demand/diagnostics.py`**

Must run automatically after every fit and **fail loudly** on sign violations:

| Check | Requirement |
|---|---|
| `β(rel_price_premium)` in Cox | **< 0** (higher relative price → lower hazard of sale) |
| `β(rel_price_premium)` in logistic | **< 0** |
| `β(log_floor)` | > 0 |
| `β(inventory_competition)` | < 0 |
| Logistic AUC (held-out) | report; flag if < 0.60 |
| Cox concordance | report; flag if < 0.60 |
| Calibration | decile plot of predicted vs observed sale rate |

Also emit a **`identification_report`** containing:
- count of `SOLD` vs censored records,
- number of `(submarket, list_month)` cells with ≥ 8 listings,
- the interquartile range of `rel_price_premium` (if this is near zero, all sellers
  priced identically and elasticity is **not** identified regardless of sample size),
- the fitted `β_price` with its standard error and 95% CI.

**Accept (this is the important one):** `tests/test_demand_synthetic.py` generates
synthetic data with `true_beta_price = -1.6`, fits the Cox model, and asserts the
recovered coefficient is within a tolerance band of the planted value and that its
95% CI excludes zero. **If this test does not pass, nothing downstream can be trusted.**
Also run it at `true_beta_price = -0.4` and `-3.0` to confirm the pipeline tracks the
parameter rather than always returning the same number.

---

### Phase 4 — Optimizer

**Decision variables.** Discretize price into K levels per unit (`discretize.py`), then

```
y[i,j,k] ∈ {0,1}   unit i released in phase j at price level k
```

**Objective.** Precompute `R[i,j,k] = p_k · area_i · D(p_k, X_i, M_j) · δ_j` where
`δ_j = 1 / (1 + r)^(t_j / 12)`, `r` = **annual** discount rate as a decimal, `t_j` =
months from project start to phase j. Then

```
maximize  Σ_i Σ_j Σ_k  y[i,j,k] · R[i,j,k]
```

Precomputing `R` is what makes the objective **linear**, turning a MINLP into a MILP
that CBC solves in seconds for a few hundred units.

**Constraints** (`constraints.py`):

1. Each unit released at most once: `Σ_j Σ_k y[i,j,k] ≤ 1`
2. **Monotone price path** — for comparable units, phase `j+1` price ≥ phase `j` price.
   A price cut between phases signals distress and kills momentum; this is a hard
   business rule, not a preference.
3. Minimum cash flow per phase: `Σ_i Σ_k y[i,j,k] · R[i,j,k] ≥ CF_min[j]`
4. Max units per phase (sales-team capacity): `Σ_i Σ_k y[i,j,k] ≤ N_max[j]`
5. Construction gate: `y[i,j,k] = 0` when phase `j` starts before unit `i`'s building is
   sellable (`completion_date − presale_lead_months`)
6. Unit-type diversity: each active phase includes ≥ `min_type_count[t]` of type `t`
7. Price bounds: `p_floor[i] ≤ p_k ≤ p_ceiling[i]`, from cost basis and hedonic comps

**Solver handling** (`solve.py`): return a structured result with status. On `Infeasible`,
run a **relaxation diagnostic** — drop constraints one at a time and report which one
restores feasibility. Never return a silent empty plan.

**Known simplification — document it in code and in the API response.** The objective
treats each unit's sale probability as independent of what else is released in the same
phase. In reality, similar units released together compete for one finite buyer pool, so
this **overestimates revenue for phases that dump many near-identical units**. Implement
`crowding.py` as an *optional* post-solve correction:

```
crowding_factor[j, cluster] = 1 − λ · (similar_units_released / buyer_pool)
```

applied per type/segment cluster (a studio and a penthouse do not compete), with
`λ ∈ [0,1]` configurable, default `0.3`. Apply it when reporting expected revenue and
label the reported figure accordingly.

**Accept:**
- On a 60-unit synthetic inventory, solves to `Optimal` in < 10 s.
- Constraint unit tests: monotonicity holds in the returned plan; no unit appears twice;
  every phase meets its cash-flow floor; no unit is released before its construction gate.
- **Elasticity-response test:** solve the same inventory with `β_price = -0.4` and again
  with `β_price = -3.0`. The recommended prices must be **materially lower** in the
  elastic case. If they are identical, the demand function is not actually entering the
  objective — that is a wiring bug and the highest-priority thing to fix.
- **Degenerate-input test:** with `β_price = 0`, assert every unit is pushed to its price
  ceiling. This documents the failure mode the whole project is designed to avoid.

---

### Phase 5 — Monte Carlo and sensitivity

`simulation/monte_carlo.py`:

- Sample uncertain parameters per draw: `β_price ~ Normal(β̂, SE(β̂))` (use the real
  fitted standard error), plus mortgage-rate path, competitor inventory shift,
  absorption shift, construction delay.
- Correlate them — do not sample independently. Higher rates should co-move with more
  negative elasticity. Use a correlation matrix + Cholesky.
- **Default mode:** fix the base optimal plan, re-evaluate its revenue under each drawn
  scenario (fast, thousands of draws). **Optional mode:** re-solve for ~100 LHS-sampled
  scenarios to test whether the *plan itself* is stable.
- Report P5 / P25 / P50 / P75 / P95, CVaR@5%, and `P(plan > flat-pricing baseline)`.

`simulation/sensitivity.py`: tornado diagram from ±1σ perturbations, plus shadow prices
from the LP relaxation (the dual on the cash-flow constraint tells you exactly how much
revenue the covenant is costing).

**Cash-flow note:** constraint 4.3 is an *expected-value* constraint, so realized cash
flow can still fall below `CF_min[j]`. Where the floor is a real covenant, report the
simulated breach probability, and if it exceeds ~5%, raise the effective `CF_min[j]`
with a buffer and re-solve. That approximates a chance constraint without a specialized
solver.

**Accept:** 10,000 draws finish in < 60 s in fixed-plan mode; the output distribution
widens when `SE(β̂)` is increased; percentiles are ordered and finite.

---

### Phase 6 — API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/config/{market}` | submarkets, view categories, bounds, defaults |
| POST | `/api/inventory/validate` | upload CSV/XLSX → parsed + validated inventory, with per-row errors |
| POST | `/api/demand/fit` | fit on a named dataset (`synthetic` or `mls`); returns coefficients + `identification_report` |
| GET | `/api/demand/current` | active fitted model metadata, incl. `is_calibrated_on_real_data: bool` |
| POST | `/api/optimize` | inventory + constraints → release plan |
| POST | `/api/simulate` | plan + uncertainty → revenue distribution (SSE progress) |
| POST | `/api/sensitivity` | plan → tornado data |

**Every** response from `/api/optimize` and `/api/simulate` must include:

```json
"provenance": {
  "demand_model": "cox_ph",
  "fitted_on": "synthetic",
  "is_calibrated_on_real_data": false,
  "beta_price": -1.58,
  "beta_price_se": 0.11,
  "beta_price_ci95": [-1.80, -1.36],
  "independence_assumption": true,
  "crowding_correction_applied": false,
  "warnings": ["Figures are illustrative: demand model fitted on synthetic data."]
}
```

This block is not optional. It is the mechanism that stops a synthetic-data number from
ever being mistaken for a real recommendation.

**Accept:** `tests/test_api.py` covers each route's happy path plus a malformed-inventory
upload returning row-level errors rather than a 500.

---

### Phase 7 — Frontend

Follow this design direction rather than defaulting. Put it in `tailwind.config.js` as
CSS variables / theme extension.

**Palette**
```
--ink      #16202B   deep slate ground, headers, primary text
--paper    #FBFAF7   warm off-white panels
--rule     #D8D4CC   warm grey hairlines, table borders
--signal   #0F6E5C   deep teal — recommended price, optimal, upside
--warn     #B4551F   burnt orange — risk, downside, constraint breach
--muted    #6B7684   secondary text, axis labels
```

**Type** — three roles, loaded from Google Fonts:
- Display / headings: **Sora** (geometric, used with restraint)
- Body / UI: **IBM Plex Sans**
- All numerals in tables and price columns: **IBM Plex Mono** with
  `font-variant-numeric: tabular-nums`

Tabular monospaced figures are the correct call here, not a stylistic flourish: the core
user action is scanning a column of hundreds of prices for anomalies, and digits must
align.

**Signature element:** the **building cross-section heatmap** — units laid out as cells
on a floor (y) × position (x) grid, filled by recommended `$/sqft`. It is the one view
that makes the whole output legible at a glance, and it is specific to this subject
rather than generic dashboard furniture. Spend the visual boldness here and keep
everything around it quiet.

**Screens / components**
- Upload inventory → validation summary with row-level errors
- Constraints panel: phase count, `CF_min` per phase, max units/phase, discount rate,
  presale lead months
- Macro assumptions: mortgage-rate outlook, demand sentiment (pessimistic/base/optimistic)
- Release plan table (TanStack): unit, type, floor, sqft, view, phase, `$/sqft`,
  total price, P(sell), expected revenue — sortable, filterable, CSV export
- Revenue summary: total, uplift vs flat-pricing baseline, 90% CI
- Charts: building heatmap, revenue distribution histogram with percentile markers,
  phase timeline, sensitivity tornado, and a **demand curve** panel plotting
  P(sell) vs price for a selected unit (this is the view that makes elasticity tangible)
- **A persistent banner** whenever `is_calibrated_on_real_data === false`, in `--warn`,
  reading: *"Illustrative output — demand model fitted on synthetic data, not market
  data."* It must not be dismissible.

Empty states are invitations to act, not apologies. Errors say what happened and what to
do next. Buttons name the action and keep that name through the flow ("Optimize" →
"Optimizing…" → "Optimized").

Quality floor, unannounced: responsive to mobile, visible keyboard focus,
`prefers-reduced-motion` respected.

**Accept:** full loop runs against the live backend on synthetic data — upload the
example inventory, set constraints, optimize, simulate, read every chart. The
uncalibrated banner is visible throughout.

---

## 5. 🛑 STOP HERE — the calibration gate

**Do not proceed past this line autonomously.**

At this point the system is complete and verified end-to-end on synthetic data. The
remaining step is fitting the demand model on the **real MLS export**, and that step is
deliberately not automated, because it requires human judgment that cannot be encoded in
advance:

- Real broker exports have idiosyncratic column names, mixed date formats, and blank
  `floor` fields at rates that vary by brokerage. The normalizer will need
  file-specific adjustment once the actual headers are visible.
- Coefficient signs must be inspected, not asserted. If `β_price` comes back positive or
  insignificant, the correct response is diagnostic (is `rel_price_premium` degenerate?
  are censored records missing? is the submarket×month cell count too thin?) — not to
  keep refitting until a number looks acceptable.
- The specification choice (which controls to include, whether to log-transform, whether
  to segment by submarket) depends on what the real data supports.

**What to hand over at the gate.** Print a checklist to stdout and write it to
`CALIBRATION_READY.md`:

1. Where to place the real files: `backend/data/raw/mls/`
2. The exact command to ingest and inspect: `python -m src.data.ingest_mls --inspect`
3. The exact command to fit: `python -m src.demand.fit --dataset mls`
4. The three numbers to read first: count of censored (non-`SOLD`) records, IQR of
   `rel_price_premium`, and `β_price` with its 95% CI
5. The decision rule: **if the 95% CI for `β_price` includes zero, elasticity is not
   identified — stop and fix the data, do not tune the model**
6. Confirmation that flipping `is_calibrated_on_real_data` to `true` is a single,
   explicit config change, not an automatic side effect of a successful fit

---

## 6. `config/miami.yaml` contents

```yaml
market: miami
currency: USD
area_unit: sqft
price_unit: usd_per_sqft

submarkets:
  brickell:            { zips: ["33129","33130","33131"], tier: 1 }
  edgewater:           { zips: ["33137","33138"],          tier: 2 }
  downtown:            { zips: ["33132"],                  tier: 2 }
  miami_beach:         { zips: ["33139","33140","33141"],  tier: 1 }
  sunny_isles:         { zips: ["33160"],                  tier: 1 }
  coconut_grove:       { zips: ["33133"],                  tier: 1 }
  surfside_bal_harbour:{ zips: ["33154"],                  tier: 0 }
  aventura:            { zips: ["33180"],                  tier: 2 }

view_categories:        # ordinal, higher = better; premiums are FIT, not fixed
  - direct_ocean
  - bay_direct
  - ocean_partial
  - bay_partial
  - intracoastal
  - city_skyline
  - pool_garden
  - urban_street
  - obstructed

defaults:
  discount_rate_annual: 0.12
  presale_lead_months: 24
  horizon_days: 180
  price_ladder_levels: 15
  min_cell_listings: 8          # (submarket, month) cell threshold
  crowding_lambda: 0.30
  monte_carlo_draws: 10000

filters:
  min_list_price: 400000
  exclude_sale_types: ["foreclosure", "short_sale", "auction", "reo"]
  property_types: ["condo", "co_op"]

notes:
  floor_premium: "alpha and gamma are ESTIMATED per project from hedonic fit; no defaults"
  hoa_effect: "model via hoa_per_sqft covariate; FL insurance costs make this material"
```

---

## 7. Non-goals for this pass

- Bucharest data pipeline (config stub only)
- Authentication, multi-tenancy, billing
- Any deployment target beyond `localhost`
- Reinforcement learning, neural demand models, LLM-based demand estimation
- Backtesting against a completed project (needs developer-internal phase/price history
  that we do not have)
- Electron/Tauri desktop packaging

---

## 8. The one-sentence test of success

When someone changes `β_price` from `-0.4` to `-3.0`, every recommended price in the
output must move down, and the revenue distribution must widen. If that does not happen,
the pipeline is decorative.
