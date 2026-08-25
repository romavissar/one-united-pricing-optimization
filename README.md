# Pricing & Release Optimizer

Decision tool for luxury residential developers: unit prices + phased release
schedule that maximize discounted expected revenue, subject to real constraints.

**Priority market:** Miami, FL (`$/sqft`, USD, sqft).

This system exists to estimate one number — **β_price**, own-price elasticity of
demand. Read `PROJECT_BRIEF.md`, `AGENTS.md`, and `MLS_SCHEMA.md` before changing
code. For a full deep-dive (architecture, models, bugs, flows), see `MASTER.md`.

## Quick start

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env   # or copy from repo-root env and sanitize
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5173`). The shell page fetches
`GET /api/health` via the Vite proxy and displays the payload.

## Data pipeline

```bash
cd backend && source .venv/bin/activate

# Inspect a real MLS export in data/raw/mls/ (exit 2 when warnings fire)
python -m src.data.ingest_mls --inspect

# Generate the synthetic export with a planted elasticity, then ingest it
python -m src.data.synth --profile rich --n 6000 --seed 42 --beta -1.6
python -m src.data.synth --profile like_export
python -m src.data.ingest_mls --inspect --file data/synthetic/miami_synth_rich.csv

# Build demand features and print the identification smoke signal
python -m src.data.features --inspect
python -m src.data.features --inspect --file data/synthetic/miami_synth_rich.csv

# Fit the demand system and print its diagnostics
# exit 0 clean / 2 warnings / 1 blocked by a failed sign check
python -m src.demand.diagnostics --inspect --file data/synthetic/miami_synth_rich.csv
python -m src.demand.diagnostics --inspect --controls --building-fe \
  --file data/synthetic/miami_synth_rich.csv

# Solve a release plan for an inventory against synthetic demand
# exit 0 optimal / 1 no plan or a monotonicity violation in the output
python -m src.optimizer.solve --inspect
python -m src.optimizer.solve --inspect --crowding-buyer-pool 25
python -m tools.make_example_inventory   # regenerate the 60-unit fixture

# Monte Carlo revenue distribution + tornado / shadow prices (synthetic demand)
python -m src.simulation.monte_carlo --inspect --tornado --shadow-prices

# Fit and save a bundle (synthetic). Real MLS requires --calibration-gate —
# see CALIBRATION_READY.md. A successful fit never flips is_calibrated_on_real_data.
python -m src.demand.fit --dataset synthetic --controls

# API (Phase 6) — from backend/
uvicorn main:app --reload --port 8000
# GET  /api/health
# GET  /api/config/miami
# POST /api/inventory/validate   JSON {market, rows} or text/csv
# POST /api/demand/fit           {dataset: synthetic|mls, calibration_gate?}
# GET  /api/demand/current
# POST /api/optimize             → plan + provenance
# POST /api/simulate             → SSE progress + result (provenance)
# POST /api/sensitivity          → tornado (+ shadow prices)
# POST /api/demand/curve         → P(sell) vs $/sqft for one unit

# UI (Phase 7) — from frontend/ (proxies /api → :8000)
npm run dev
# Full loop: Fit synthetic → upload example_inventory.csv → Optimize →
# Simulate → Sensitivity. Non-dismissible illustrative banner stays up
# while is_calibrated_on_real_data is false.
```

Reports are written to `data/processed/{market}/`, namespaced by source
directory so a synthetic run cannot overwrite the real export's report.

The synthetic generator plants a known `true_beta_price` and returns it in
`SyntheticTruth` alongside a `latent` frame (aggressiveness, fair-value noise,
realized `rel_price_premium`, uncensored time-to-sale). That is what lets
Phase 3 assert *recovery* of a known answer instead of merely producing one.

### Data-availability profiles

Every phase from 2 on must be verified under **both**, because a generator that
emits every canonical field is a poor stand-in for an export that omits ten of
them.

| | `rich` | `like_export` | real export |
|---|---|---|---|
| Canonical fields matched | 34 | 24 | 24 |
| `IMPORTANT` missing | none | unit_number, property_type, sale_type | same |
| SOLD | 58% | 31% | 38% |
| PENDING | 0% | 4.4% | 5.8% |
| Median list $/sqft | $1,329 | ~$680 | $683 |
| `rel_price_premium` IQR | 0.28 | 0.65 | 0.62 |

`like_export` withholds `view_description`, but the view premium is still baked
into fair value — so it becomes genuine unobserved quality, which is exactly
what the real export leaves you with.

### Hazard basis

`--hazard-basis` selects what the sale hazard actually responds to, which
decides what a recovery test proves.

- `realized` (default): the hazard responds to `rel_price_premium`, the same
  quantity the estimator regresses on. Recovery is then an exact test of
  **estimator code**.
- `latent`: the hazard responds to the seller's own aggressiveness `a_i`, while
  the estimator still only sees `rel_price_premium`. Since that variable is
  mostly unit quality, a naive fit is attenuated toward zero and only hedonic
  controls recover the planted β. This tests **specification**, not code.

Naive slope of `event_sold` on `rel_price_premium`, planted β = −1.6:

| profile | `realized` | `latent` |
|---|---|---|
| `rich` | −0.484 | −0.116 |
| `like_export` | −0.287 | **−0.002** |

The bottom-right cell is the one to design against: a real elasticity of −1.6
that a naive fit reports as zero.

`--building-quality-share` splits the fair-value noise into a component constant
within a tower and a component that varies unit by unit. It decides how much of
the quality contamination building fixed effects can absorb, and therefore how
far a controlled fit under `latent` can close on the planted β. Both profiles
default to 0.65.

## Demand model, measured

Cox PH recovery of a planted `true_beta_price`, `rich` profile, n = 6,000.

Under `realized`, the estimator regresses on the variable that generated the
hazard, so this is a test of estimator code and it passes cleanly:

| planted | recovered | 95% CI | error |
|---|---|---|---|
| −0.4 | −0.380 | [−0.545, −0.214] | 5.1% |
| −1.6 | −1.497 | [−1.671, −1.323] | 6.4% |
| −3.0 | −2.954 | [−3.144, −2.763] | 1.5% |

Under `latent`, the hazard responds to the seller's aggressiveness while the
estimator still only sees `rel_price_premium`. Every rung of the control ladder
recovers more of the planted −1.6, and none of them recovers all of it:

| controls | quality 65% building-level | quality 100% building-level |
|---|---|---|
| brief covariate list | −0.241 (15%) | −0.282 (18%) |
| + log area, + view | −0.339 (21%) | −0.417 (26%) |
| + building fixed effects | −0.680 (42%) | −1.185 (74%) |

The residual gap is not a bug to fix. Unobserved quality inside
`rel_price_premium` is classical measurement error in a regressor, which
attenuates the coefficient toward zero no matter how large the sample. **The
fitted `β_price` is a lower bound on the true elasticity in magnitude**, and
`run_diagnostics` says so in the identification report whenever a hedonic
control the export supports has been left out.

## The optimizer

A MILP over `y[i,j,k]` — unit `i` released in phase `j` at price level `k`. The
demand model is called once per (phase, level) before the solve, so the
objective CBC sees is linear: `R[i,j,k] = p_k · area_i · P(sale) · δ_j`, with
`δ_j = 1/(1+r)^(t_j/12)` on an **annual decimal** rate.

Two design choices are worth knowing before reading a plan.

**Monotonicity binds on the price-ladder index, not on dollars.** Index `k` is
the k-th step across a unit's *own* comps band, so the rule reads "no later
release is less aggressive relative to its comps than an earlier one". Binding
on raw $/sqft instead would let a small unit at the top of its band block a
large one at the bottom of its, purely because bigger units carry a higher
$/sqft — a size effect masquerading as a price cut. Comparable is
`unit_type × floor_bucket`. Ordering is enforced across *all* phase pairs, not
just consecutive ones: with consecutive links alone, a group that sits out a
phase resets its ladder and can come back at the bottom of the band.

**Comps are priced as of the latest quarter in the hedonic sample.** The surface
carries a time control and has never seen 2028, so the band is today's market
value rather than a forecast of the phase date. Where a developer believes comps
will move, that belief goes in `Comps.monthly_drift`, which defaults to zero.

Measured on the 60-unit `example_inventory.csv`, four phases, 15 price levels
(3,600 binaries), Cox fitted on 4,000 synthetic listings at `β_price = −1.53`:

| | value |
|---|---|
| Solve status | `optimal` in 3.4 s |
| Units released | 60 of 60 |
| Discounted expected revenue | \$76.98M |
| Construction gate | blocks 30 of 240 unit-phase cells |

The two acceptance tests that matter are in `tests/test_optimizer.py`. At
`β_price = −0.4` versus `−3.0` the mean recommended price ratio is below 0.95
and no unit prices higher under elastic demand — if that ratio were exactly 1.0,
the demand function would not be reaching the objective at all. At
`β_price = 0` every released unit lands on its ceiling, which is the *correct*
answer with no price response and is exactly what a broken elasticity estimate
would look like coming out of this system.

Monotonicity is a real constraint rather than a decorative one: removing it on
the same inventory raises the objective by about \$4k and produces eight
violations of the rule.

### Prices are checked against the evidence

A Cox linear predictor evaluates anywhere. Asked about a price 50% beyond
anything in the fitting sample it returns a probability with the same units and
the same decimal places as a real one, and nothing about the float says which it
is. So `CoxDemandModel` records the range of `rel_price_premium` it was fitted
over — measured on the rows that survived listwise deletion, since those are the
listings the coefficient is evidence about — and every solved plan is checked
against it:

```
PRICE SUPPORT [ok]
  All prices imply a rel_price_premium within [-0.383, +0.582],
  well inside the fitted range.
  chosen rel_price_premium spans +0.008 to +0.267
```

Prices beyond the observed min/max are reported as extrapolation and the message
leads the caveat list; prices inside the range but outside its 1st–99th
percentiles are reported as thin support. A model that reports no support at all
returns `checked: false` rather than passing silently — absence of the check has
to look different from a passed check.

## The smoke signal, measured

`corr(rel_price_premium, event_sold)` before any hedonic control:

| source | correlation |
|---|---|
| synthetic `rich` | −0.202 |
| synthetic `like_export` | −0.323 |
| synthetic `like_export`, `latent` basis | −0.003 |
| **real Miami export** | **−0.034** |

The real export behaves like the `latent` case, not the `realized` one — which
is what you would expect if the sale hazard responds to the seller's pricing
choice while the observable is dominated by unit quality. Recovering `β_price`
here depends on the hedonic controls, not on the raw correlation.

A naive Phase 3 fit on this data returns **β_price = −0.171** (SE 0.046, 95% CI
[−0.261, −0.081]); with hedonic controls, **−0.190** (SE 0.045, CI [−0.279,
−0.102]). Both are correctly signed and exclude zero, so the sign check passes —
but both sit well below the −0.5 magnitude the specification treats as the floor
of a usable elasticity. Read them as a lower bound, not an estimate, and see
`audit/AUDIT_REPORT.md` §9.4 for the three separate mechanisms attenuating them.

## Monte Carlo and sensitivity

Fixed-plan mode re-evaluates the solved release schedule under correlated
draws of `β_price ~ Normal(β̂, SE(β̂))` plus optional absorption, comps-drift,
competitor-inventory, and construction-delay shocks (`scenarios.py`). Under
proportional hazards the sale probability updates in closed form, so 10,000
draws finish in seconds. Each draw also samples Bernoulli sales — covenants
are about realized cash, not expected cash. The report includes P5–P95,
CVaR@5%, `P(plan > price-at-comps baseline)` on the same schedule, per-phase
breach probability with a suggested buffered floor, and a variance split
between parameter uncertainty and lumpy sales.

`sensitivity.py` adds a ±1σ tornado on expected revenue and LP-relaxed shadow
prices on the cash-flow floors (CBC duals require continuous `[0,1]`
variables). Optional `--resolve N` re-solves under N LHS scenarios to test
whether the *plan* itself is stable — marked slow in tests.

Only `β_price`'s dispersion is fitted. The other channels are user assumptions
and are labelled as such in every response, alongside `provenance` including
`is_calibrated_on_real_data`.

## Known gaps

- **The export is a sample of listings that went off market in a window, not a
  sample of listings that started in one.** Every terminal date in
  `PRICING_MODEL_export.csv` falls between 2025-10-14 and 2026-07-30 — 289 days —
  while list dates span 837, and no listing is still ACTIVE. A listing begun
  before that window appears only if it lasted into it, and a listing still
  unsold at the export date is absent entirely. The listings missing are
  disproportionately the slow, over-priced ones, which are exactly the
  observations that identify a price response, so `β_price` is attenuated toward
  zero by an amount this data cannot measure. `ingest_mls --inspect` now detects
  the pattern and warns. Fixing it needs an export drawn on **list date**,
  including active listings.
- **The export is sorted by status and capped at exactly 5,000 rows** against a
  search that reported "5000+". The file arrives in six contiguous status blocks
  ending with `Expired` (n=901), so the cap truncated the purest did-not-sell
  outcome — truncation on the outcome variable. `--inspect` warns. Re-pull
  sorted on ML# to fix.

- **Calibration gate:** do not fit on real MLS without a human decision — see
  `CALIBRATION_READY.md`. `POST /api/demand/fit` with `dataset=mls` requires
  `calibration_gate: true` and still leaves `is_calibrated_on_real_data` false
  unless config is flipped and `assert_calibrated` is set. The UI banner is
  non-dismissible while that flag is false.
- Revenue distribution chart sketches density from returned percentiles (the
  API does not stream the full draw sample). Exact P5/P50/P95/CVaR numbers in
  the summary are authoritative.
- **The objective treats units as independent, and the crowding correction does
  not fix that.** Ten near-identical two-bedrooms released into one phase are
  each scored against the whole buyer pool, so phase revenue is overstated.
  `crowding.py` scales the answer down afterwards by
  `1 − λ·(released/pool)`, but it is a post-solve scaling: the plan was *chosen*
  under independence, and correcting the revenue does not re-stage the phases
  that caused the crowding. Both `λ` and `buyer_pool` are user assumptions, not
  estimates. A correct treatment models sale probabilities jointly against a
  finite pool. Monte Carlo draws sales independently for the same reason.
- **The cash-flow floor binds on expected revenue.** A phase whose expected
  revenue exactly equals a loan covenant breaches it roughly half the time.
  `simulate_plan` reports the realized breach probability and, above ~5%, a
  buffered floor to re-solve against. The default basis is nominal dollars, not
  the discounted `R` of `PROJECT_BRIEF.md` §4.3 — a covenant is measured in the
  cash that arrives. Set `ConstraintSet(cash_flow_basis="discounted")` to match
  the brief exactly, and read `CF_min` as present-value dollars if you do.
- **Mortgage rates enter as an absorption (log-hazard) shift, not a fitted
  rate→hazard path.** There is no macro series joined to listings, so inventing
  a sensitivity would hardcode an estimated parameter. Config holds bounds;
  the user owns the shock size.
- **A unit the demand model cannot score gets no price at all.** Missing a
  covariate the model was fitted on means a NaN probability, which becomes an
  excluded unit with a stated reason rather than an imputed average. Same for a
  unit whose cost basis plus margin exceeds its comps ceiling: it is named in
  `excluded` with the arithmetic, not quietly pinned to a price the model says
  will not clear. Check `excluded_units` on every result.
- **A demand model that reports no fitted support disables the extrapolation
  check.** `check_extrapolation` reads `premium_support` off the model; only
  `CoxDemandModel` records one. Any other estimator degrades to
  `checked: false` with a caveat saying so — visible, but no longer verified.
  The support is also the *marginal* range of `rel_price_premium`, not the
  joint range with the other covariates, so a price that is ordinary on its own
  but unprecedented for a penthouse still passes.
- **`PULP_CBC_CMD` is deprecated in PuLP 3.x.** `solve.py` prefers `COIN_CMD`
  when a system `cbc` is on PATH and falls back to PuLP's bundled binary, which
  is what makes the repo work without a system install. The fallback emits a
  deprecation warning per solve until CBC is installed separately.
- **On export-shaped data the demand model returns `FAILED`, and that is the
  correct answer.** Fitting the `like_export` profile under the `latent` hazard
  basis — no view column, fair-value spread roughly six times the spread of
  sellers' pricing choices — puts `β_price` at approximately zero with a
  confidence interval covering it, sometimes on the positive side. Under
  `AGENTS.md` §3 that is an unconditional hard failure, and
  `test_export_like_data_cannot_identify_beta_and_says_so` pins the behaviour.
  Read it as the system working: this data, with this specification, cannot
  identify the elasticity. Do not respecify until the sign flips.
- **`gamma` in the floor premium is a normalization, not an estimate.** In
  `premium(floor) = α·ln(floor+1) + γ` only `α` is identified by a log-linear
  surface; `γ` is collinear with the intercept. It is set to
  `−α·ln(reference+1)` so the premium is zero at the reference floor, held in
  `config/miami.yaml` as `floor_premium_reference_floor`. Fitting `γ` off the
  floor partial residual returns exactly zero, since OLS residuals are
  orthogonal to both the constant and `log_floor`.
- **`inventory_competition` counts entries, not standing inventory.** It was
  originally the number of listings whose live interval overlapped the month,
  which made it a function of other listings' durations: a slow month looked
  crowded *because* its listings were slow. On synthetic data with no planted
  inventory effect that produced a positive coefficient in six of eight seeds
  and a significant one in one of them, hard-failing an otherwise correct model.
  It now counts other listings entering the same (submarket, month), which is
  fixed before any duration is realized. What that gives up is unsold standing
  inventory from earlier months — a real competitive force. Recovering it
  without the endogeneity needs an as-of-month-start active count built from
  listing dates alone.
- **The logistic cross-check does not fit on the real export.** It fails with a
  singular information matrix where the Cox model on the same covariates
  succeeds, so the AUC, the calibration table, and the second sign check are all
  unavailable there. It does not reproduce on the `like_export` profile, so the
  cause is real-data sparsity rather than the specification. The design builder
  now drops linearly dependent columns before fitting, which may resolve it;
  confirm at the calibration gate. The absence is reported as a WARN check
  rather than left in a log line.
- **Repeat listings are an unused identification strategy.** The same unit
  relisted at a different price gives within-unit variation that is pure pricing
  choice with quality held exactly fixed — the one source here that is not
  contaminated. The real export has 232 relisted `unit_key`s. That is thin, but
  it would let the attenuation factor be *measured* rather than only warned
  about, turning `β_price` from a lower bound into a corrected estimate. Not
  built; the strongest candidate for the next modeling increment.
- **Model bundles are pickled.** `registry.py` stores fitted lifelines and
  statsmodels objects with `pickle`, which does not survive library upgrades.
  `metadata.json` sits beside each bundle carrying the coefficients, diagnostics,
  and library versions as plain JSON, so a stale pickle costs a refit rather
  than the numbers.
- The Cox concordance on synthetic data sits near 0.59, below the 0.60 flag, and
  the diagnostics warn accordingly. Most of the variation in time-to-sale is
  Weibull noise by construction, so weak discrimination is expected and does not
  imply a biased coefficient. On real data, treat it as a genuine caution.
- Synthetic identification under `realized` is verified by construction: the
  hazard is driven by the same `rel_price_premium` the estimator regresses on.
  Use `--hazard-basis latent` for the specification test and
  `confound_strength > 0` to break identification deliberately.
- **No view field in the Miami export.** `view_description` is absent, so the
  view premium in the hedonic surface cannot be fitted. It is dropped from the
  specification and belongs on the broker data-request list — for a Miami
  luxury tower, direct-ocean versus obstructed is plausibly the largest single
  amenity premium, and omitting it leaves that variation in the residual.
  `available_covariates` drops it automatically, treating an all-null column as
  absent: normalization materialises every canonical field whether the export
  had it or not, so keeping it would delete the whole sample through listwise
  deletion. On synthetic data adding view roughly doubles the recovered
  `β_price` under the `latent` basis, which is the size of what is being lost.
- **No `sale_type` or `property_type` either**, so the distressed and condo
  filters are no-ops on the real export. `clean_mls` drops zero rows for
  distressed there. Any foreclosure or short sale in the pull is currently
  sitting in the hedonic surface. Also on the data-request list.
- **`rel_price_premium` is dominated by unit quality, not pricing choice.** In
  the real export its IQR *within* a (submarket, month) cell is 0.617 — cells
  hold studios and penthouses alike, so the median controls for location and
  time but not for the unit. In `like_export`, under 2% of its variance is the
  seller's pricing decision. Phase 3 must residualize quality hedonically
  before the coefficient on it can be read as an elasticity. `features.py`
  warns on both tails; `PROJECT_BRIEF.md` §3d only specifies the narrow one.
- **13.2% of the real export cannot enter the demand fit.** 169 rows lack a
  submarket and 117 more sit in cells too thin for even a quarterly median, so
  286 (5.8%) carry a null `rel_price_premium` rather than a borrowed one; the
  rest of the loss is other covariates. Full-covariate usable rows: 4,298 of
  4,954. This was 27.4% before the audit, when 661 rows — every PENDING and
  every WITHDRAWN listing — were excluded because the export leaves `List Date`
  blank for exactly those statuses. `normalize.reconstruct_list_date` now
  derives it as terminal date minus days on market and marks it
  `list_date_source="derived_from_dom"`. See `audit/AUDIT_REPORT.md` F03.
- **Penthouses carry no numeric floor.** `parse_floor` returns
  `floor_source = "parsed"` with `floor = None` for a PH unit, so `log_floor` is
  null and the row drops out of the demand fit. Only 1 row in the current
  export, because `Unit Floor Location` is reported almost everywhere — but an
  export keyed on unit numbers instead would silently drop a developer's
  highest-value units. The count is in the feature report.
- **238 real rows have an implausible HOA** (outside $0.05–$5.00/sqft/month)
  and are nulled with `hoa_per_sqft_source = "implausible"`. With no
  `hoa_frequency` column there is no evidence for what those figures mean, so
  they are not rescaled. Add `hoa_frequency` to the broker data request.
- `rel_price_premium` / IQR identification check lands in Phase 2; ingest `--inspect`
  notes that deferral explicitly.
- Bucharest: config stub only; no data pipeline this pass.
- Real MLS calibration is gated — see `PROJECT_BRIEF.md` §5.
- Optional LLM column-mapping via `OPEN_ROUTER` is not wired; alias table is primary.
- **HOA frequency is unverifiable in the Miami export.** It has no
  `AssociationFeeFrequency` column, so `hoa_monthly` stores the reported figure
  as-is; some values are plainly annual. `hoa_per_sqft` becomes a Phase 2
  covariate, so confirm the units before relying on it.
- **Submarket ZIP coverage.** Major export ZIPs (Key Biscayne, Fisher Island,
  Coral Gables, Doral, North Miami, Pinecrest/Dadeland) are mapped in
  `config/miami.yaml`. Thin leftover ZIPs may still appear as unmapped rows;
  `--inspect` warns only when the unmapped rate exceeds 5%.
- `map_columns`, `clean_mls`, and `compute_duration_days` still return bare tuples
  rather than dataclasses, against the `AGENTS.md` convention. `normalize_mls` and
  `ingest_mls` return dataclasses.
- `normalize_mls` makes two `.iterrows()` passes (~1.5s per 5k rows). Fine now;
  revisit if Phase 3 refits make it a bottleneck.
