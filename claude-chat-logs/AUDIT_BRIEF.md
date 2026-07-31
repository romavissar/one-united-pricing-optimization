# AUDIT_BRIEF.md — Full-System Audit of the Dynamic Pricing & Unit Release Optimizer

**Read this entire file before touching any code.**

You are being asked to perform an adversarial, exhaustive, end-to-end audit of a system that has
already been built. You are not here to add features. You are here to find **every** reason the
system's outputs might be wrong, misleading, insecure, or indefensible — and then to **fix
everything that can be fixed** and document precisely what cannot.

Assume the code runs. Assume the tests pass. Neither fact means the model is correct. A pricing
optimizer that silently mis-estimates one coefficient will still return a beautifully formatted
release plan and a confident revenue number. That is the failure mode this audit exists to catch.

### Two standing instructions that override everything else in this document

**1. This checklist is a floor, not a ceiling.**

Every item below was written by someone reasoning about the system from its specification
documents — *without having read the code you are about to read*. That means the list is
necessarily incomplete. It captures the failure modes that were foreseeable in the abstract. It
cannot capture the ones that live in the specific implementation choices Cursor made, the
libraries' actual behavior, the interactions between modules, or the shape of the real data.

Treat every section as a set of worked examples of *how to think*, not as a list to tick off.
If you complete every checkbox and find nothing else, you have almost certainly not audited
hard enough. The most damaging defect in this repository is, statistically, one that nobody
anticipated well enough to write down — which means it is not in this document. §1.3 tells you
how to hunt for those. Read it as the most important section here.

**2. Fix everything you find.**

This is not a report-and-await-instructions engagement. Where a defect has a correct fix, make it:
correct the math, close the vulnerability, repair the leak, tighten the constraint, add the guard,
write the test. Do not leave a known-wrong line of code in place because it fell outside the scope
of a checklist item. Do not defer a fix because it touches a module you weren't explicitly pointed
at. If it is wrong and it can be made right, make it right.

The only things you leave unfixed are the ones that **cannot** be fixed with the data and
information available — a limitation of the dataset, an assumption the model cannot escape, a
validation that requires data nobody has. Those get documented in exact, publishable wording
(§9.4), because the honest statement of a limitation is itself a deliverable.

**The one boundary on "fix everything":** fixing means making the code do the correct thing. It
never means adjusting the model until a number looks better. If `β_price` comes back implausible,
the fix is to find and repair the reason — not to add a prior, clamp the coefficient, drop
inconvenient rows, or switch estimators until the sign flips. Repairing a cause is a fix.
Suppressing a symptom is fabrication, and on this project it is the single worst outcome
available. Where you genuinely cannot tell which one you are doing, stop and say so in the report.

---

## 0. Orientation

### 0.1 What the system is

A quantitative decision tool for luxury residential real estate developers. Input: a project's unit
inventory (floor, area, type, features). Output: a recommended **price per unit** and a **phased
release schedule** that maximizes expected discounted revenue, plus a risk distribution around that
recommendation.

Six layers, built in dependency order:

| Layer | Phase | What it does |
|---|---|---|
| 1 | 1 | Data ingestion, cleaning, feature engineering |
| 2 | 2 | Demand model — `D(price, features, macro) → P(sale within T)` |
| 3 | 3 | MILP optimizer — chooses price × phase per unit |
| 4 | 4 | Monte Carlo — revenue distribution and risk metrics |
| 5 | 5 | Backtest — would this have beaten what actually happened? |
| 6 | 6 | React/Plotly dashboard |
| — | 7 | Tests, docs, deployment |
| — | 8 | SaaS: auth, multi-tenancy, Stripe billing (may be partially built) |

Two markets: **Miami** (USD, price per sqft — this is the calibrated one) and **Bucharest,
Romania** (EUR, price per sqm — config stub, not calibrated on real data).

### 0.2 Why it exists

Two purposes, and the audit must serve both:

1. **A real tool.** The primary intended client is One United Properties (BVB: ONE), Romania's
   largest listed residential developer. Real people may make real pricing decisions with this.
2. **A case study for university applications.** Results will be written up and presented. A
   quantified claim like "8% revenue uplift" will be read by people who may not have the technical
   background to challenge it — which makes it *more* important, not less, that the number is
   honest and that its limitations are stated in the artifact itself.

Both purposes fail the same way: a confident, wrong number. Optimize the audit for catching that.

### 0.3 How it was built (relevant to where bugs hide)

The repo was built by **Cursor in agent mode**, one phase at a time, from a set of hand-written
specification documents. Characteristics of this build process that shape where you should look:

- **Each phase was implemented and reviewed in relative isolation.** Cross-phase contract drift is
  the most likely class of defect: Phase 2 defines a demand interface, Phase 3 consumes it, Phase 4
  consumes it differently. Check the seams, not the interiors.
- **The agent asked design questions mid-build and the answers were given conversationally.** Those
  decisions may be implemented in code but absent from the spec documents, or present in the docs
  but not the code. Reconcile both directions.
- **Acceptance criteria were written before the data was seen.** Several are shape checks
  ("returns values in [0,1]") that a badly wrong model passes trivially. Treat the phase exit
  checklists as a floor, not a definition of correctness.
- **A synthetic data generator (Phase 1.5) exists with two profiles**, `rich` and `like_export`.
  Tests may be passing against `rich` — clean, ideal, generous — while the real data resembles
  `like_export`. Verify which profile each test actually uses.

### 0.4 Documents you must read first, in this order

| File | Why |
|---|---|
| `AGENTS.md` | Hard rules and conventions. The rule *"never hardcode estimated parameters"* is the single most important line in the repo — audit against it explicitly. |
| `dynamic_pricing_optimization_guide.md` | The full technical guide (§1–16): math, parameters, market configs, output schemas. This is the specification of intent. |
| `MLS_SCHEMA.md` | The data contract: canonical field names, column aliases, status normalization, floor parsing, duration computation, validation report spec. |
| `project_structure.md` | Directory tree, API endpoint table, dependency lists, run commands. |
| `phases/phase-0` … `phase-8` | The build plan with per-mini-phase acceptance criteria. |
| `updates.md` | Six verified corrections to the guide (floor premium coefficients, Miami floor premium caveat, independence assumption + crowding adjustment, seasonal/view priors, chance-constrained cash flow, discount factor units). **Where `updates.md` and the guide disagree, `updates.md` wins.** Verify the code followed the correction, not the original. |
| `.env.example` | The configuration surface. |

### 0.5 The data

**`PRICING_MODEL_export.csv`** — the real calibration dataset. 5,000 Miami-Dade condo records
exported from CoreLogic Matrix (MIAMI Association of Realtors MLS), listings live between
2024-07-01 and 2026-07-30, minimum $400,000, Miami-Dade County, condo only, REO and short sales
excluded.

Status distribution:

| Status | Count |
|---|---|
| Closed | 1,896 |
| Cancelled | 1,542 |
| Expired | 901 |
| Pending | 290 |
| Withdrawn | 285 |
| Temp Off Market | 86 |

Fill rates that matter: Original List Price 100%, Sale Price 37.9%, Unit Floor Location 100%,
SqFt Liv Area 98.9%, **List Date 86.8%**, Off Market Date ~100%.

Known gaps, already decided and not to be relitigated: no view field (dropped from the hedonic
spec, `Waterfront Y/N` used instead), no sale_type / property_type / city (filtered at search time
or superseded by ZIP), no tax field. **The export is capped at 5,000 rows and the underlying search
returned "5000+" — this is a truncated, not complete, sample.** Whether that truncation is random
is an open question you must investigate (see §2.6).

### 0.6 The one thing this system exists to estimate

Everything downstream depends on a single quantity: **`β_price`**, the coefficient on
`rel_price_premium` in the demand model, where

```
rel_price_premium = (list_price_per_sqft / submarket_month_median_ppsf) - 1
```

If `β_price` is wrong, the optimizer produces confident nonsense, the Monte Carlo produces a
confident distribution around nonsense, and the backtest produces a confident uplift number
attributable to nonsense. **Audit this coefficient harder than anything else in the repo.**

---

## 1. How to run this audit

### 1.1 Method

Work in this order. Do not skip ahead — later sections assume findings from earlier ones.

1. **Read** all documents in §0.4. Build a mental model of intent before reading code.
2. **Static pass** — read the code end to end without running it. Note every discrepancy from
   spec, every silent failure path, every place a wrong answer would not raise.
3. **Reconciliation pass** — for each of the two frozen contracts (unit inventory schema, demand
   interface `predict(units_df, prices, macro) -> probabilities`), trace every consumer and confirm
   they agree on shape, units, ordering, and semantics.
4. **Numerical pass** — run the checks in §2–§5. Write the diagnostic scripts you need; put them
   under `audit/` so they are reproducible and don't pollute `src/`.
5. **Adversarial pass** — §6 and §7. Try to break it.
6. **Open-ended discovery pass** — §1.3. This is the pass that finds what this document didn't
   anticipate. Budget more time for it than for any other single pass, and do not treat it as
   optional cleanup at the end.
7. **Claims pass** — §8. Check that what the system *says about itself* is true.
8. **Fix pass** — repair everything repairable, with a regression test per fix.
9. **Re-verification pass** — after fixing, re-run every numerical check in §2–§5 and the four
   headline numbers in §9.2. Fixes to a data pipeline change coefficients; fixes to coefficients
   change plans; fixes to plans change risk metrics. A fix applied without re-verification
   downstream is not a completed fix.
10. **Report** — §9.

### 1.2 Ground rules

- **Fix it, then log it.** Every defect with a correct fix gets fixed. Every fix gets a finding ID,
  a diff, a regression test, and an entry in the report. Fixing without recording is as bad as
  recording without fixing.
- **Never suppress a symptom.** Repairing the cause of a bad coefficient is a fix. Clamping the
  coefficient, adding a prior, dropping rows, or switching estimators to change the number is
  fabrication. See the boundary stated at the top of this document.
- **Reproduce before you conclude.** If you assert a bug, produce the minimal script that
  demonstrates it, and keep that script under `audit/` as the regression artifact.
- **Fail loudly is the house style.** Per `AGENTS.md`, diagnostics that detect a problem should
  raise or emit a prominent warning, not log at DEBUG and continue. Where you find a silent
  failure path, make it loud — that is itself a fix.
- **Distinguish "wrong" from "unvalidated."** Both matter, but they carry different fixes: one gets
  a code change, the other gets documented wording (§9.4).
- **Scope creep on correctness is not scope creep.** If tracing a bug leads you into a module the
  checklist never mentions, follow it. The boundaries in this document describe where to *start*
  looking, not where to stop.
- **When two documents disagree, the more recent correction wins** (`updates.md` over the guide),
  and when a document disagrees with the code, neither automatically wins — determine which is
  right on the merits and fix the other.

### 1.3 The open-ended pass: finding what this document didn't anticipate

The checklist sections are worked examples. This section is the actual method. Apply it to every
layer, including ones where the checklist already found nothing — a clean checklist result is
weak evidence of a clean module.

**Read for intent, then read for behavior.** For each module: first read the spec and form a
precise expectation of what the code should compute. Then read the code and derive what it
*actually* computes, symbolically, without assuming the names mean what they say. The gap between
those two derivations is where defects live. A function named `compute_relative_premium` that
divides by the wrong median is invisible to any checklist that trusts the name.

**Ask the six questions of every computation you encounter:**

1. *What are the units, and does dimensional analysis balance?* Currency, area, time, rate, and
   probability all flow through this system and all are silently multiplicable.
2. *What is the domain of validity, and does anything call this outside it?* Extrapolation past
   the support of the data is this system's defining risk and it recurs in places the checklist
   doesn't list.
3. *What happens at the boundary?* Zero, one, empty, all-identical, all-missing, negative, maximum,
   duplicate keys, single category, single time period.
4. *If this were wrong, how would I find out?* If the answer is "I wouldn't," that is a finding in
   itself — a missing assertion — regardless of whether the code is currently correct.
5. *What is assumed that is nowhere stated?* Independence, stationarity, exchangeability, linearity,
   completeness of the sample, alignment of two indexes, that a merge key is unique. Every one of
   these is a claim, and untested claims in a pricing model become false claims in a report.
6. *Who consumes this, and do they agree with it?* Shape, ordering, units, semantics, null
   convention, and time base must match at every interface.

**Techniques that surface defects a checklist cannot:**

- **Property-based testing.** For any function with a mathematical property — monotonicity in
  price, probabilities in [0,1], revenue non-negative, percentiles ordered, the optimum being at
  least as good as any feasible point — assert the property over randomized inputs rather than
  over one example. Reach for `hypothesis` where it fits. This finds edge cases nobody enumerates.
- **Metamorphic testing.** Where you have no ground truth, you still have relations that must
  hold: doubling every unit's area should scale revenue proportionally; adding a strictly dominated
  price level must not change the optimum; relabeling unit IDs must not change the plan; shifting
  all prices and the submarket median by the same factor must leave `rel_price_premium` unchanged.
  Violations of these are unambiguous bugs and they are found by no checklist.
- **Differential testing.** Reimplement a critical computation independently and crudely — a slow,
  obvious, loop-based version — and compare against the optimized one on real inputs. Do this at
  minimum for the demand tensor, the objective value, and the survival-probability conversion.
- **Fuzz the inputs.** Malformed CSVs, unicode in unit IDs, extreme values, all-null columns,
  columns in a different order, dates in a different format, a single unit, ten thousand units.
- **Adversarial reading.** For each module, spend five minutes trying to construct the input on
  which it silently returns a wrong answer. If you succeed, that is a finding even if it seems
  unlikely to occur.
- **Follow the data, not the call graph.** Trace one real unit from the raw CSV row all the way to
  its recommended price on the dashboard, printing its state at every transformation. Do this for
  a normal unit, a penthouse, a unit with a missing list date, and the most expensive unit in the
  file. End-to-end traces of single records expose misalignments that aggregate statistics hide.
- **Check the silences.** Every warning that is filtered, every exception that is caught, every
  `fillna`, every default parameter, every `if not X: return` — each one is a decision made without
  announcing itself. Enumerate them all and justify each.
- **Re-derive the math on paper.** For the objective function, the Cox survival conversion, the
  discount factor, the big-M bound, and the CVaR definition, write out the mathematics
  independently and compare to the implementation. Do not read the implementation first; you will
  anchor on it.

**Categories of defect explicitly worth hunting beyond the checklist:** numerical (catastrophic
cancellation, overflow in exponentials of the linear predictor, float comparison, accumulation
order); concurrency (shared mutable state under async FastAPI, non-thread-safe model objects,
per-worker RNG); statistical (any assumption listed in question 5 above); economic (does the
recommendation make sense to someone who knows real estate? a plan that releases every penthouse
first, or prices two identical units differently, is a signal); presentational (a correct number
with a wrong label is a wrong number); and operational (what happens on a re-run, a partial
failure, a concurrent request, a retrain mid-solve).

**If a section of this document is silent about a layer, that is not permission to skip it.** The
frontend (Phase 6) gets one paragraph in §8 and no section of its own; audit it anyway — chart
axes, aggregation in the UI, rounding, currency formatting, and tooltip claims are all places a
correct backend becomes a wrong product.

---

## 2. Statistical & identification audit (highest severity)

This is where a wrong system looks most like a right one.

### 2.1 Confirm what variable actually carries the price effect

- [ ] Trace the exact feature matrix passed to the fitted estimator. Not the docstring, not the
      config — the array. Print the column names at fit time.
- [ ] Confirm the price variable is `rel_price_premium` (or an equivalent normalized measure) and
      **not** raw `list_price_per_sqft`, `list_price`, or `log(price)`.
- [ ] **Why this matters:** absolute price per sqft is dominated by location and quality. A $2,000/sqft
      Fisher Island unit sells fine; a $700/sqft unit in a weak submarket may not. Regressing on
      absolute price recovers a coefficient near zero or *positive* — and a positive `β_price` makes
      the optimizer price every unit at the ceiling, which will look like a spectacular revenue uplift
      and be entirely fictitious.
- [ ] Confirm both `list_price_per_sqft` and the submarket median use the **same** area field
      (`SqFt Liv Area` vs `Main Living Area` — these differ) and the same price field
      (`Original List Price`, not `List Price` or `Current Price`, which reflect post-listing
      reductions and are endogenous to failure to sell).

### 2.2 Audit the denominator: submarket × month medians

- [ ] Report the full distribution of cell sizes for (submarket × month). How many rows fall in cells
      with n < 8? What fraction fell back to (submarket × quarter)? What fraction had no valid cell
      at all, and what happened to them?
- [ ] **If most cells are thin, the median is noise and `rel_price_premium` is noise around noise.**
      The regression will then attenuate `β_price` toward zero through classical
      errors-in-variables. Quantify this: estimate the sampling variance of the cell median and
      report the implied attenuation factor.
- [ ] Check whether the focal unit is included in its own cell median. It should be excluded
      (leave-one-out), otherwise there is a mechanical negative correlation between a unit's premium
      and the median in small cells, biasing `β_price`.
- [ ] Check the submarket definition. ZIP code is the stated choice. Verify ZIPs are not being
      merged, truncated (ZIP+4), or read as integers (leading-zero loss is not a Florida problem but
      check anyway), and that low-count ZIPs are handled deliberately.

### 2.3 Audit the survival panel construction

- [ ] **The 13.2% of rows with missing `List Date`.** Where did they go? If dropped, test whether the
      drop is random with respect to status, price, and submarket. If Closed rows are more likely to
      have a list date than Cancelled rows (or vice versa), the panel is selected on the outcome and
      every coefficient is biased. This is a plausible and severe failure.
- [ ] **Duration construction per status.** Verify:
      - Closed → `Closing Date` (or `Pending Date` if the spec chose contract date) − `List Date`
      - Pending → `Pending Date` − `List Date` (Pending rows have no closing date)
      - Cancelled / Expired / Withdrawn / Temp Off Market → `Off Market Date` (or the status-specific
        date) − `List Date`
      - Check for negative durations, zero durations, and durations exceeding the observation window.
        Report counts. Any of these silently poison a Cox fit.
- [ ] **Event indicator.** Confirm SOLD and PENDING → `event=1`; CANCELED, EXPIRED, WITHDRAWN,
      TEMP OFF MARKET → `event=0`. Then challenge the Pending decision: pending durations are
      systematically shorter than closed durations because the clock stops earlier. Mixing them
      creates a mismatch in what the "event" is. Quantify the effect by refitting with Pending
      excluded and reporting the change in `β_price`. If the coefficient moves materially, this must
      be documented.
- [ ] **Censoring window.** The data covers 2024-07-01 to 2026-07-30. Listings near the right edge
      have artificially short follow-up. Check that the panel handles administrative censoring
      correctly and doesn't treat "still active at export date" as an event or as a completed spell.
- [ ] **Left truncation / relists.** A unit that was listed, expired, and relisted appears as two
      rows. Check for duplicate `Address Line` + `Complex Name` combinations and decide whether they
      are independent observations. Treating relists as independent overstates sample size and
      understates standard errors.

### 2.4 Audit the Cox model itself

- [ ] **Baseline hazard centering.** `lifelines` centers covariates internally. If any code manually
      computes `S(t|x) = S₀(t)^exp(βᵀx)` using raw (uncentered) `x` against `baseline_survival_`, the
      resulting probabilities are systematically wrong while remaining in [0,1] and monotone — it
      passes every acceptance criterion in the Phase 2 spec. Verify the conversion against
      `predict_survival_function` on a sample of rows; assert agreement to numerical tolerance.
- [ ] **Extrapolation beyond observed follow-up.** The baseline hazard is only identified over the
      range of observed durations. If `T` (the target window) exceeds the maximum observed duration,
      or if the optimizer's phase horizon extends beyond it, the model is extrapolating a
      non-parametric baseline. Check what the code does past the last event time — many
      implementations return the last value, silently flattening the hazard to zero.
- [ ] **Proportional hazards assumption.** Run `check_assumptions` / Schoenfeld residual tests.
      If `rel_price_premium` violates PH — which is likely, since price effects often intensify over
      time on market — the single `β_price` is a time-averaged artifact. Report the test; if it
      fails, this belongs in the limitations section of every downstream artifact.
- [ ] **Collinearity.** Compute the condition number and VIFs of the design matrix. `floor`,
      `total_floors`, `unit_floor_location`, and `year_built` are plausibly collinear; so are
      `sqft`, `beds`, and `baths`. Check for convergence warnings suppressed anywhere in the code.
- [ ] **Coefficient sign and magnitude gate.** Report `β_price`, its standard error, and its 95% CI.
      The gate: **negative, CI excluding zero, magnitude roughly in [−0.5, −3.0]**. If it is outside
      that band, do not adjust it — diagnose it. A `β_price` of −12 means the price variable is
      capturing something else; a `β_price` of −0.05 means the optimizer will corner-solution to the
      price ceiling.
- [ ] Run the same sign checks the spec requires on the other coefficients (`β_floor > 0`,
      `β_mortgage_rate < 0`, `β_competitor_inventory < 0`) and confirm the checker actually fails
      loudly rather than warning into a log nobody reads.

### 2.5 Audit for leakage

- [ ] **`DOM` / `CDOM` as features.** These are functions of the outcome. If they appear anywhere in
      the feature matrix, that is a fatal leak — and it will produce an impressive AUC.
- [ ] **`List Price` / `Current Price` / `Sale Price` as features.** `Current Price` reflects
      reductions made *because* the unit wasn't selling. `Sale Price` is the outcome. Only
      `Original List Price` is legitimate as a predictor.
- [ ] **`Status Change Date`, `Last Status`, `Off Market Date`** — all post-outcome.
- [ ] **`inventory_competition`.** The stated definition counts listings whose live interval
      `[list_date, list_date + duration]` overlaps the focal month. That uses each competitor's
      *realized* duration, which is not knowable at the focal unit's listing date. This is a
      defensible modeling choice (it measures true competing supply) but it is *not* a
      point-in-time-available feature. Confirm it is documented as such, and check whether the
      backtest (Phase 5), which has an explicit no-leakage mandate, uses this feature — if so, the
      backtest leakage guard is being violated by a feature it doesn't know to check.
- [ ] **Train/test split.** Verify the split is time-based or at minimum that no unit appears on both
      sides. Verify any scaler/imputer is fit on train only. Verify `random_state` is fixed.
- [ ] Confirm the leakage assertion in Phase 5.1 ("assert every feature's timestamp ≤ decision date")
      actually inspects derived features, not just raw columns.

### 2.6 Audit sample selection and external validity

- [ ] **The 5,000-row cap.** Determine Matrix's sort order for the truncated export (by ML# ? by list
      date ?). If the cap truncated on any variable correlated with price or outcome, the sample is
      selected. Test by comparing the status distribution and price distribution of the first vs last
      1,000 rows as exported. Document the finding either way.
- [ ] **The $400,000 floor.** Every model coefficient is conditional on the upper segment of the
      market. That is intentional (luxury), but confirm it's stated in the model card.
- [ ] **The big one — resale vs primary.** This dataset is *resale condo listings by individual
      sellers.* The optimizer's application is *new-construction primary sales by a developer,
      released in phases, often pre-completion.* These are different markets with different buyers,
      different financing, different price discovery, and different elasticities. Verify that this
      gap is documented prominently — in the model card, the API response provenance, the report, and
      the dashboard. **This is the single most important limitation of the entire system and it must
      not be buried.**
- [ ] **Romania.** Confirm that a Miami-estimated `β_price` is not being loaded, defaulted, or
      copied into the Romania config path. If a Romania run is possible at all, verify it either
      refuses to run or returns results flagged `IS_CALIBRATED_ON_REAL_DATA=false` with that flag
      propagated into every response and every rendered chart. A Romanian developer receiving a
      recommendation derived from Miami resale elasticity, unflagged, is the worst realistic outcome
      of this project.

---

## 3. Optimization & mathematical audit

### 3.1 Objective function correctness

- [ ] Verify the implemented objective matches the spec:
      `Σ_i Σ_j Σ_k y[i,j,k] · p_k · A_i · D(p_k, X_i, M_j) · δ_j`
- [ ] **Unit consistency.** `p_k` is per-area; `A_i` is area. Confirm the product is a total price and
      not a per-area price multiplied twice or not at all. Confirm sqm vs sqft and EUR vs USD are
      consistent through the entire chain, including the discount factor and any cash-flow constraint.
      Add an assertion on the order of magnitude of total revenue against the inventory's notional
      GDV — an off-by-10.76 (sqm↔sqft) error is easy to make and hard to see.
- [ ] **Discount factor units.** `updates.md` contains an explicit correction here. Verify δ is
      computed on the right time base (monthly vs annual) and that phase index → elapsed time uses
      the configured phase length. A rate applied per-phase-as-if-per-year is a silent 10× error in
      the timing incentive.
- [ ] **Timing mismatch.** `D` is a probability of sale *within horizon T*. δ_j discounts to the
      *phase release date*. Expected revenue from a unit released at phase j is therefore being
      discounted as if it arrives at release, when it actually arrives distributed over [release,
      release+T]. Check whether the code corrects for this. If not, the model systematically
      under-penalizes late sales and the phase-sequencing recommendation is biased.

### 3.2 The unsold-unit problem

- [ ] A unit released in phase j with `D = 0.7` contributes `0.7 · p · A` to the objective. **What
      happens to the other 30%?** In reality the unit remains in inventory and can be sold later. In
      the MILP, C1 ("each unit released at most once") means it simply vanishes.
- [ ] Determine whether the formulation models carry-over. If it does not, the model:
      - understates total revenue,
      - is biased toward early release (a late release has fewer remaining phases in which its
        probability mass is *also* not counted, so this partially cancels — work out the direction
        empirically, don't assume),
      - and makes "expected revenue" not comparable to the actual GDV of a sold-out project, which
        breaks the backtest comparison in Phase 5.
- [ ] This may be an accepted simplification. If so, it must be named, its direction of bias
      established, and its magnitude bounded. Compute the total unmodeled probability mass
      (`Σ_i (1 − D_i)` over the chosen plan) and report it as a headline diagnostic. If it is large,
      every revenue number in the system is a partial sum being presented as a total.

### 3.3 The cross-elasticity contradiction

- [ ] Phase 2.5 builds `predict_portfolio(units, prices, macro)` with substitution effects, because
      units within a project compete.
- [ ] Phase 3.2 pre-computes `R[i,j,k]` **per unit, over a grid**, which is what makes the objective
      linear — and which structurally *cannot* represent dependence on other units' prices.
- [ ] **These two specifications are mathematically incompatible.** Determine what the code actually
      does. Likely outcomes: (a) `predict_portfolio` exists but the optimizer calls the per-unit
      `predict`, making 2.5 dead code that nonetheless appears in the write-up as a capability; or
      (b) the optimizer calls `predict_portfolio` with some fixed reference price vector, making the
      cross-effects a constant offset rather than a genuine interaction.
- [ ] Whichever it is, report it plainly. The independence assumption is defensible for a first
      version (and `updates.md` acknowledges it with an optional crowding correction, λ=0.3) — but
      the system must not claim to model substitution while optimizing as if units are independent.
- [ ] If a crowding correction is applied, check it is applied consistently in the optimizer, the
      simulator, *and* the baseline. Applying it to one and not the other manufactures uplift.

### 3.4 Corner solutions — the optimizer exploiting model error

- [ ] **Report the fraction of units priced at the ladder ceiling and at the floor in the optimal
      solution.** This is the most diagnostic single number about whether the optimizer is doing
      economics or exploiting extrapolation.
- [ ] Expected revenue `p · D(p)` has an interior maximum only if demand falls fast enough. With a
      weak `β_price`, revenue is monotonically increasing in price and **every unit goes to the
      ceiling.** The reported "uplift vs baseline" is then just "we raised prices," which the
      demand model is not credible enough to justify.
- [ ] Check the price ladder bounds against the **support of the estimated data**. If
      `p_ceiling` implies a `rel_price_premium` of +0.8 but the training data only contains premiums
      in [−0.3, +0.4], the optimizer is evaluating the demand model far outside its support, where a
      Cox model's linear predictor extrapolates without limit. **Clamp the ladder to the empirical
      support of `rel_price_premium` (e.g. the 5th–95th percentile) or, at minimum, flag and report
      every grid point outside it.**
- [ ] Verify `p_floor` / `p_ceiling` come from config, not hardcoded, per `AGENTS.md`.

### 3.5 MILP formulation mechanics

- [ ] **Big-M in C2 (monotone prices).** Check the M value. Too large → numerical instability and
      solutions that satisfy the LP relaxation but violate intent at integer tolerance. Too small →
      silently cutting off the optimum. Derive the tightest valid M from the ladder bounds and use
      that. Test the constraint in isolation with a case where the unconstrained optimum violates
      monotonicity, and assert the constraint binds.
- [ ] **PuLP variable naming.** PuLP silently mangles variable names containing spaces or special
      characters, and can produce name *collisions* if two variables normalize to the same string.
      If unit IDs come from user-uploaded CSVs (they do), this is a live risk. Assert unique
      sanitized names at model build time.
- [ ] **Integer tolerance.** Check the extracted solution for fractional binaries. CBC returns values
      like 0.9999999997; if extraction uses `== 1` it will silently drop assignments, and if it uses
      `> 0` it may accept two price levels for one unit. Use a tolerance and assert
      `Σ_j Σ_k y[i,j,k] ≤ 1` on the *extracted* solution.
- [ ] **Independent solution verification.** Phase 3.4 requires a checker independent of the solver.
      Confirm it exists, that it recomputes the objective from the extracted plan rather than reading
      `pulp.value(prob.objective)`, and that it verifies every constraint including the ones that
      were toggled off. A checker that shares code with the formulation verifies nothing.
- [ ] **Solver status handling.** Confirm `Infeasible`, `Unbounded`, `Not Solved`, and
      `Undefined` are each handled distinctly and that the code never proceeds to extract a solution
      from a non-optimal status. Test with a deliberately infeasible instance (e.g., `CF_min`
      exceeding total inventory value) and confirm a diagnostic, not a crash and not a silent plan.
- [ ] **Unbounded / degenerate cases.** Zero units, one unit, one phase, K=1, all units identical.
- [ ] **C3 cash flow.** `updates.md` specifies a chance-constrained treatment. Verify what is
      implemented. An expected-value cash-flow constraint (`E[revenue] ≥ CF_min`) is satisfied ~50% of
      the time in realization. If the expected-value form is used, the breach probability must be
      simulated and reported — confirm it is, and confirm the reported number is computed from the
      Monte Carlo rather than assumed.

### 3.6 NLP refinement (3.5)

- [ ] Verify the refined objective is compared to the MILP objective **using the same evaluation
      function**. If the MILP objective comes from the pre-computed tensor and the refined objective
      is recomputed by calling `predict` directly, any interpolation error in the tensor shows up as
      spurious "improvement."
- [ ] Verify the "never worse" guarantee is enforced by an explicit comparison-and-revert, not by
      assumption.
- [ ] Verify monotonicity and bounds are re-checked after refinement, and that the refined solution
      is re-run through the independent constraint checker.

---

## 4. Simulation & risk audit

### 4.1 Parameter sampling

- [ ] **Full covariance, not marginal SEs.** Phase 4.1 says sample `β ~ N(β̂, SE)`. If each
      coefficient is sampled independently from its marginal standard error, the joint draws are
      wrong — `β_price` and `β_floor` are correlated, sometimes strongly. Verify the code uses the
      full coefficient covariance matrix (`variance_matrix_` from lifelines) via Cholesky, and that
      Phase 2 actually persisted it (Phase 4 is where a missing covariance matrix first bites).
- [ ] Verify the covariance matrix is positive semi-definite before decomposition, with a clear
      error if not.
- [ ] Check that macro parameter volatilities (`σ_hist` for mortgage rates, FX) are estimated from
      the actual historical series pulled in Phase 1, not hardcoded. This is a direct `AGENTS.md`
      violation if hardcoded.
- [ ] Verify sampled parameters that must be bounded stay bounded: probabilities in [0,1],
      construction delay a non-negative integer, factors positive. Check what happens to a sampled
      `β_price` that comes out *positive* (possible if SE is large) — does the optimizer then price
      everything at the ceiling in that scenario, producing a fat right tail that inflates P95?

### 4.2 The tautology problem

- [ ] **`P(model > baseline)` is computed by evaluating both the model plan and the baseline plan
      under the model's own demand function.** This is not evidence that the model beats flat
      pricing in the world. It is evidence that the optimizer successfully optimized its own
      objective — which is a tautology, since the baseline is a feasible point in the optimizer's
      search space and the optimum is by construction at least as good.
- [ ] Confirm this is exactly what the code does, then confirm the framing in every output artifact
      is honest about it. `P(model > baseline)` near 1.0 is not a result; it's a unit test.
- [ ] The same critique applies to the headline "X% revenue uplift." Under the model's own demand
      function, uplift is guaranteed to be ≥ 0. The only informative version of this number comes
      from Phase 5 against actual historical outcomes, with the caveats in §5 attached.
- [ ] **This is the finding most likely to appear in a write-up in a form that overstates it.** Make
      the report explicit about what the number does and does not mean, and check the dashboard
      labels it accordingly.

### 4.3 Risk metrics

- [ ] Percentile ordering (P5 ≤ P25 ≤ P50 ≤ P75 ≤ P95) and `CVaR₅ ≤ P5` — assert, don't assume.
- [ ] Verify CVaR is the *mean of the tail*, not the 5th percentile itself, and that it uses ≤ (not <)
      and handles ties.
- [ ] Verify percentiles use a consistent interpolation method and that 10,000 draws is enough for a
      stable P5 (it is; check P1 if it's reported anywhere — it isn't stable at 10k for a heavy tail).
- [ ] **Convergence check.** Report the Monte Carlo standard error of the mean and of P5. If the
      P5 standard error is a meaningful fraction of the spread being reported, 10,000 iterations is
      not enough and the risk report is noise.
- [ ] **Seed control.** Verify seeding is explicit and reproducible, and that it is *not* set from a
      global `np.random.seed` inside a parallelized loop (each worker would then draw identical
      scenarios, collapsing the distribution to a spike while still returning 10,000 numbers). If
      parallelism is used, verify per-worker independent generators (`np.random.default_rng` with
      spawned seeds).
- [ ] Verify Approach A (fixed strategy, re-evaluated) and Approach B (re-optimized) are not being
      conflated in the reported distribution. They answer different questions and Approach A is
      strictly narrower — reporting A's spread as "the uncertainty in revenue" understates it,
      because it holds the plan fixed at the plan chosen under the *base* parameters.

### 4.4 Sensitivity

- [ ] Verify ±1σ tornado moves use the same σ as the Monte Carlo marginals.
- [ ] Verify each parameter is moved holding others at base, and that the base case is re-evaluated
      (not cached from a different configuration).
- [ ] Check whether the tornado accounts for re-optimization. Varying `β_price` ±1σ while holding
      the *plan* fixed measures something different from varying it and re-solving. Both are
      legitimate; the chart must say which.

---

## 5. Backtest audit (Phase 5)

The backtest is the only part of the system that can produce a non-tautological result. It is
therefore the part most worth attacking.

- [ ] **Does a real backtest exist?** The session record notes that backtesting against completed
      projects requires developer-internal unit-level data, which was not obtained. If Phase 5 was
      built against synthetic or reconstructed data, **the output is not a backtest** and must not be
      described as one anywhere in the codebase, API, dashboard, or documentation. Check every label.
- [ ] Verify the leakage guard (`every feature timestamp ≤ decision date`) covers derived features,
      the demand model's *training window* (a model trained on 2024–2026 data cannot be used to
      backtest a 2021 launch), and the submarket-month medians.
- [ ] **The demand model itself is the leak.** If `β_price` was estimated on data spanning the
      backtest period, the model has seen the outcome distribution it is being tested against. This
      is a fundamental limitation of backtesting with a single dataset. Confirm it is documented.
- [ ] Verify the honesty mandate from the spec is implemented as *output*, not just prose in a doc:
      results framed as an **upper bound under perfect macro foresight** and a **lower bound under
      macro uncertainty**, with both bounds actually computed and reported.
- [ ] Check the attribution decomposition (pricing vs sequencing vs timing). Verify the components
      sum to the total and that the decomposition order is stated (these decompositions are
      path-dependent; a different order gives different attributions).
- [ ] Compare model expected revenue against actual realized revenue **on a like-for-like basis** —
      accounting for the unsold-mass problem in §3.2, VAT/fees, and any units excluded from the
      model.

---

## 6. Software correctness & bug hunt

Read for these specifically. Each has bitten a pandas/sklearn pipeline before.

### 6.1 Data manipulation

- [ ] **Merge row inflation.** Any `merge` on a non-unique key silently multiplies rows. Assert
      row counts before/after every join in the pipeline. This is the most common silent
      data-science bug and it changes every coefficient without raising anything.
- [ ] **Silent NaN drops.** `dropna()` without a subset, or sklearn silently erroring vs pandas
      silently dropping. Log row counts at every stage and emit a waterfall (rows in → rows out →
      reason) in the validation report.
- [ ] **`SettingWithCopyWarning`** and chained assignment that doesn't take effect.
- [ ] **`groupby().apply()` vs `.transform()`** — index misalignment producing NaN or, worse, silently
      misaligned values assigned back to the wrong rows.
- [ ] **Date arithmetic.** Off-by-one in day counts; mixing timezone-aware and naive timestamps;
      `pd.to_datetime` with `errors='coerce'` producing NaT that then flows into a duration as NaN.
      Check the date parsing format is explicit (US M/D/Y from Matrix) rather than inferred per-row.
- [ ] **String-to-numeric coercion.** MLS exports contain currency symbols, commas, and blanks.
      Verify parsing is explicit and that failures are counted, not coerced to 0 (a price of 0 is
      catastrophic and will not obviously look wrong in a mean).
- [ ] **Floor parsing.** `Unit Floor Location` in MLS data is free-text-ish ("PH", "GR", "2", "12A",
      "LPH"). Verify the parser handles these and that unparseable values become null, not 0 or 1.
      A penthouse coded as floor 0 inverts the floor premium.
- [ ] **Duplicate handling.** Same ML# appearing twice; same address across relists.

### 6.2 Modeling code

- [ ] Fixed `random_state` everywhere.
- [ ] Scalers/imputers fit on train only, inside a `Pipeline` where possible.
- [ ] Categorical encoding consistent between train and predict (unseen categories at predict time —
      does it crash, or silently encode as a wrong level?).
- [ ] **Feature order dependence.** If `predict` builds its feature matrix by column position or by
      an unordered `dict`/`set`, a caller passing columns in a different order gets a silently wrong
      answer. Assert column names match the fitted feature list on every predict call.
- [ ] Model artifacts versioned, and the version pinned by the optimizer and backtest (Phase 2.7
      requires this — verify the pin is enforced, not just recorded).
- [ ] Mutable default arguments; global mutable state in FastAPI dependencies; a model object loaded
      once at import and mutated per-request.

### 6.3 Performance

- [ ] **Time the demand pre-computation.** Phase 3.2 evaluates `N × P × K` grid points. At 500 units
      × 4 phases × 15 levels = 30,000 calls. If `predict` calls
      `lifelines.predict_survival_function` per row, this takes minutes to tens of minutes and the
      Monte Carlo (which may rebuild demand 10,000 times) becomes infeasible. Verify vectorization
      and record a benchmark.
- [ ] Verify the tensor cache key includes **model version and macro scenario**, not just inventory
      hash — otherwise a re-solve after retraining returns stale revenue.
- [ ] Verify the 10,000-iteration simulation completes within an acceptable wall time and that the
      SSE progress events don't block the compute loop.

### 6.4 Error handling

- [ ] Find every bare `except:` and every `except Exception: pass`. Each one is a place a wrong
      answer becomes an invisible answer.
- [ ] Verify the fallback chains. Per `AGENTS.md`, diagnostics fail loudly. If the LLM estimator,
      or a default parameter set, can substitute for a failed real fit **without the response
      changing its provenance**, that is a critical finding.

---

## 7. Security & robustness audit

Even at prototype stage, and much more so if Phase 8 (SaaS, auth, billing) is live.

### 7.1 Input handling

- [ ] **File upload** (`/api/projects/{id}/upload/inventory`): enforce a max file size, a max row
      count, and an allowlist of extensions verified by content, not filename. Test a 500 MB CSV, a
      zip-bomb XLSX, and a `.csv` that is actually a shell script.
- [ ] **Path traversal** in any filename used to construct a path (`../../etc/passwd`,
      absolute paths, null bytes). Never join user input into a filesystem path without
      normalization and containment checks.
- [ ] **Formula injection** on export: cells beginning with `=`, `+`, `-`, `@` in a generated
      CSV/XLSX execute in Excel. Prefix-escape on write.
- [ ] **CSV parsing DoS**: pathological quoting, extremely wide rows, deeply nested archives.
- [ ] Validate uploaded inventory against the frozen schema *before* any modeling code touches it,
      with clear per-row errors.

### 7.2 Deserialization and code execution

- [ ] **`pickle` / `joblib.load` on any path influenced by a user is remote code execution.** Model
      artifacts must load only from a server-controlled directory with a validated version
      identifier. If model artifacts are ever uploaded or restored from user-controlled storage,
      this is critical severity.
- [ ] Grep for `eval`, `exec`, `pd.eval`, `df.query` with user input, `subprocess` with
      `shell=True`, and `yaml.load` without `SafeLoader`.

### 7.3 API and tenancy

- [ ] **Tenant isolation.** For every endpoint touching a tenant-owned resource, verify the query
      filters on `tenant_id` derived from the *authenticated token*, not from a request parameter.
      Write a test: user A requests user B's project ID and must get 404/403, never 200.
      This class of bug (IDOR) is the most common and most damaging flaw in multi-tenant apps.
- [ ] Verify `/api/health` and the Stripe webhook are the only unauthenticated endpoints, and the
      webhook verifies its signature.
- [ ] **CORS** locked to specific origins — no `allow_origins=["*"]` combined with credentials.
- [ ] **Rate limiting** on the expensive endpoints. `/simulate` runs 10,000 optimizer evaluations;
      unauthenticated or unlimited access is a trivial denial-of-service and, on metered
      infrastructure, a billing attack.
- [ ] JWT: strong secret from environment (no default in code), sensible expiry, algorithm pinned
      (reject `alg: none` and algorithm-confusion).
- [ ] SQL: parameterized queries only; no f-string interpolation into SQL, including in DuckDB paths.
- [ ] Error responses must not leak stack traces, file paths, SQL, or environment values in
      production. Verify `DEBUG` is off by default.

### 7.4 Secrets and supply chain

- [ ] `.env` git-ignored; only `.env.example` committed; **run a scan over git history**, not just the
      working tree — a key committed once and later removed is still a leaked key. Check for
      `FRED_API_KEY`, `BLS_API_KEY`, `OPENAI_API_KEY`, `JWT_SECRET`, Stripe keys, and MLS credentials.
- [ ] **Check whether `PRICING_MODEL_export.csv` is committed to the repo.** It is licensed MLS data
      containing addresses and transaction details, obtained under a broker's membership. It should
      almost certainly not be in a public repository, and if the repo is ever made public for the
      application write-up, this becomes a real problem for the broker who provided access. Flag it
      and confirm `.gitignore` covers `data/raw/`.
- [ ] Dependencies pinned; run a vulnerability scan; confirm no package is installed from an
      unpinned git URL.
- [ ] **Prompt injection into the LLM estimator.** If scraped listing text or user-uploaded fields are
      interpolated into an LLM prompt, a crafted listing description can manipulate the returned
      parameters. Confirm the LLM output is schema-validated and range-checked before use, and that
      it can never silently override a real fitted coefficient.

---

## 8. Claims, provenance & honesty audit

This section is about what the system *says*. It matters as much as what it computes.

- [ ] **Every numeric output carries provenance.** Per `AGENTS.md`, every response should say what
      model version produced it, what data it was calibrated on, and whether that calibration was
      real or synthetic. Verify `IS_CALIBRATED_ON_REAL_DATA` is not merely defined but **propagated
      to the API response and rendered in the UI**, and that it cannot be true for a Romania run.
- [ ] **Search for hardcoded estimated parameters.** Grep for numeric literals near
      elasticity/premium/coefficient names. `updates.md` gives α=0.058, γ=-0.038 for floor premium
      as *expected ranges for validation*, not values to use. If those numbers appear as defaults or
      fallbacks anywhere in the modeling path, that is a direct violation of the repo's central rule
      and it will produce plausible-looking output with no empirical basis.
- [ ] **Confidence intervals on the headline number.** If the system reports "X% uplift" without an
      interval derived from `β_price`'s standard error, add one. A point estimate of 8% with a
      credible range of −2% to +19% is a different claim than 8%.
- [ ] **Write or verify a model card** stating, in plain language: what data the model was fit on
      (Miami-Dade resale condos ≥ $400k, 2024–2026, n=5,000, truncated export); what it estimates;
      what it does *not* transfer to (new construction, primary sales, Romania); the known
      assumptions (independence across units, no unsold carry-over, PH assumption, view field
      missing); and the fact that `P(model > baseline)` is model-internal.
- [ ] **Check the dashboard's labels and tooltips against this list.** A chart titled "Expected
      Revenue" that is actually "expected revenue conditional on the model's demand function, summing
      only over units that sell within horizon T" is a claim the interface is making on the model's
      behalf.
- [ ] Verify the README and any generated report do not describe capabilities that are stubs
      (Bucharest pipeline, backtest against real completed projects, cross-elasticity in the
      optimizer, RL, LLM demand estimation).

---

## 9. Required output

Produce **`audit/AUDIT_REPORT.md`** with:

### 9.1 Executive verdict

Three to five sentences answering one question: **can the numbers this system produces be shown to
a real developer and to an admissions committee?** Give one of: `SOUND WITH CAVEATS` /
`SOUND ONLY FOR X` / `NOT YET DEFENSIBLE`, and the reasoning.

### 9.2 The four numbers

Report these prominently, whatever else the audit finds:

| Number | Why |
|---|---|
| `β_price` estimate, SE, 95% CI | The system's foundation |
| % of (submarket × month) cells with n ≥ 8 | Whether the identification variable is signal or noise |
| % of units at the price ceiling in the optimal plan | Whether the optimizer is doing economics or exploiting extrapolation |
| Total unmodeled probability mass `Σ(1 − D_i)` | How much of "expected revenue" is a partial sum |

### 9.3 Findings register

One table, every finding, sorted by severity:

| ID | Severity | Layer | Title | Evidence (file:line, script, or output) | Impact on the headline number | Fix status |
|---|---|---|---|---|---|---|

Severity rubric:

- **CRITICAL** — the output is wrong or a security flaw is exploitable. Examples: leakage in the
  demand model, positive `β_price` shipped as-is, tenant isolation broken, RCE via deserialization.
- **HIGH** — the output is materially biased or a stated capability doesn't exist. Examples:
  selection on missing `List Date`, cross-elasticity claimed but not implemented, backtest label
  applied to synthetic data.
- **MEDIUM** — correctness is fine but the result is fragile, unvalidated, or misleadingly
  presented.
- **LOW** — code quality, performance, maintainability, cosmetics.

Add a column marking whether the finding came **from this checklist** or **from open-ended
discovery (§1.3)**. Report the ratio explicitly. A register composed almost entirely of checklist
items means §1.3 was underworked — say so honestly rather than presenting it as a clean result.

### 9.4 Fixed vs documented

Two lists.

**Fixed** — every change made, with the diff, the reasoning, the regression test, and the effect on
the four numbers in §9.2. The default is that a finding appears here. If a defect was found and
*not* fixed, the report must state why, in one of exactly two forms: (a) it cannot be fixed with
available data or information, or (b) fixing it would require a specification decision that is not
yours to make — in which case state the decision required and your recommendation. "Out of scope"
is not an acceptable reason and should not appear in the report.

**Documented** — limitations that survive the fix pass, each with the *exact wording* that should
appear in the model card, the API provenance block, the dashboard, and the written report. Write
these as publishable sentences, not as notes to self. This list is not a failure; it is the output
that makes every other number in the project credible.

### 9.5 Regression tests added

Every finding at MEDIUM or above gets a test under `tests/` that fails on the old behavior and
passes on the new. List them with the finding ID each one guards. Include the property-based and
metamorphic tests from §1.3 — those guard against the class of defect, not just the instance, and
are the most valuable artifacts this audit produces.

### 9.6 Coverage statement

State plainly, per layer: what you audited, what you audited *and* fixed, what you inspected but
could not verify, and what you did not reach. An honest map of the unexamined is worth more than
an implied claim of completeness. If you ran out of depth anywhere, name it here rather than
letting silence imply coverage.

### 9.7 What a competent skeptic would still attack

Close with the questions a quantitatively literate developer, or a professor reading the write-up,
would ask that this audit could not fully answer. Give at least three; give more if they exist.
Being able to state these is worth more than being able to avoid them.

---

## 10. Explicit non-goals

The mandate is to fix everything wrong. It is not a mandate to change everything. Do not, during
this audit:

- **add features, models, or endpoints** that don't exist — correctness work only
- **refactor for style or taste.** Restructure only where the structure is the defect (e.g. shared
  mutable state, a function that cannot be tested in isolation, duplicated logic that has already
  drifted between copies)
- **tune hyperparameters or alter model specification to improve a metric.** Fixing a cause is in
  scope; improving a number is not
- **build the Bucharest data pipeline** or implement unbuilt Phase 8 items
- **remove or soften a documented limitation** because the audit found it uncomfortable
- **rewrite from scratch.** If a module is beyond repair, say so and specify what a correct
  replacement requires — don't spend the engagement rebuilding it

Note what is *not* on this list: touching unfamiliar modules, following a bug across layer
boundaries, fixing defects the checklist never mentioned, and adding assertions, guards, and tests
anywhere they are missing. All of that is in scope and expected.

If you believe a specification is wrong (as opposed to the implementation being wrong), say so in
the report with your reasoning. Do not unilaterally change the spec.

---

## 11. The standard

The test of this audit is not whether every box below is ticked. It is whether, after you are
finished, the following is true:

> A quantitatively literate real estate developer could act on this system's recommendations
> without being misled, and a hostile expert reviewer given full access to the code and the data
> could not find a material defect that the audit missed or a claim the system makes that the
> report does not already qualify.

Work until that is true, or until you can state precisely why it isn't and what would be required
to make it so.
