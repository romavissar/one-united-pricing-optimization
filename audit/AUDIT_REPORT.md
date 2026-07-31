# AUDIT_REPORT.md — Full-system audit of the Dynamic Pricing & Unit Release Optimizer

Engagement per `claude-chat-logs/AUDIT_BRIEF.md`. All numbers in this report were
re-measured against the repaired system; reproduction scripts are under `audit/`.

Test suite at start: **143 passed, 4 failed**. At finish: **188 passed, 0 failed**.

---

## 9.1 Executive verdict

**`SOUND ONLY FOR` demonstrating the method — not yet for pricing a building, and
not for a write-up that quotes an elasticity or a revenue uplift as a finding.**

The engineering is genuinely good: the objective is dimensionally correct and
matches an independent loop-based recomputation to the cent, the MILP is clean
(no fractional binaries, no name collisions, monotonicity verified on the
extracted plan rather than assumed), the survival conversion correctly delegates
centering to lifelines instead of hand-rolling the classic uncentered bug, and
there is no outcome leakage in the feature matrix. The elasticity is really
wired into the objective: at `β_price = 0` every released unit goes to its
ceiling, and at −3.0 versus −0.4 no unit prices higher. Those are the checks
that distinguish a working pipeline from a decorative one, and they pass.

What is not sound is the number the whole system exists to produce. On the real
export `β_price` is **−0.190** (95% CI [−0.279, −0.102]) — correctly signed and
excluding zero, so it passes every sign check the repo enforces, but roughly a
third of the −0.5 magnitude the specification treats as the floor of a usable
elasticity. Three separate mechanisms push it toward zero, and I could measure
all three but repair none of them: the export is a stock sample of listings that
*terminated* in a 289-day window (so the slow, over-priced listings that identify
a price response are structurally absent), it was truncated at 5,000 rows against
a status-sorted result (so the purest did-not-sell outcome is under-represented
by an unknown amount), and `rel_price_premium` still has 37% of its variance
explained by unit characteristics rather than pricing choice. None of these is a
code defect. All three are properties of how the data was pulled.

Two findings would have been serious if shipped. The market config carried
`is_calibrated_on_real_data: true` while nothing had ever been calibrated, and
the frontend ORed that market-level flag into its banner logic — so
synthetic-data plans rendered with no illustrative banner at all, which is
precisely the screenshot-into-a-client-deck failure `AGENTS.md` §4 exists to
prevent. Both are fixed. Separately, 661 rows — **every** PENDING and **every**
WITHDRAWN listing — were dropped from identification because the export leaves
`List Date` blank for exactly those statuses; that is a panel selected on the
outcome (χ² = 4954, p ≈ 0), and it is now repaired by deriving the date from
terminal-minus-DOM.

An admissions committee can be shown this as a well-built decision system with an
honestly-stated identification failure. It cannot be shown "an 8% revenue uplift".
The uplift this system currently reports has a 90% band of **−$4.9M to +$9.7M**
and contains zero — under the model's own demand curve, on synthetic data.

---

## 9.2 The four numbers

Measured after the fix pass by `audit/a12_headline_numbers.py`.

| # | Number | Value | Reading |
|---|---|---|---|
| 1 | `β_price` (real export, hedonic controls) | **−0.1902**, SE 0.0451, 95% CI **[−0.2785, −0.1019]** | Correctly signed, excludes zero, **outside the [−0.5, −3.0] plausibility gate — too weak**. Naive covariate set gives −0.1708 (SE 0.0460). |
| 2 | (submarket × month) cells with n ≥ 8 | **96 of 145 = 66.2% of cells**, holding **96.9% of keyed rows** | The denominator is signal, not noise. Median cell 17 listings; measured attenuation from cell-median sampling error is a factor of 0.996 — negligible. |
| 3 | Units at the price ceiling in the optimal plan | **40 of 60 = 66.7%** (0% at floor, mean level 12.72 of 14) | On synthetic data at `β_price = −1.44`. Two thirds at the ceiling is a strong corner-solution signal; the alarm threshold was 0.80 and did not fire. Now 0.50, and it fires. |
| 4 | Unmodelled probability mass `Σ(1 − Dᵢ)` | **19.30 of 60 units = 32.2%**; **$36,454,105 = 33.8% of asking value** | A third of the released stack's value is never counted. "Expected revenue" is a partial sum presented as a total. Now reported as a first-class field and as a caveat. |

Note on #1 and #3: they are measured on different data by necessity. The
calibration gate (`PROJECT_BRIEF.md` §5) reserves the decision to fit on real MLS
for a human, and it is not this audit's to open, so #3 and #4 come from the
example inventory against a synthetic fit. #1 and #2 come from the real export
because that is what the coefficient would be calibrated on. Running the real-data
*diagnostics* (which saves no bundle and flips no flag) was necessary to audit
§2 at all; I did not save a bundle from real data and did not flip the flag.

---

## 9.3 Findings register

`Src` = **C**hecklist (§2–§8) or **O**pen-ended discovery (§1.3).

| ID | Sev | Layer | Title | Evidence | Impact on the headline number | Src | Fix |
|---|---|---|---|---|---|---|---|
| F01 | CRITICAL | Config | `is_calibrated_on_real_data: true` with nothing calibrated | `config/miami.yaml:76`; `test_api.py` asserted `False` and failed | None on the maths; removes the banner that stops a synthetic number reaching a client | C §8 | Fixed |
| F02 | CRITICAL | Frontend | Banner keys off market config ORed with result provenance | `ProjectContext.jsx:53-56` | A synthetic-fit plan renders unbannered whenever the market flag is true | O | Fixed |
| F03 | HIGH | Data | 661 rows — 100% of PENDING and 100% of WITHDRAWN — dropped for a blank `List Date` | `audit/a05_panel_audit.py` §2.3: χ²=4954, p≈0 | Panel selected on outcome. Fit sample 3,738 → 4,284; events 1,661 → 1,892 | C §2.3 | Fixed |
| F04 | HIGH | Data | Export sorted by status, capped at 5,000 vs a "5000+" search → `Expired` truncated | `audit/a06_sampling_window.py` §2 | Truncation on the outcome variable; sold:unsold mix is not the market's | C §2.6 | Detected + documented |
| F05 | HIGH | Data | Export is a stock sample on *terminal* date (289-day window vs 837-day list span, zero ACTIVE) | `audit/a06_sampling_window.py` §1 | Slow/unsold listings structurally absent → `β_price` attenuated toward zero | O | Detected + documented |
| F06 | HIGH | Data | Two corrupt records ($49.5M/798 sqft) set `premium_support.high = +109.46` | `audit/a06_sampling_window.py` §3 | Extrapolation guard could flag no price at all. Premium sd 1.867 → 0.752 | O | Fixed |
| F07 | HIGH | Simulation | Coefficient covariance computed, discarded, never persisted; only `β_price`'s marginal SE propagated | `audit/a09_simulation_audit.py` §4.1 | Reported P5–P95 band is a lower bound on parameter uncertainty | C §4.1 | Partly fixed — see §9.4 |
| F08 | HIGH | Data | `pd.NA` leaks every null guard written `isinstance(x, float) and pd.isna(x)` | 25 sites in `normalize.py`; `slug(pd.NA) == "na"` | `unit_key` fallback never fires and all such units collide; `hoa_frequency` = `"<NA>"` | O | Fixed |
| F09 | HIGH | Demand | Building fixed effects pool 94% of towers and emit suppressed convergence warnings on real data | `a12` run; lifelines complete-separation + `norm(delta)=0.109` | `--building-fe` is the *recommended* calibration command and silently does not do what it says | O | Fixed |
| F10 | MEDIUM | Optimizer | Unsold probability mass (32.2%, $36.5M) never surfaced | `audit/a08_optimizer_audit.py` §3.2 | Expected revenue is a partial sum presented as a total | C §3.2 | Fixed |
| F11 | MEDIUM | Optimizer | Revenue discounted to phase release date though `D` is P(sale within T) | `audit/a08_optimizer_audit.py` §3.1d | Overstates present value by 2.83% on the example inventory | C §3.1 | Documented — see §9.4 |
| F12 | MEDIUM | Demand | Cox baseline flat past last event; caller-settable `horizon_days` returns it silently | `audit/a07_cox_and_leakage.py` §4 | P(sale by 3650d) == P(sale by 180d) exactly | C §2.4 | Fixed |
| F13 | MEDIUM | Tests | Recovery acceptance test mis-calibrated: 20% relative = 0.95 SE at β=−0.4 | `audit/a04_recovery_seed_sweep.py`: fails 12/24 seeds on an unbiased estimator | The repo's headline test was a coin flip at its smallest parameter | O | Fixed |
| F14 | MEDIUM | Tests | Synthetic generator coupled to `config/miami.yaml`; 8→14 submarkets silently changed every benchmark | `audit/a11_building_fe_coupling.py` | FE recovery 74% → 44% with no estimator change; README tables stale | O | Fixed |
| F15 | MEDIUM | Security | CSV export formula injection + incomplete quoting | `ReleasePlanTable.jsx:119-135` | `=`/`+`/`-`/`@` in an uploaded `unit_id` executes in Excel | C §7.1 | Fixed |
| F16 | MEDIUM | Security | `/api/inventory/validate` buffers an unbounded body; no row cap | `routes.py:60` | Trivial memory-exhaustion DoS | C §7.1 | Fixed |
| F17 | MEDIUM | Security | `market` and bundle `name` joined into filesystem paths without containment | `config.py`, `registry.model_dir` | The path's endpoint is a pickle `load_bundle` unpickles | C §7.2 | Fixed |
| F18 | MEDIUM | Frontend | "Revenue distribution" draws a fabricated density from 7 percentiles | `RevenueDistribution.jsx:28-29` | Shape communicates skew/tails the data never showed | O | Fixed |
| F19 | MEDIUM | Frontend | Uplift reported as difference of medians, no interval | `RevenueSummary.jsx:19-20` | Real paired uplift band is −$4.9M to +$9.7M and contains zero | O | Fixed |
| F20 | MEDIUM | API | `/api/optimize` echoes the *request's* comps, not the resolved ones | `services.py:260` | Client round-tripping to `/simulate` loses bundle-derived comps | O | Fixed |
| F21 | MEDIUM | Frontend | Tornado does not distinguish fitted from assumed bars | `SensitivityTornado.jsx` | Largest bar ($11.3M absorption) is an assumption; fitted β_price is $1.7M | O | Fixed |
| F22 | LOW | Optimizer | Ceiling alarm at 0.80 never fires at the observed 66.7% | `solve.py:57` | The corner-solution warning was unreachable in the common case | O | Fixed |
| F23 | LOW | API | `_synthetic_result` measured floors on nominal revenue regardless of basis | `services.py:306` | Spurious breach report under `cash_flow_basis="discounted"` | O | Fixed |

**Checklist vs open-ended: 10 from the checklist, 13 from open-ended discovery
(§1.3) — a ratio of 1 : 1.3.** The three findings I would call most consequential
(F02, F03, F05) all came from open-ended work: F02 from tracing what the banner
flag actually reads rather than trusting that a flag named for calibration
governs calibration; F03 from asking "the brief says test whether the missing
`List Date` is random — random *with respect to what*?" and cross-tabulating
against status; F05 from noticing while checking §2.3's censoring question that
the terminal-date range was implausibly narrow next to the list-date range.

---

## 9.4 Fixed vs documented

### Fixed

Each entry: what changed, why, the regression test, and the effect on §9.2.

**F01 — market config claimed calibration.**
`config/miami.yaml`: `is_calibrated_on_real_data: true` → `false`. Every document
in the repo (`CALIBRATION_READY.md` §6, `.env.example`, `README.md`) states this
must stay false until a human inspects a real fit, and the shipped API test
asserted `False` and was failing. Test: `test_f01_market_config_is_not_marked_calibrated`.
Effect on §9.2: none numerically; restores the banner.

**F02 — banner suppressed by a market-level flag.**
`ProjectContext.jsx`: `isCalibrated` now reads only the provenance of the result
on screen, defaulting to *not calibrated* when provenance is absent. The market
config flag says a human has cleared the market for real-data fits; it says
nothing about what produced the numbers currently displayed. Under the old logic,
flipping the config legitimately after a real fit would have unbannered every
subsequent synthetic run. Effect: none numerically.

**F03 — `list_date` recovered, restoring PENDING and WITHDRAWN to the panel.**
New `normalize.reconstruct_list_date`. Validated identity, measured on the
statuses where all three fields are present: `pending_date − DOM` matches the
reported list date exactly for 93.6% of SOLD rows; `off_market_date − DOM` for
90.0% of CANCELED and 90.3% of EXPIRED. (`close_date − DOM` matches 0.9% — DOM
stops at buyer commitment, not at closing, which is why the escrow period must
not be used.) Every recovered value carries `list_date_source="derived_from_dom"`
per `AGENTS.md` §2. This is a derivation from two present columns, not an
imputation: nothing is borrowed from another listing.
Tests: `test_f03_list_date_is_recovered_from_terminal_minus_dom`,
`test_f03_recovered_rows_carry_a_source_marker_not_a_silent_value`.
Effect on §9.2: **#1** fit sample 3,738 → 4,284 rows, events 1,661 → 1,892;
`β_price` moved −0.1716 → −0.1708 (naive). **#2** dense cells 90 → 96.
That the coefficient barely moved on a materially better and less
outcome-selected sample is the most reassuring single result in this audit.

**F06 — implausible derived $/sqft nulled and marked.**
`clean.py`: `_PPSF_PLAUSIBLE_BOUNDS = (100, 20_000)` $/sqft, applied to derived
`list_ppsf` and `close_ppsf`, mirroring the gate `coerce_area` already applies to
raw area. Per `AGENTS.md` §2 the **row is kept**; only the untrustworthy derived
value is nulled, with a `*_source` marker. The band is set from physical
plausibility, not from the fit: the genuine $10,145/sqft Brickell trophy
penthouse is retained, and the export's own distribution jumps from $6,090/sqft
at the 99.9th percentile to $53,866 at the 99.99th — the gap between listings and
typos. Catches exactly 2 records ($33.99M/750 sqft and $49.5M/798 sqft, both
1-bed).
Test: `test_f06_implausible_ppsf_is_nulled_and_marked_not_dropped` (asserts both
that the typos are caught **and** that the real trophy price survives).
Effect: `rel_price_premium` sd 1.867 → 0.752, max +109.46 → +12.19;
`premium_support.high` 109.46 → 10.04, so the extrapolation guard can flag
something again.

**F08 — `pd.NA` recognised as missing.**
New `normalize.is_missing`, replacing 25 instances of
`isinstance(x, float) and pd.isna(x)` across `normalize.py`, `clean.py`,
`features.py`. `pd.NA` is an `NAType`, not a float, and `map_columns`
materialises every canonical field the export omitted as `pd.NA` — so whole
columns took the fall-through path. Consequences repaired: `slug(pd.NA)` returned
`"na"`, so `build_unit_key` never fell back to the street address and every unit
with no building name collided in one `na|<unit>` namespace; `hoa_frequency` came
out as the literal string `"<NA>"` on every row.
Tests: `test_f08_every_null_flavour_is_recognised` (parametrised over `None`,
`nan`, `pd.NA`, `pd.NaT`), `test_f08_present_values_are_not_treated_as_missing`
(guards the converse — `0`, `""`, `False` must not read as missing),
`test_f08_missing_building_name_falls_back_to_street_address` (asserts two
different unnamed buildings do not collide),
`test_f08_absent_hoa_frequency_stays_null_not_the_string_NA`.

**F09 — fixed-effect and convergence failures made loud.**
`survival.py` now captures lifelines' `ConvergenceWarning`/`RuntimeWarning`
during the fit rather than letting them go to stderr, and reports two structural
conditions: the share of fixed-effect levels pooled into `__rare__`, and the
listings-per-retained-level density. `diagnostics.py` promotes all three to
first-class `Check`s so they appear in the printed report and the API rather than
in a notes list nobody prints. On the real export with `--building-fe` — the
command `CALIBRATION_READY.md` §3 tells the operator to run — this now reports:
2,136 of 2,283 towers (94%) pooled, 28.8 listings per retained tower, and
lifelines' own complete-separation and `norm(delta)=0.109` non-unique-solution
warnings. Test: `test_f11_cox_reports_the_follow_up_it_was_fitted_over` covers
the related follow-up exposure; the FE thresholds are exercised by the real-data
CLI path.

**F10 — unmodelled probability mass surfaced.**
`OptimizeResult` gains `gross_ask_usd`, `unmodelled_probability_mass`, and
`unmodelled_revenue_usd`, all emitted in `as_dict()`, plus a caveat stating the
figure in units and dollars. Test: `test_f10_unmodelled_probability_mass_is_reported`.
Effect: makes §9.2 #4 a reported number rather than one an auditor has to compute.

**F12 — horizon beyond the fitted follow-up.**
`CoxDemandModel.max_observed_duration_days` exposes the range the non-parametric
baseline is identified over; `build_revenue_tensor` warns when
`spec.horizon_days` exceeds it. `horizon_days` is settable through the API, and
past the last event lifelines carries the last value forward, so a 365-day
horizon on a 180-day fit returns the 180-day probability with a longer label.
Test: `test_f11_cox_reports_the_follow_up_it_was_fitted_over` asserts the
probabilities at the edge and at 10× the follow-up are *identical* — which is the
point.

**F13 — recovery test re-expressed in standard errors.**
`RECOVERY_TOLERANCE = 0.20` (relative) → `RECOVERY_TOLERANCE_SES = 3.0` plus a
loose relative backstop. Justification is arithmetic, not taste: at n=6,000 the
fitted SE is ~0.084 whatever β is planted, because it is set by the spread of
`rel_price_premium` and the event count, not by the coefficient's size. So 20% is
6.1 SE at −3.0 and **0.95 SE** at −0.4. Over 24 seeds the estimator is unbiased
(mean −0.3957 against −0.4000, t = +0.21) and the old rule failed **12 of 24**.
A test a correct estimator fails half the time measures the seed, not recovery.
Evidence: `audit/a04_recovery_seed_sweep.py`.

**F14 — synthetic benchmarks decoupled from market config.**
The generator builds `_BUILDINGS_PER_SUBMARKET` towers per configured submarket,
so extending the config from 8 to 14 submarkets (done to cover the real export's
ZIPs) raised towers 96 → 168 and cut density ~60 → ~34 listings per tower.
Building-FE recovery fell 74% → 44% of the planted value **with no estimator
change** — incidental-parameters bias in a partial likelihood carrying one dummy
per level. `audit/a11_building_fe_coupling.py` measures the curve, including a
control that separates tower *count* from *density*. Fixes: the FE test pins the
submarket set it was measured on (restoring the documented 74%), and
`test_synth.py` derives its expected cell count from the config instead of
hardcoding 192.

**F15–F17 — security.**
CSV export prefixes `=`/`+`/`-`/`@`/tab/CR with an apostrophe and quotes any cell
containing a quote, comma, or newline (previously comma-only, so an embedded
quote broke the file). `/api/inventory/validate` streams with an 8 MB cap and a
20,000-row cap, checking both the declared `content-length` and the actual
streamed size so a chunked request cannot understate it. `market` and bundle
`name` are validated against allowlists before any path join —
`validate_market_id` and `_BUNDLE_NAME` — because the endpoint of that path is a
pickle `load_bundle` unpickles. Tests:
`test_f16_market_ids_that_are_paths_are_rejected`,
`test_f16_bundle_names_that_are_paths_are_rejected`,
`test_f16_valid_identifiers_still_resolve`.

**F18, F19, F21 — presentation.**
The revenue-distribution chart now carries an on-screen warning that the marked
percentiles are measured and the curve between them is interpolated. The uplift
statistic is now the **paired per-draw difference** (both arms already share the
sale lottery via common random numbers) reported with a 90% band, rather than a
difference of two separately-ranked medians with no interval; new
`RevenueDistribution.uplift_vs_baseline_usd`. The tornado draws assumed channels
hollow and suffixes their labels `(assumed)`.

**F20, F22, F23 — smaller correctness.**
`/api/optimize` echoes the comps the solve actually used. The ceiling alarm moved
0.80 → 0.50 (it could not fire at the observed 66.7%). `cash_flow_basis` now
threads into `/simulate` and `/sensitivity` so floors are measured on the basis
the plan was solved under.

### Documented — limitations that survive the fix pass

Publishable wording. These belong in the model card, the API provenance block,
the dashboard, and any written report.

**D1 — the sample is drawn on termination, not on listing.** *(cannot be fixed
with available data)*

> This model was fitted on listings that went off market between 14 October 2025
> and 30 July 2026 — a 289-day window — while the listings' start dates span 837
> days. No listing that was still on the market at the export date appears in the
> sample. A listing that began before the window is present only if it lasted
> long enough to survive into it, and a slow-selling, over-priced listing that had
> not yet terminated is absent entirely. Because those are precisely the
> observations that identify how demand responds to price, the estimated
> elasticity is attenuated toward zero by an amount this dataset cannot measure.
> The fitted coefficient should be read as a lower bound on the true elasticity in
> magnitude, not as an estimate of it.

*Recommendation:* re-pull the export filtered on **list date**, including ACTIVE
listings, so that right-censoring is administrative rather than a condition of
being in the sample. The ingest report now detects and warns on this pattern.

**D2 — the export is truncated on the outcome variable.** *(cannot be fixed with
available data)*

> The export contains exactly 5,000 rows against a search that reported more
> matches than it returned, and the file arrives sorted by status in six
> contiguous blocks ending with Expired. A row cap applied to a status-sorted
> result truncates the last block, so expired listings — the purest record of a
> unit that was offered and did not sell — are under-represented by an unknown
> amount. The ratio of sold to unsold in this sample is therefore not the
> market's ratio.

*Recommendation:* re-pull sorted on ML# or list date, and compare the delivered
row count against the search's reported total.

**D3 — resale listings, not new-construction primary sales.** *(cannot be fixed
with available data — the single most important limitation of the system)*

> Every coefficient in this model is estimated from **resale condominium
> listings by individual sellers** in Miami-Dade County at $400,000 and above.
> The tool's application is **new-construction primary sales by a developer**,
> released in phases and often before completion. These are different markets:
> different buyers, different financing, different price discovery, different
> inventory dynamics, and different elasticities. A developer pre-selling a tower
> is not observing the same demand curve as an owner relisting a unit. Nothing in
> this dataset measures the transfer, and the size of the gap is unknown.

**D4 — `β_price` is below the magnitude a pricing recommendation requires.**
*(cannot be fixed with available data)*

> The fitted own-price elasticity is −0.190 (95% CI [−0.279, −0.102]). It is
> correctly signed and statistically distinguishable from zero, but roughly a
> third of the −0.5 magnitude below which an optimizer's expected-revenue
> objective becomes monotonically increasing in price. At this value the
> recommended prices are driven substantially by where the price ceiling was
> placed rather than by measured demand response. 37% of the variation in the
> identifying variable is explained by unit characteristics rather than by the
> seller's pricing choice, which is classical measurement error in a regressor
> and attenuates the coefficient toward zero regardless of sample size.

**D5 — `P(plan > baseline)` and the revenue uplift are model-internal.** *(cannot
be fixed without out-of-sample outcome data)*

> The comparison against flat pricing evaluates both the recommended plan and the
> baseline under the model's own demand function. The baseline is a feasible
> point in the optimizer's search space, so in expectation the optimum is at
> least as good by construction. `P(plan > baseline)` is therefore a check that
> the optimizer optimized, not evidence that the plan beats flat pricing in the
> market. On the current synthetic fit the paired uplift is $3.19M with a 90%
> band of −$4.88M to +$9.67M — an interval containing zero. No backtest against
> realized outcomes exists in this repository, and none is claimed.

**D6 — expected revenue is a partial sum.** *(accepted simplification;
direction of bias established, magnitude bounded)*

> A unit released at a 70% sale probability contributes 70% of its price to
> expected revenue. The remaining 30% does not reappear: the formulation releases
> each unit at most once and carries no unsold inventory into a later phase, so
> that share leaves the model entirely. On the example inventory this is 19.3
> units' worth of probability mass across 60 released units — 33.8% of asking
> value, $36.5M. Expected revenue is therefore **not comparable to the gross
> development value of a sold-out project**, and the bias is toward
> understatement of total revenue while distorting phase sequencing.

**F11 — timing mismatch in the discount factor.** *(requires a specification
decision that is not mine to make)*

`D` is the probability of sale *within* a horizon `T`, but the whole of a unit's
expected revenue is discounted to its phase's release date rather than to when
the cash arrives, which is distributed over `[t_j, t_j + T]`. Measured
overstatement on the example inventory: **2.83%** at a 12% annual rate and a
180-day horizon. The correction requires choosing what the arrival-time
distribution is — discounting to the horizon midpoint, or integrating the fitted
survival density over the window — and that choice changes the phase-sequencing
recommendation, not just the headline. **My recommendation:** integrate against
the fitted survival function, since the model already produces it, and report
both the release-date and arrival-weighted figures during a transition so the
change is visible rather than silent. A caveat now states the direction and
approximate size on every plan.

**F07 — only `β_price`'s marginal uncertainty is propagated.** *(fixed in part;
the remainder requires a specification decision)*

The Cox coefficient covariance is now computed in raw units
(`CoxDemandModel.coefficient_covariance`) and persisted with every bundle as
`cox_coefficient_covariance`, closing the gap the brief identified — Phase 2 not
storing what Phase 4 needs. Every simulation now states in its caveats that the
band is a **lower bound** on parameter uncertainty. What is *not* done is drawing
the full coefficient vector: the closed-form perturbation in `monte_carlo.py`
shifts only the price channel, and extending it needs each released unit's design
row carried on the `RevenueTensor` and a decision about whether to resample the
hedonic surface's coefficients jointly with the demand model's (they are fitted
on overlapping samples, so treating them as independent would be wrong in a
direction nobody has measured). **My recommendation:** carry the (units × phases
× levels × p) design tensor — about 7 MB at 500 units — and draw
`Δβ ~ N(0, Σ)` via Cholesky, applying `Δη = Σ_m Δβ_m x_im` for every coefficient
except `rel_price_premium`, which the existing drift-aware path already handles.
Treat the hedonic surface as fixed in the first increment and say so.

**D7 — building fixed effects do not do what their name says on this export.**
*(cannot be fixed with available data)*

> The Miami export contains 2,596 distinct buildings across 4,899 listings — a
> median of one listing per building. With the minimum level size set to five,
> 94% of towers are pooled into a single residual category that absorbs nothing
> building-specific. The `--building-fe` option, which the calibration checklist
> recommends, therefore removes far less unobserved quality than its name
> implies, and the fitter reports complete separation and a non-unique solution
> when it is used. On synthetic data where the contamination is entirely
> building-level, recovery of a planted −1.6 falls from 74% of the planted value
> at ~60 listings per tower to 44% at ~34, purely from incidental-parameters
> bias. These conditions are now reported as diagnostic checks rather than left
> in the fitter's stderr.

---

## 9.5 Regression tests added

All in `backend/tests/test_audit_regressions.py` unless noted. 41 tests; each
fails on the pre-audit behaviour.

| Finding | Test |
|---|---|
| F01 | `test_f01_market_config_is_not_marked_calibrated` (also `test_api.py::test_config_miami_exposes_defaults_and_calibration_flag`, which was failing) |
| F03 | `test_f03_list_date_is_recovered_from_terminal_minus_dom`, `test_f03_recovered_rows_carry_a_source_marker_not_a_silent_value` |
| F04 | `test_f04_status_sorted_export_is_detected`, `test_f04_interleaved_status_is_not_flagged_as_sorted` (guards against a false positive) |
| F05 | `test_f05_terminal_date_window_selection_is_detected` |
| F06 | `test_f06_implausible_ppsf_is_nulled_and_marked_not_dropped` |
| F07 | `test_f07_coefficient_covariance_is_available_and_in_raw_units` |
| F08 | `test_f08_every_null_flavour_is_recognised` (parametrised ×5), `test_f08_present_values_are_not_treated_as_missing` (×5), `test_f08_missing_building_name_falls_back_to_street_address`, `test_f08_absent_hoa_frequency_stays_null_not_the_string_NA` |
| F10 | `test_f10_unmodelled_probability_mass_is_reported` |
| F12 | `test_f11_cox_reports_the_follow_up_it_was_fitted_over` |
| F13 | `test_cox_recovers_the_planted_beta_price` rewritten in `test_demand_synthetic.py` |
| F14 | `test_building_fixed_effects_recover_beta_when_quality_is_building_level` pinned; `test_written_csv_ingests_with_full_coverage` derives its expectation from config |
| F17 | `test_f16_market_ids_that_are_paths_are_rejected` (×6), `test_f16_bundle_names_that_are_paths_are_rejected` (×4), `test_f16_valid_identifiers_still_resolve` |
| F22 | `test_f21_ceiling_alarm_triggers_below_four_fifths` |

**Property-based and metamorphic tests** — these guard the *class* of defect
rather than the instance, and are the most valuable artifacts here:

| Test | Property |
|---|---|
| `test_metamorphic_scale_invariance_of_the_identification_variable` | Scaling prices and the comp median together leaves `rel_price_premium` fixed — catches a currency or sqft↔sqm conversion applied to one side of a ratio and not the other |
| `test_metamorphic_non_positive_median_yields_nan_not_zero` | A missing comp median must never read as "priced exactly at comps" |
| `test_property_percentiles_ordered_and_cvar_below_p5` | Parametrised over degenerate, constant, heavy-tailed, and symmetric samples |
| `test_property_summarize_rejects_an_empty_sample` | Fails loudly rather than returning NaN percentiles |

`audit/a10_metamorphic.py` carries a further 13 relations run as a script (unit-id
relabelling, inventory row-order permutation, elasticity monotonicity, β=0 →
ceiling, optimum ≥ a feasible point, monotone P(sale) in price on both a
fixed-β model and the fitted Cox). 13 pass; the 2 "failures" are a relation I
mis-stated — area *is* a demand covariate, so revenue does not scale exactly with
it (ratio 2.046, not 2.000). That is correct behaviour and is recorded as such.

---

## 9.6 Coverage statement

**Audited and fixed.** Data ingestion and normalization (alias mapping, type
coercion, null handling, floor parsing, duration construction, `unit_key`);
cleaning filters and the derived-price plausibility gate; feature engineering and
the submarket×month median ladder; the Cox model's design matrix, scaling
round-trip, survival conversion, and follow-up range; diagnostics and the sign
gates; the model registry and provenance; the MILP objective, constraints,
solver-status handling, and independent checker; Monte Carlo distribution, CVaR,
seeding, and the baseline comparison; the sensitivity tornado; the API request
and response surface; the React state layer and every chart's labels.

**Audited, verified correct, not changed.** The objective's dimensional
consistency (differential loop-based reimplementation matches PuLP to the cent;
an sqft↔sqm swap would show as a ratio of 8.6 against the 0.801 observed).
Discount factors (independently recomputed; annual-decimal convention correct,
and `npv.py` already rejects a rate passed as a percentage). Survival conversion
(delegates to lifelines, so covariate centering is handled; a hand-rolled
uncentered version differs by 3.2e-2, and that bug is *not* present). Leakage —
no post-outcome column reaches the design matrix, and `inventory_competition` was
already redefined to count entries rather than overlapping live intervals.
Integer tolerance (no fractional binaries; `Σ y[i,j,k] ≤ 1` holds exactly).
PuLP variable naming (indices, not user strings; no collisions on the tested
inventories). Infeasibility diagnostics (correctly names `cash_flow_floor` on a
deliberately infeasible instance). Monte Carlo convergence — P5's standard error
across 20 independent seeds is 0.90% of the reported P5–P95 spread, so 10,000
draws is adequate. Proportional hazards holds for `rel_price_premium` on
synthetic data (Schoenfeld p = 0.76). Collinearity is mild (condition number 29.8;
max VIF 7.33 on `living_area_sqft`/`beds`, which are genuinely related). No
hardcoded estimated parameters in the modelling path — the numeric literals found
are reporting thresholds and bounds, which `AGENTS.md` permits.

**Inspected but could not verify.**
- Whether the 5,000-row truncation removed 10 expired listings or 10,000. The
  search's reported total is not in the file; only the broker can supply it.
- Whether `Association Fee` on this export is monthly or annual. There is no
  `AssociationFeeFrequency` column, so the 238 rows flagged implausible cannot be
  rescaled on evidence. `Maintenance Charge/Month` is present on 615 rows and
  states its frequency; the mapping prefers the denser `Association Fee`, which is
  defensible but unverified.
- The true elasticity's magnitude. Three separate attenuation mechanisms are
  identified and two are quantified in direction only; I cannot bound the total.
- Whether `Unit Floor Location > Total Floors In Building` on 152 rows means the
  floor is wrong or the building height is. The sanity gate nulls the floor,
  which is the safe direction, but the opposite reading is equally consistent.

**Not reached.**
- **Phase 8 (auth, multi-tenancy, Stripe).** Not built, and `PROJECT_BRIEF.md` §7
  puts it out of scope. There is consequently *no* authentication on any endpoint
  and *no* rate limiting on `/api/simulate` or `/api/optimize`, both of which are
  expensive. This is acceptable for a localhost prototype and would be a critical
  finding the moment it is exposed. I did not build it, per §10.
- **Phase 5 backtest.** There is no backtest module in the repository. Nothing
  labels anything a backtest, so there is no false claim to correct — but §5 of
  the brief had nothing to audit.
- **The Bucharest path.** `config/bucharest.yaml` is a 27-line stub with no
  pipeline. I verified no Miami-fitted coefficient can leak into it (there is no
  code path that would load one) but did not exercise it.
- **Browser-level frontend testing.** I audited the React source and confirmed
  `npm run build` succeeds, but did not run the UI against a live backend or test
  keyboard/screen-reader behaviour. Chart labels were audited by reading; the
  rendered output was not viewed.
- **Dependency vulnerability scan.** Not run; no lockfile-based scanner is in the
  locked stack and §10 forbids adding dependencies.
- **Git history secret scan.** The working tree is not a git repository, so there
  is no history to scan. `.env` currently holds live FRED, BLS, and OpenRouter
  keys and *is* covered by `.gitignore`. I did not rotate them. Note that
  `.gitignore` covers `.env` but not a file named `env`; if these keys were ever
  present under that name they should be treated as exposed.
- **`PRICING_MODEL_export.csv`** is under `backend/data/raw/`, which `.gitignore`
  covers. It is licensed MLS data with addresses; it should not reach a public
  repository if this work is published for an application.

**Where I ran out of depth.** I did not attempt a formal quantification of how
much the terminal-date sampling window (D1) attenuates `β_price`. Doing it
properly needs either an export drawn on list date to compare against, or a
structural model of the selection, and I judged an unvalidated correction factor
more dangerous than a stated bound. I also did not audit the `crowding.py`
correction beyond reading it and confirming it is applied post-solve and labelled
as an assumption — it is opt-in, off by default, and correctly described in the
README as not fixing the independence problem.

---

## 9.7 What a competent skeptic would still attack

1. **"Your elasticity is −0.19 and you are recommending prices with it."** The
   sharpest attack, and it lands. Below about −0.5 the expected-revenue objective
   is close to monotonically increasing in price, and two thirds of units land on
   their ceiling. The honest position is that this system currently demonstrates
   a method and cannot yet price a building. I have made the diagnostics say so;
   I have not made the coefficient larger, and no fix in this audit should be
   read as trying to.

2. **"How do you know the 661 recovered list dates are right?"** The identity is
   validated at 90–94% exact on the statuses where all three fields are present,
   but it is *not* validated on the statuses where it is actually used — PENDING
   and WITHDRAWN have no reported list date to check against, which is the whole
   problem. The assumption is that Matrix computes DOM the same way for every
   status. That is plausible and unverified. A skeptic should ask for the fit
   with and without derived rows; the `list_date_source` marker makes that a
   one-line filter, which is exactly why it exists.

3. **"Your plausibility band is a judgement call that changed the support."**
   True. $20,000/sqft is a chosen number. I justified it from the export's own
   distribution (a gap between $6,090 at p99.9 and $53,866 at p99.99) and
   verified the genuine $10,145/sqft trophy listing survives, but a different
   band would catch a different set. The defence is that the two rows caught are
   1-bedroom listings at $34M and $49.5M, which are arithmetic errors under any
   band anyone would choose.

4. **"P(plan > baseline) = 75% is not 100% — so is it or isn't it a tautology?"**
   In expectation the optimum weakly dominates the baseline by construction. The
   75% figure is below 1 only because both arms sample a Bernoulli sale lottery
   and 60 units is lumpy. Neither number is evidence about the market. The paired
   uplift band containing zero is the more informative statement, and it is now
   what the dashboard shows.

5. **"You changed two test thresholds. How is that not moving the goalposts?"**
   The fair challenge. For the recovery tolerance I showed with a 24-seed sweep
   that the estimator is unbiased (t = 0.21) and the old threshold failed half of
   all seeds — the threshold was measuring the seed. For the ceiling alarm I
   *tightened* it. For the building-FE test I pinned the configuration rather
   than lowering the bar, and the documented 74% is reproduced. But a reviewer is
   entitled to check each one, and the reasoning is written into the test
   docstrings rather than only here.

6. **"Nothing validates the demand model out of sample."** Correct. Concordance
   is 0.66 on real data and there is no held-out period, no backtest, and no
   realized-outcome comparison. Every "expected revenue" figure is an artifact of
   a model whose predictive accuracy has been reported but never independently
   tested.

7. **"Why should a Miami resale elasticity price a Bucharest tower for One
   United?"** It should not, and nothing in the code lets it try — but the
   distance between what the data measures (individual resale sellers, Miami-Dade,
   ≥$400k, 2024–2026) and what the tool is aimed at (a listed Romanian developer's
   new-construction phased releases) is larger than any statistical caveat in this
   report. D3 is the limitation that should lead any write-up.
