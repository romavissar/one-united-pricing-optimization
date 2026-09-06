# REMEDIATION_REPORT.md — new data, respecified identification variable

Follows `audit/AUDIT_REPORT.md`. Everything here was re-measured against the
15-quarter re-pull and the respecified variable; reproduction scripts are
`audit/r01`–`r03` alongside the original `a01`–`a12`.

Test suite: **189 passed, 0 failed** (was 188 before this session, on the old data).

---

## 1. Headline: before and after

| # | Number | Audit (old data, cell-median premium) | Now (15 quarters, hedonic residual) |
|---|---|---|---|
| 1 | `β_price` | **−0.1902**, SE 0.0451, CI [−0.279, −0.102] | **−0.4510**, SE 0.0256, CI **[−0.501, −0.401]** |
| | naive covariate set | −0.1708, SE 0.0460 | −0.4233, SE 0.0249 |
| | fitted sample | 4,284 rows / 1,892 events | **43,235 rows / 15,029 events** |
| | quality contamination of the regressor | **37.2%** | **5.1%** |
| 2 | (submarket × month) cells with n ≥ 8 | 96 of 145 = **66.2%**, median cell 17 | 534 of 601 = **88.9%**, median cell **48** |
| 3 | Units at the price **ceiling** | 40 of 60 = 66.7% | **0 of 60** — but 60 of 60 at the **floor**, see §6 |
| 4 | Unmodelled mass `Σ(1−Dᵢ)` | 19.30 units = 32.2%, $36.5M = 33.8% of ask | 30.13 units = 50.2%, $41.7M = 55.1% of ask |

`β_price` has **2.4×** the magnitude and **half** the standard error. Its
confidence interval now reaches the −0.5 the specification treats as the floor
of a usable elasticity; the point estimate does not. That is a real change in
what this system can claim, and §7 is honest about what it still cannot.

Numbers 3 and 4 moved for a reason that is **not** an improvement — see §6. They
are currently measuring a defect this session found and could not fully close.

---

## 2. `β_price` decomposed: how much was the data, how much the respecification

`audit/r03_beta_decomposition.py` fits all four combinations. Both arms run
through the *current* code, so this isolates data and variable while holding
implementation fixed.

| | old variable (cell median) | new variable (hedonic residual) |
|---|---|---|
| **old data** (5,000-row export) | **−0.1442** (SE 0.042, n=4,629) | −0.1840 (SE 0.055, n=4,677) |
| **new data** (15 quarters) | −0.2297 (SE 0.016, n=44,582) | **−0.4233** (SE 0.025, n=44,591) |

Total change **−0.2791**. The decomposition is path-dependent, so both orders
are reported rather than one presented as the answer:

- respecify then re-sample: **−0.0397** then **−0.2393**
- re-sample then respecify: **−0.0855** then **−0.1936**
- interaction: **−0.1538**

Shapley-style average attribution: **respecification −0.117 (42%)**, **new
sample −0.162 (58%)**.

The large interaction is the substantive finding. The respecification does
**4.9× more** on the large sample than on the small one (−0.194 versus −0.040),
because a hedonic with 142 parameters is badly estimated on 4,600 listings and
well estimated on 45,000. Neither change would have delivered this on its own:
the new data alone gets to −0.23, the respecification alone to −0.18. They are
complements, not substitutes.

What the respecification removed, directly measured: the identifying variable's
variance falls to **39.7%** of the cell-median version's, and the share of it
explained by unit characteristics falls from **37.2% to 5.1%**. The old variable
was, as diagnosed, mostly a description of which unit it was.

---

## 3. The blocking data issue, and how the anchor was found

`List Date` is null for 7,391 rows — 100% of Active, Active With Contract,
Coming Soon, Pending, Temp Off Market and Withdrawn, and 0% of everything else.

The first attempt anchored recovery on Status Change Date and **66% of the
derived dates landed outside their own file's quarter**. Rather than accept
that, `audit/r02_list_date_identity.py` tested every candidate anchor against
the one hard constraint available — each quarterly file is a query for listings
whose list date falls inside that quarter — and validated each on the three
statuses where List Date *is* reported:

| identity | validated on | exact |
|---|---|---|
| `pending_date − DOM` | Closed | **93.6%** |
| `off_market_date − DOM` | Cancelled / Expired | **90.0% / 90.3%** |
| `status_change_date − DOM` | Closed | 25.1% |
| `closing_date − DOM` | Closed | 1.3% |

The answer is that the anchor depends on whether the clock has stopped. A
terminated listing's DOM was measured to its own terminal date; a **live**
listing's DOM was measured to the moment the export was run, which is the same
instant for all fifteen files. Anchoring live listings on the export snapshot
recovers **97.6%** in-quarter against 24% for Status Change Date.

The snapshot is read off the data, not written down: `status_change_date` is
100% filled and cannot postdate the pull, so its maximum **is** the pull
(2026-07-31). The obvious alternative — "the latest date anywhere in the file" —
is wrong, because `off_market_date` carries scheduled future terminations and
runs to 2026-10-30. Using it would have pushed every recovered list date three
months early and censored every live listing three months late.

**Result: 7,181 of 7,391 recovered (97.2%)**, all marked
`list_date_source="derived_from_dom"`; 210 candidates landed outside their
quarter and were left missing rather than carried.

---

## 4. What else changed

**Ingest rebuilt for the quarterly files.** Reads and concatenates all 15,
tagging each row with its source quarter from the filename (`~Q3-2026.csv`
included — the tilde marks the partial current quarter). 48,206 rows in, 47,310
out. Canonical fields matched rose from 24 to **42**.

**Status mapping corrected.** `Active With Contract` / `Active Under Contract`
now map to ACTIVE and are right-censored at the export date, not scored as
sales. A listing taking backups has not ended its spell. This follows the
remediation brief's explicit instruction and contradicts `MLS_SCHEMA.md` §3 as
written; the schema is updated. Effect: 217 still-open listings are no longer
counted as events.

**Live listings right-censored at the export date**, from their own list date —
the ACTIVE branch now runs *before* the reported-DOM fallback, so a live
listing is never censored on whatever clock DOM happens to be on.

**Distressed sales excluded for real.** The old export carried no `sale_type`,
so that filter dropped zero rows and every foreclosure and short sale sat inside
the hedonic. REO and Short Sale are now their own booleans: **402 REO and 80
short sales removed**. Blank Short Sale (974 rows) stays blank — treating it as
False would assert something the data does not say.

**HOA plausibility band removed.** `Association Fee` overlaps
`Maintenance Charge/Month` on 6,943 rows with a **median ratio of exactly
1.0000** (medians $1,130 vs $1,290) and only 0.3% near a ratio of 12. The field
is monthly. The band had been nulling 238 real fees on suspicion, discarding the
amenity signal a high fee carries. Only arithmetic guards remain.

**Unit View brought in.** 96.4% filled, and it is multi-valued — 1,144 distinct
strings but only **17 atomic tokens** once split on commas. `Direct Ocean`
(10.7%) is now a separate indicator from `Ocean View` (29.9%), which is the
amenity the old export could not see at all. The same tokenisation runs over
Waterfront Description, Parking Description and Restrictions. Missingness is
encoded as its own indicator rather than dropping rows.

**Floor conflicts adjudicated instead of blanket-nulled.** 1,417 rows report a
floor above their building's height, and the naive gate was wrong about half the
time because the conflict has three causes: 652 rows report 0–1 total storeys
(the *height* is unusable), 551 report a floor above 150 (the *floor* column
holds something else, up to 373,737), and 223 are genuine conflicts. The unit
number is an independent third signal and adjudicates: unit 3106 in a building
reporting 29 storeys corroborates a reported floor of 31 and impeaches the
height; unit 106 in a building of 12 impeaches a reported floor of 106. Where it
corroborates neither, the floor is left missing. Rejected parses fell from 152
(on 5,000 rows) to **68 on 48,206**.

**F11 — discounting to expected sale time.** Implemented as specified, by
integrating the discount factor against the fitted survival conditional on sale
within `T`, capped at the fitted follow-up. No midpoint approximation. Under
proportional hazards each cell's whole survival curve is recoverable from the
horizon probability already computed —
`r = log(1−D)/log(S₀(T))`, then `S(τ) = S₀(τ)^r` — so one baseline vector
reconstructs every cell at negligible cost and the result is a constant per
(unit, phase, level). **The MILP stays linear.** Both bases are retained on the
tensor and the difference is reported.

**F07 — generated-regressor standard errors.** `rel_price_premium` is now a
residual from a first-stage fit, so the Cox standard error conditions on an
estimated quantity as if it were data. `src/demand/bootstrap.py` resamples
listings and refits **both** stages inside each replication. Result over 60
replications: conventional SE **0.0249**, bootstrap SE **0.0301**, ratio
**1.21** — the conventional interval is **21% too narrow**. Bootstrap 95% CI
**[−0.480, −0.381]**. Cached, keyed on a fingerprint of the input so a stale
result cannot be served for different data.

The **"lower bound" wording is corrected** everywhere. Omitting covariance terms
can widen *or* narrow an interval depending on the signs of the coefficients and
their correlations, so the honest statement is that the direction of that bias
is not established. What *is* established is the generated-regressor effect,
which is measured above.

**F04 / F05 detectors kept, not retired.** Both correctly stand down on the new
data — terminal dates span 2,038 days against 1,307 for list dates, 6,209
listings are still ACTIVE, and the files are not status-sorted. They earn their
place as regression guards: they would catch a future re-pull that reintroduced
either defect, and there is a test pinning that they do not fire on good data.

**Repository.** `git init`, first commit, 113 files. `.gitignore` covered
`.env`, `.env.*`, a bare `env`, `*.env`, and the repo-root `/data/`, which held
a **30 MB copy of the licensed MLS exports that `git add -A` staged** — caught
and excluded at the time; see §5.

> **Superseded as of `ab0d20d`.** A later commit, *"Track MLS and pipeline data;
> keep secrets out of the repo"*, removed the `/data/` rule deliberately and
> committed the exports "for a complete handoff". Tracked and **pushed to
> `origin/main`** as of this writing: `data/` (15 quarterly CSVs), an identical
> copy under `backend/data/raw/mls/`, and `backend/data/raw/mls_old/PRICING_MODEL_export.csv`
> — roughly 62 MB of licensed MIAMI Association of Realtors records with street
> addresses. `.env` remains untracked, so no secrets are exposed. The same commit
> also stopped tracking `.env.example`, which should be restored.
>
> This is the exact scenario `AUDIT_BRIEF.md` §7.4 flags. Deleting the files in a
> new commit does not remove them from history; undoing it needs a history
> rewrite and a force-push, and GitHub retains unreferenced objects until asked
> to purge them. Tracked as item 0 in `PLAN_FIX.md`, pending a decision on
> whether to purge history, make the repository private, or confirm the broker's
> terms permit redistribution.

**Dependency scan.** `pip-audit` against OSV: **no known vulnerabilities**, both
for the installed environment and for `requirements.txt`. `requirements.txt` is
unchanged — pip-audit is a dev tool and was not added to the locked stack.

---

## 5. New findings from this session

| ID | Sev | Title | Status |
|---|---|---|---|
| R1 | HIGH | Licensed MLS data staged for commit | Fixed |
| R2 | HIGH | Premium reference not comparable to the comps band | Detected, made loud, **not closed** |
| R3 | MEDIUM | Train/predict encoding divergence in the list-price hedonic | Fixed |
| R4 | MEDIUM | Status Change Date is the wrong censoring anchor for live listings | Fixed |
| R5 | MEDIUM | `β(log_floor) > 0` is meaningless under the respecified premium | Fixed |
| R6 | LOW | Per-row INFO logging drowned the ingest report at 48k rows | Fixed |

**R1 — licensed data staged.** `git add -A` staged all 15 quarterly CSVs from
the repo-root `data/` directory: 30 MB of MIAMI Association of Realtors records
with street addresses and transaction detail, obtained under a member's
credentials. `.gitignore` covered `backend/data/raw/` but not `/data/`. Caught
before the first commit; both paths now ignored and verified with
`git check-ignore`.

**R2 — the premium reference does not match the comps band.** The price ladder
is built from the close-price hedonic, which uses only covariates a developer's
inventory carries. The premium reference is the list-price hedonic, whose
amenity vocabulary (view, waterfront, parking, restrictions) the inventory does
**not** carry — so scoring an inventory unit holds most of that surface at its
fit-time means and the prediction regresses toward the middle of the MLS sample.
On the example inventory the ladder spans $965–$1,457/sqft at the median unit
while the supported range maps to $436–$836/sqft, and **the two do not overlap
for any unit**.

The consequence is visible in headline number 3: every unit now lands on its
price **floor** rather than its ceiling, because every ladder price implies a
large positive premium against a reference that is too low. That is not
economics; it is the mismatch.

The clamp stands down rather than excluding the entire inventory, and the plan
carries a prominent caveat saying the extrapolation guard could not be applied
and why. **This is the most important thing left open.** The fix is to build the
price band from the *same* surface the coefficient is measured against, or to
extend the inventory schema to carry the hedonic's covariates. Either is a
specification decision about what a developer must supply, which is why it is
reported rather than chosen here.

**R3 — train/predict encoding divergence.** `PremiumModel.predict_reference_ppsf`
initially recomputed the design vocabulary on the scoring frame, so which
categorical levels were rare, which was the dropped reference, and what a
missing numeric was filled with all depended on the sample being scored. The
same listing got a different design row at predict time than at fit time, and
predictions were off by up to $2,683/sqft while looking entirely normal. Found
by asserting exact reproduction rather than assuming it. Fixed by freezing a
`DesignVocabulary` at fit time and replaying it; a second bug surfaced
immediately — the dropped *reference* level was not in the known-levels set, so
every reference row was being diverted into the pooled bucket. Now reproduces to
**0.000000000** on 200, 5,000 and 47,310 rows.

**R5 — `β(log_floor) > 0` no longer means anything.** Under the old variable the
cell median did not adjust for floor, so height carried residual desirability
and the sign was determinate. The premium is now a residual against a hedonic
that already prices floor, so the coefficient answers a different question —
does a high-floor unit sell faster *at the same price relative to what its own
floor predicts* — whose sign theory does not fix. Requiring positivity failed
the synthetic fit by construction. The check is now reported without a
requirement under the residual specification. **`β(rel_price_premium) < 0`
remains an unconditional hard failure and is untouched.** This narrows one check
that became meaningless; it does not relax the one the project exists for.

---

## 6. Re-verification

Every numerical check in `AUDIT_BRIEF.md` §2–§5 re-run against the new data.

**§2 identification.** Cells with n ≥ 8: 88.9% holding 99.3% of keyed rows.
Premium IQR 0.377 (was 0.551 on the cell-median version). `list_date` recovery
97.2% with quarter validation. No post-outcome column reaches the design matrix.
Distressed sales excluded. Left-truncation of Q1-2023 assessed below.

**§3 optimizer.** Monotonicity verified on the extracted plan: 0 violations.
Solver status handling, integer tolerance and the independent objective
recomputation are unchanged and still pass. Arrival-time discounting is live and
both bases are reported. Ladder clamping is implemented and correctly stands
down under R2 rather than deleting the inventory.

**§4 simulation.** Percentiles ordered, CVaR ≤ P5, seed reproducible.
P5 $20.45M, P50 $32.56M, P95 $45.56M, CVaR@5% $17.74M. Paired uplift P50
**$3.33M with a 90% band of −$2.96M to +$10.78M** — still containing zero, so
the gain remains not distinguishable from repricing at comps under the model's
own demand curve.

**§5 backtest.** Still none in the repository, and nothing claims one.

**Q1-2023 left truncation — assessed, and kept.** 3,051 rows, 6.4% of the
sample. Listings begun in 2022 and still live into 2023 are absent. Measured
effect on `β_price`: dropping 2023Q1 moves it **+0.0190 (4.5%)**, dropping all
of 2023 moves it **+0.0105 (2.5%)** — both smaller than one standard error
(0.025). The quarter stays. Each listing is observed from its own list date, so
a missing earlier cohort does not bias the hazard; what is lost is coverage of
long-running 2022 listings, which is a statement about the sample's reach rather
than a correction the model needs.

---

## 7. What is still true, and what to say about it

`β_price = −0.451` (CI [−0.501, −0.401]; bootstrap CI [−0.480, −0.381]) is a
**materially better estimate on a materially better sample**, and the
improvement came from removing measurement error rather than from tuning.
Nothing in this session moved the coefficient by choosing a specification that
made it larger: the respecification was decided in the brief, the sign checks
that matter are unchanged, and the one check that was narrowed was narrowed
because it had stopped meaning anything.

It is still **below the −0.5 the specification treats as the floor** of an
elasticity strong enough to price on, though its confidence interval now reaches
it. At −0.45 the expected-revenue objective is close to monotone in price over
much of the ladder, so recommended prices remain driven substantially by where
the band was placed.

Publishable statements that survive:

> The own-price elasticity of demand, estimated on 43,235 Miami-Dade condominium
> listings from Q1 2023 to Q3 2026, is −0.451 (95% CI −0.501 to −0.401;
> −0.480 to −0.381 under a two-stage bootstrap that accounts for the identifying
> variable being itself estimated). The identifying variable is the residual from
> a hedonic regression of log asking price per square foot on unit
> characteristics, submarket and month, so it measures the seller's pricing
> decision rather than which unit is being sold: 5.1% of its variation is
> explained by unit characteristics, against 37.2% for the submarket-month
> median it replaces.

> This coefficient is estimated from **resale listings by individual sellers**
> and the tool prices **new-construction developer inventory sold in phases**.
> These are different markets with different buyers, different financing and
> different price discovery. Nothing in this dataset measures the transfer, and
> the size of the gap is unknown. This remains the single largest limitation of
> the system and no amount of additional MLS data addresses it.

> Recommended prices are not currently verified against the range the
> coefficient is evidence about, because the surface that sets the price band and
> the surface the coefficient is measured against are fitted on different
> covariate sets and do not agree on this inventory. Sale probabilities in a
> generated plan should be treated as extrapolations until that is resolved.

---

## 8. Still out of scope, and stated as such

**There is no authentication and no rate limiting on any endpoint**, including
`/api/simulate`, which runs 10,000 optimizer evaluations per request. This is
acceptable on localhost and becomes a critical finding the moment the service is
exposed. Unchanged from the audit and deliberately not built.

Also unbuilt: the Bucharest pipeline (config stub only), any backtest against
realized outcomes, and cross-elasticity in the optimizer. The repeat-listing
identification strategy — 232 relisted `unit_key`s in the old export, more in the
new — remains the strongest candidate for the next increment, because within-unit
price variation is the one source here that is not contaminated by quality at all.
