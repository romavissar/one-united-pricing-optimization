# FIXES_AND_USABILITY.md

What the three remediation batches changed, and what still stands between this
and a tool someone can price real inventory with.

Written 2026-09-06, covering commits `cbe8321` (Batch A + B) and `0deb2fc`
(Batch C). It supersedes the numbers in `audit/REMEDIATION_REPORT.md` §7, which
were measured before Batch B moved them.

**Scope.** This describes the pipeline as of `0deb2fc` plus one follow-up fix
(§3.5). It does **not** cover the uncommitted macro-scenario work in the working
tree (`backend/src/data/macro.py` and its API/frontend changes), which I did not
write and have not audited — see §8.

---

## 0. Where things stood before these batches

The audit (`audit/AUDIT_REPORT.md`, findings F01–F23) and the first remediation
pass had already done the heavy work: the calibration banner was fixed, the
panel was no longer selected on the outcome, `pd.NA` stopped leaking through
every null guard, and the identifying variable was respecified from a
submarket-month median to a hedonic residual on a 15-quarter, 48,206-row re-pull.

That left a residue of twelve known-but-unfixed items, catalogued in
`PLAN_FIX.md`. Three of them were not cosmetic. The batches were ordered so the
cheap ones landed first and the ones that could move `β_price` landed last, each
with before/after reported.

---

## 1. Batch A — the tidy-up (items 5, 7, 8, 10, 11)

Five items, none of which changed a number. Grouped together because each was a
place where the code said something that was no longer true.

- **Dead HOA code** (item 7). `_HOA_PSF_MONTHLY_BOUNDS`, `hoa_units_suspect`,
  and the `HOA UNITS SUSPECT` warning could no longer fire once the plausibility
  band was removed (Association Fee is confirmed monthly). The report printed
  `hoa_per_sqft implausible (nulled): 0` every run, which reads as a measurement
  but was an unreachable branch. Deleted, replaced with a `hoa_negative` counter
  that can actually fire.

- **Stale `PRICING VARIATION TOO WIDE` explanation** (item 5). The warning text
  described the old cell-median variable. Suppressed under the residual spec,
  kept intact for the cell-median path.

- **First-stage hedonic absent from bundle metadata** (item 8). The premium
  model is now persisted with the bundle, so a saved model records the surface
  its coefficient was measured against.

- **pandas `FutureWarning`** (item 10). `clean.py`'s `flag.fillna(False).astype(bool)`
  → `flag.eq(True)`. Same answer, and it will not become an error.

- **A false paragraph in `REMEDIATION_REPORT.md` §4** (item 11), which claimed
  the MLS data was untracked. It was not. Corrected.

---

## 2. Batch B — the two that mattered (items 1, 2, and 3 as a consequence)

### 2.1 `view_description` entered the Cox as a 1,139-level categorical

The single worst defect in the register. `view_description` is a multi-valued
field — `"Ocean View, Bay, Skyline"` — and it was being one-hot encoded on its
*raw concatenated string*. That produced 1,139 levels, of which 694 were pooled
into a `__rare__` bucket, consuming **420 of 443 design columns** while the 16
atomic `view_*` indicators sat unused beside them.

Three things went wrong at once. The design matrix was mostly noise; listwise
deletion cost rows it did not need to; and the encoding was recomputed from the
data at scoring time rather than replayed from the fit, which is the same class
of defect already fixed once in `premium.py`.

Fixed by routing view through the atomic indicators. Design matrix **443 → 81
columns**, no source field contributing more than 13, and 1,356 more rows
surviving deletion. `β_price` moved −0.4510 → −0.4310 with controls.

One complication worth recording: adding the view tokens made the design
singular, because synthetic view is single-valued and the indicators summed to
1 — a dummy-variable trap. The fix drops the most frequent token as a reference
level whenever the indicators form a complete partition.

### 2.2 `premium_model` travelled on `frame.attrs`, which `pd.concat` drops

`β_price` is a coefficient on a hedonic residual. Computing that residual
requires the fitted hedonic. It was being passed via `frame.attrs`, which does
not survive `pd.concat` — and when it went missing, the code **silently fell
back to the submarket median**.

That is the dangerous failure mode: wrong units, entirely plausible numbers, no
warning. A caller who concatenated two frames would get a coefficient measured
against a quantity it was never fitted on and no way to notice.

`premium_model` is now returned on `FeatureResult` and passed explicitly, and
fitting a residual-spec frame without one raises `IdentificationError`.

A third instance of the same encoding-divergence class surfaced while testing:
an inventory carrying fewer view tokens than the fit was raising. That one is
*not* a bug — an unset view is a fact, not a gap — so absent binary indicators
now default to 0, while a missing continuous covariate is still a hard error.

### 2.3 Item 3 closed as a side effect

The premium reference and the comps band had been incomparable: the demand model
was fitted with waterfront / parking / restriction blocks the inventory cannot
supply. With the design cleaned up, band overlap went **0/60 → 60/60 units** and
the reference price moved $605 → $1,213/sqft. Every price in a generated plan
now sits inside the fitted range, and the extrapolation guard reports so.

This retires the fourth "publishable statement" in `REMEDIATION_REPORT.md` §7 —
recommended prices *are* now verified against the range the coefficient is
evidence about.

---

## 3. Batch C — the assumptions that were never tested (items 4, 6, 9)

### 3.1 `CANCELED` was hardcoded as censored across 27% of the sample

Not a bug — `censored` is defensible — but it was a *choice* presented as a
constant, on a quarter of the data, with nothing showing what it was worth.

`defaults.cancelled_treatment` now selects `censored | excluded | event`,
validated in `config.py` rather than silently defaulted (a typo here would move
every coefficient with no signal). `excluded` marks rows in `normalize` and
drops them in `clean_mls`, so the count lands in the filters waterfall instead
of the rows quietly vanishing.

`python -m src.demand.diagnostics --cancelled-sweep` refits all three:

| Treatment | Rows fitted | Events | `β_price` | 95% CI |
|---|---|---|---|---|
| `censored` *(in force)* | 44,591 | 16,176 | **−0.4310** | [−0.4795, −0.3824] |
| `excluded` | 32,562 | 16,176 | −0.3994 | [−0.4474, −0.3513] |
| `event` | 44,591 | 29,070 | −0.2336 | [−0.2659, −0.2012] |

**The sign and the exclusion of zero survive every coding.** That conclusion does
not rest on the assumption. The magnitude moves by a factor of 1.8, so any
statement about the *size* of the elasticity has to name the treatment it
assumed. Written into `MLS_SCHEMA.md` §3.

### 3.2 `Last Status` was 87.7% filled and read by nothing

It sat in `_unmapped` with no documented semantics, which meant nobody could
tell whether the event coding was reading the wrong column.

`audit/r04_last_status.py` settled it against the hard evidence of a closing date:

| Column | Agrees with `Closing Date` on a sale | Claims a sale with no closing date |
|---|---|---|
| `Status` | 16,014 / 16,014 | 0 |
| `Last Status` | 1 / 16,014 | 0 |

`Last Status` is the *previous* status. Every `Closed` row's is `Pending`
(10,744) or `Active With Contract` (4,419) — the state immediately before
closing. Coding outcomes from it would have censored essentially every completed
sale.

Mapped to canonical `last_status` so it is no longer undocumented, with
`MLS_SCHEMA.md` §3 recording that it deliberately never codes an outcome, and a
test asserting `event_sold` never depends on it.

One incidental finding, kept for whoever revisits §3.1: among Cancelled
listings, 11,101 have a `Last Status` of `Active` (a plain withdrawal) while 338
reached `Pending` (246) or `Active With Contract` (92) before cancelling — a
contract that fell through, which is a different event from a unit the market
ignored. Available, not yet wired in.

### 3.3 Building support was invisible

`--building-fe` is a recommended calibration command, and on the old
single-quarter export the median building carried **one** listing — a fixed
effect that fit its own row and absorbed nothing (audit finding F09).

The ingest report now prints the cardinality. On the quarterly pull: **9,360
named buildings, median 2 listings each, 2,204 with ≥5 listings covering 35,074
rows (74%)**. `--building-fe` now has real within-building support. That was
true before this change and nobody could see it.

### 3.5 A stale caveat that contradicted the plan it was attached to

Found while assembling `IMPORTANT_CAVEATS.md`, after Batch C. The F11
remediation moved revenue discounting from the phase release date to expected
sale time, and the tensor emits a note saying so. But `_caveats` in `solve.py`
kept emitting the pre-F11 text unconditionally, so **every plan carried both**:

> revenue is discounted to expected sale time, not to the phase release date

> the whole of a unit's expected revenue is discounted to its phase's release
> date rather than to when the cash actually arrives

A reader has no way to tell which is true, which is worse than either statement
alone. The caveat is now conditional on whether arrival discounting actually
applied, and where it did, it names the approximation that genuinely remains —
the horizon cap — instead of one that no longer exists. Tested both ways.

### 3.4 A note Batch B had quietly inverted

Found while doing the above. The attenuation warning matched `view_description`
by column name — but after Batch B, view enters the design as fifteen tokenized
indicators. So the report was printing *"this export carries `view_description`
but the fit does not control for it"* while view was fully controlled.

That note exists to warn that `β_price` is attenuated by a **missing** control.
A false positive inverts its meaning. Fixed via `_is_controlled`, which resolves
tokenized proxies, with a test. It now correctly names only `building_name`.

---

## 4. Where the numbers landed

| Quantity | Before audit | After Batch C |
|---|---|---|
| `β_price` (with controls) | −0.1442 | **−0.4310** (95% CI [−0.4795, −0.3824]) |
| Quality contamination of the identifying variable | 37.2% | **3.1%** |
| Design matrix columns | 443 | **81** |
| Rows fitted | 3,738 | **44,591** |
| (submarket × month) cells with n ≥ 8 | — | 534 of 601 (88.9%), median cell 48 |
| Premium band overlap with comps | 0/60 units | **60/60** |
| Cox concordance | — | 0.6217 |
| Logistic AUC (random holdout) | — | 0.6496 |
| Max calibration decile gap | — | 0.0195 |
| Tests | 189 | **198** |

Bootstrap SE 0.0301 against a conventional 0.0249 — a ratio of 1.209, which is
the price of the identifying variable being itself estimated.

Every MEDIUM+ finding across all three sessions has a regression test. The full
suite passes, `--inspect` fires no new warnings, and the two guards that would
indicate a regression in sampling (`export_sorted_by_status`, terminal-date
selection) remain silent.

---

## 5. Is it usable? — the honest answer

**Not yet for pricing real inventory. Yes as an analysis instrument.**

The engineering is sound. What is not yet established is that the number it
optimizes describes the market it is pointed at. Four things stand in the way,
and only one of them is about code.

**1. `β_price` is below the threshold the tool needs.** At −0.4310 the estimate
sits just outside the specification's plausible band of [−0.5, −3.0], though its
CI now reaches −0.48. The consequence is visible in the output: **100% of the 60
example units land at the price ceiling.** At this elasticity, expected revenue
is close to monotone in price over the whole ladder, so the optimizer is not
really choosing prices — it is reporting "price at the top of the band," and the
band comes from the hedonic surface, not from the elasticity. The MILP is
correct; it just has little to do at this β.

**2. Nothing has been validated out of time.** Concordance 0.6217 and the
calibration table look healthy, and the logistic AUC of 0.6496 is a genuine
holdout — but a *random stratified* holdout, not a temporal one. No model here
has ever been asked to predict a quarter it did not see. That is the cheapest
credibility gap to close and the most conspicuous one to a reviewer.

**3. The resale → new-construction transfer gap is unmeasured.** The coefficient
comes from resale listings by individual sellers. The tool prices
new-construction developer inventory sold in phases. Different buyers, different
financing, different price discovery. This is the largest limitation in the
system and the remediation report is right that *more MLS data does not fix it* —
but it currently sits as a caveat when it could be an estimate.

**4. 33.4% of probability mass is unmodelled** — 20 of 60 units, $41.1M of ask,
never assigned an outcome. This is surfaced correctly (finding F10) rather than
hidden, and the paired uplift band (P50 +$5.66M, 90% band −$4.43M to +$13.40M,
P(beat comps) = 83.3%) **contains zero**. The tool says so. But a plan whose
headline gain is not distinguishable from repricing at comps is not yet a
decision aid.

Separately, and absolutely: **there is no authentication and no rate limiting on
any endpoint**, including `/api/simulate`, which runs 10,000 optimizer
evaluations per request. Fine on localhost. Critical the moment it is exposed.

---

## 6. What to do to make it usable

Ordered by value per unit of effort. The first three are the ones that change
whether anyone should believe the output.

### 6.1 Backtest out of time — *highest value, lowest cost*

Fit on Q1 2023 – Q4 2025, predict Q1–Q3 2026, compare predicted against realized
sale rates and times to sale. The data spans 2023-01-01 to 2026-07-31, so the
split costs nothing to construct.

This answers the question every reviewer asks first and no current diagnostic
addresses. If predicted and realized absorption agree out of time, most of the
other objections get much smaller. If they do not, that is the finding, and it is
better found now.

*Estimate: half a day. Add as `audit/r05_temporal_backtest.py` plus a diagnostic
check.*

### 6.2 Bound the transfer gap empirically instead of caveating it

The feature frame already carries `is_new_construction`: **3,739 rows with 592
sales.** Fit `β_price` on that subsample and compare it to the resale estimate.

592 events will give a wide interval, and that is fine — a wide empirical bound
is a categorically different object from "the size of the gap is unknown." If
the new-construction β is indistinguishable from the resale β, the tool's core
assumption has support for the first time. If it is materially different, that
difference is the correction factor.

*Estimate: half a day. Report both in `diagnostics`, the same way
`--cancelled-sweep` reports the three codings.*

### 6.3 Fit the repeat-listing (within-unit) estimator

Within-unit price variation is the one identifying source in this data that is
**not contaminated by quality at all** — the unit is literally the same unit. The
audit named this the strongest candidate for the next increment when the old
export had 232 relisted units. The quarterly pull has far more:

- **7,969 `unit_key`s appearing 2+ times**, spanning **19,097 rows**
- 2,137 appearing 3+ times
- **3,768 sales** within the repeat set

That is a real panel. A stratified Cox with a per-unit stratum, or a
within-`unit_key` fixed-effects specification, gives a `β_price` whose remaining
bias is from time-varying unobservables only. If it comes back materially larger
in magnitude than −0.43, the errors-in-variables attenuation diagnosis is
confirmed and the tool has an elasticity it can actually price on.

*Estimate: 2–3 days. This is the one most likely to move `β_price` past −0.5.*

### 6.4 Reframe what the product claims, now

Independent of the above, and doable immediately. At the current β the honest
description is:

> a comps-band and absorption-timing tool, with an elasticity adjustment that is
> directionally established but too weak to drive price selection

not an elasticity optimizer. The 100%-at-ceiling result should be presented as
what it is — the band is doing the work — rather than as an optimization
outcome. The 10 caveats already attached to each plan are good; the headline
framing has not caught up with them.

*Estimate: an afternoon, mostly frontend copy and the plan summary.*

### 6.5 Before any exposure beyond localhost

Authentication, rate limiting, and a request budget on `/api/simulate`.
Deliberately unbuilt so far, and correctly so — but it is a hard gate, not a
backlog item.

### 6.6 Lower priority

- Wire the 338 Cancelled-after-contract listings (§3.2) into the
  `cancelled_treatment` decision — they are a different event from the 11,101
  plain withdrawals and are currently pooled with them.
- Run `--building-fe` seriously now that §3.3 shows it has support, and report
  whether absorbing 2,204 building effects moves `β_price`.
- Cross-elasticity in the optimizer (units within a project compete; the model
  treats them as independent).
- The Bucharest pipeline is a config stub.

---

## 7. The one open decision

**Item 0 in `PLAN_FIX.md`, unresolved and deliberately untouched.**

Licensed MIAMI Association of Realtors MLS data — 48,206 listings including
street addresses, obtained under a broker's membership — is committed and
**pushed to `origin/main`** at `github.com/romavissar/one-united-pricing-optimization`
via commit `ab0d20d`. Roughly 62 MB across `data/` (15 CSVs),
`backend/data/raw/mls/` (15 identical copies), and
`backend/data/raw/mls_old/PRICING_MODEL_export.csv`.

`.env` is **not** tracked; no secrets leaked. `.env.example` was made untracked
by `ab0d20d` and should be restored.

Three options, none of which I have taken:

1. **Purge** — `git filter-repo`, force-push, and ask GitHub Support to expire
   cached refs. Most thorough; rewrites history and breaks any existing clone.
2. **Make the repository private.** Fastest mitigation, does not remove the data
   from history.
3. **Accept**, if the broker's MLS terms permit redistribution at this scope.
   Worth actually reading the terms rather than assuming either way.

This needs a decision from you before anything else in this file matters for a
repository anyone else can see. `AUDIT_BRIEF.md` §7.4 flags exactly this
scenario.

---

## 8. Uncommitted work I did not audit

While these batches were running, a substantial macro-scenario feature appeared
in the working tree and is **not committed**:

- `backend/src/data/macro.py` (901 lines) — derives Monte-Carlo channel
  dispersions and correlations from FRED and BLS series rather than taking typed
  assumptions
- `backend/tests/test_macro.py` (16 tests, network-mocked)
- a `macro:` block in `config/miami.yaml` listing series ids
- changes to `routes.py`, `schemas.py`, `services.py`, `frontend/src/api/client.js`,
  `README.md`, `MASTER.md`
- an autouse `MACRO_DISABLE_NETWORK` fixture in `test_api.py` keeping route tests
  hermetic

It looks deliberate and well-constructed, and I left it untouched. But **none of
it is covered by the audit, the remediation, or these three batches.** It
introduces the system's first outbound network dependency and moves scenario
assumptions from user input to derived data — both of which change what the
caveats in `IMPORTANT_CAVEATS.md` need to say. It should get its own pass before
anyone relies on it.

Test counts: 198 with macro excluded, 214 with it. The table in §4 quotes 198,
which is the number these batches are responsible for.
