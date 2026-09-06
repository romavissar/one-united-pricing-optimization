# IMPORTANT_CAVEATS.md

Everything that qualifies a number this system produces.

Read §1 before quoting any output to anyone. §2 is what the code attaches to
every plan automatically. §3–§5 are the limits that no amount of additional MLS
data removes.

Companion to [`FIXES_AND_USABILITY.md`](FIXES_AND_USABILITY.md), which covers
what was fixed and what to do next. Current as of commit `0deb2fc`, 2026-09-06.

**Scope.** These caveats describe the committed pipeline. They do **not** cover
the uncommitted macro-scenario work in the working tree
(`backend/src/data/macro.py`, plus API, config and frontend changes), which is
unaudited and which changes two things this file asserts: scenario dispersions
would become **data-derived from FRED and BLS** rather than user-supplied, and
the system would acquire its **first outbound network dependency** — with a
documented offline fallback, but a fallback is itself a provenance state that
needs a caveat here. Treat §2 and §5 as incomplete until that work is reviewed.
See `FIXES_AND_USABILITY.md` §8.

---

## 1. The four that decide whether you can quote a number

If you remember nothing else from this file.

### 1.1 The system is not calibrated, and says so

`is_calibrated_on_real_data: false` in `config/miami.yaml`. Every plan renders
behind a non-dismissible "illustrative" banner. **That flag is flipped by a
human after inspecting the fit, never by a successful fit.** It was `true` with
nothing calibrated before the audit (finding F01) — that is the exact failure
this gate exists to prevent.

### 1.2 `β_price` is too weak to price on

**−0.4310** (95% CI [−0.4795, −0.3824]). The specification treats **[−0.5, −3.0]**
as the band where an elasticity is strong enough to drive price selection. This
sits outside it, though the CI now reaches −0.48.

The consequence is visible in every plan: **100% of units land at the price
ceiling.** At this elasticity expected revenue is close to monotone in price, so
the optimizer is not choosing prices — it is reporting "price at the top of the
band," and the band comes from the hedonic surface, not from the elasticity.

The estimate is almost certainly **attenuated** — biased toward zero by
measurement error in the identifying variable — so the true elasticity is larger
in magnitude. That is a direction, not a size. Nothing here tells you how much
larger.

### 1.3 The coefficient is measured on a different market than the tool prices

`β_price` comes from **resale listings by individual sellers.** The tool prices
**new-construction developer inventory sold in phases.** Different buyers,
different financing, different price discovery.

**Nothing in this dataset measures the transfer, and the size of the gap is
unknown.** This is the single largest limitation in the system, and more MLS
data does not address it. (`FIXES_AND_USABILITY.md` §6.2 proposes bounding it
empirically against the 3,739 new-construction rows already in the frame — until
that runs, this is an unquantified assumption, not a small one.)

### 1.4 The headline gain is not distinguishable from doing nothing clever

On the example inventory, paired uplift against simply pricing at comps:

- median **+$5.66M**
- 90% band **−$4.43M to +$13.40M**
- P(plan beats baseline) = **83.3%**

**The band contains zero.** The tool reports this correctly rather than hiding
it. But a plan whose advantage cannot be distinguished from repricing at comps
is not yet a decision aid.

---

## 2. Caveats the code attaches to every plan

These are generated per-run from the actual plan, not boilerplate. The example
inventory produces ten. They are surfaced in the API response (`caveats`) and in
the UI.

**Wherever a number below is inventory-specific, it is quoted from the example
run — your plan's will differ.** The *kinds* are stable.

### Comps and the price band

**[1] The band is today's market value, not a forecast.**
> comps priced as of `list_quarter=2025Q4`, the latest period in the hedonic
> sample. The band is today's market value, not a forecast of the phase date.

A phase releasing in 18 months is being priced against comps from the end of the
fitted sample. There is no price forecast in this system.

**[3] The ladder is clamped to the range the coefficient is evidence about.**
> the price ladder was clamped to the 1st–99th percentile of the fitted premium
> support ([−0.280, +0.381]): 20 of 900 grid points sat outside the range
> `beta_price` is evidence about and would have been extrapolation had the
> optimizer chosen them

This is a guard working as intended. It also means the optimizer physically
cannot recommend a price outside the observed premium range, however attractive
the model thinks it would be.

### Timing and discounting

**[2] Revenue is discounted to expected sale time.**
> revenue is discounted to expected sale time, not to the phase release date:
> across the whole grid that is −2.10% against the release-date basis (horizon
> 180d, capped at the fitted follow-up of 180d)

**[7] What still approximates.** The discount factor integrates against the
fitted survival conditional on selling within the horizon — deliberately *not* a
midpoint approximation, because time-to-sale is right-skewed and the skew itself
varies with price. Two approximations remain: the horizon is **capped at the
longest follow-up the demand model observed**, so a sale arriving later is valued
as if it arrived at the cap; and the arrival distribution is the fitted one, so
it inherits whatever the hazard gets wrong.

> **Note:** before commit `0deb2fc` this caveat still described the pre-F11
> behaviour and contradicted [2] outright — a plan carried both "discounted to
> expected sale time" and "discounted to its phase's release date." Fixed, with
> a test. If you see a plan carrying both, it was generated on an older build.

**[4] The construction gate binds.**
> the construction gate blocks 30 of 240 unit-phase cells (presale lead 24
> months)

### What the formulation does not model

**[5] Sale probabilities are independent across units.**
> Units released together compete for one buyer pool, so phase revenue is
> overstated where a phase releases many near-identical units.

There is no cross-elasticity in the optimizer. Units in one project are treated
as if they do not compete with each other. For a phase of near-identical units,
this is optimistic in a way the model cannot see.

**[6] Expected revenue is a partial sum, not a total.**
> Each unit contributes only its P(sale) share, and the remaining **20.0 units'
> worth of probability mass across 60 released units (33% of the stack)** is not
> carried into a later phase — the formulation releases a unit at most once and
> models no unsold inventory, so that share vanishes. **$41,148,292 of the
> $114,236,415 asking value is therefore never counted.** Do not compare this
> figure to the gross development value of a sold-out project.

The most frequently misread number in the system. Expected revenue of $73.1M
against $114.2M of ask is **not** a $41M loss forecast — it is a third of the
stack having no modelled outcome at all.

**[8] Competitor pricing is exogenous.**
> Nothing here models a rival cutting price in response to this plan; that
> belongs in scenario analysis.

**[9] Demand parameters are fixed across every phase.**
> A real 12–18 month sales period drifts; re-fitting between phases is not yet
> built.

**[10] Everything is at the ceiling.**
> 60 of 60 released units sit at their price ceiling (100%). Charging the
> maximum is what an optimizer does when demand barely responds to price — check
> that `beta_price` is negative and materially different from zero, and that the
> ceilings are real comps rather than an unbounded band, before treating these
> as recommendations.

This alarm fires correctly and should be read together with §1.2. It is the
model telling you that the band, not the elasticity, produced the answer.

### Conditional caveats

Not in the example run, but emitted when applicable:

- **Cash-flow floors bind on *expected* revenue**, so realized cash flow can
  still fall below one. Where the floor is a covenant, simulate the breach
  probability and buffer the floor rather than treating it as guaranteed.
- **Units the demand model could not score are excluded** and counted, rather
  than silently dropped.

---

## 3. What the estimate rests on

### 3.1 The identifying variable is itself estimated

`rel_price_premium` is the **residual from a hedonic regression** of log asking
price per sqft on unit characteristics, submarket and month. It measures the
seller's pricing decision rather than which unit is being sold: **3.1%** of its
variation is explained by unit characteristics, against **37.2%** for the
submarket-month median it replaces.

Because it is a *generated regressor*, conventional standard errors are too
small. A two-stage bootstrap (resample listings → refit hedonic → recompute
residual → refit Cox) gives **SE 0.0301 against a conventional 0.0249** — a
ratio of **1.209**. Quote the bootstrap interval.

Read the 3.1% in one direction only. **A high value would be an alarm.** A low
value is *not* reassurance — it is equally consistent with controls too weak to
explain anything.

### 3.2 The CANCELED coding moves the magnitude by 1.8×

27% of the export is `Cancelled`, and the status is genuinely ambiguous — a
relist under a new agent, a brokerage change, and a seller giving up all look
identical. `defaults.cancelled_treatment` selects the coding:

| Treatment | Rows | Events | `β_price` | 95% CI |
|---|---|---|---|---|
| `censored` *(in force)* | 44,591 | 16,176 | −0.4310 | [−0.4795, −0.3824] |
| `excluded` | 32,562 | 16,176 | −0.3994 | [−0.4474, −0.3513] |
| `event` | 44,591 | 29,070 | −0.2336 | [−0.2659, −0.2012] |

**The sign and the exclusion of zero survive all three.** That conclusion is
robust. **The magnitude is not** — any statement about the *size* of the
elasticity has to name the treatment it assumed. Run
`python -m src.demand.diagnostics --cancelled-sweep` to reproduce.

### 3.3 15% of list dates are reconstructed

`List Date` is null for 100% of six live and off-market statuses. Rather than
drop them — which would select the panel on the outcome — **7,181 of 48,206
rows (14.9%)** have `list_date` derived as terminal date minus days-on-market,
marked `list_date_source='derived_from_dom'`. A further 210 candidates landed
outside their file's own quarter and were left missing.

The derivation is validated against the quarter window, but it is a
reconstruction. Anything sensitive to exact list timing on those rows inherits
its error.

### 3.4 Building fixed effects are available but unused by default

**9,360 named buildings, median 2 listings each, 2,204 with ≥5 listings covering
35,074 rows (74%).** `--building-fe` now has real support — on the old
single-quarter export the median was 1 and it absorbed nothing (finding F09).

The headline `β_price` does **not** absorb building effects. Building-level
unobserved quality therefore remains inside the residual. `building_name` is
also the one hedonic control the export carries that the fit does not use, which
the diagnostics report names explicitly.

### 3.5 Nothing has been validated out of time

Concordance **0.6217**, max calibration decile gap **0.0195**, logistic AUC
**0.6496**. The AUC is a genuine holdout — but a **random stratified** split, not
a temporal one. **No model here has ever been asked to predict a quarter it did
not see.** There is no backtest against realized outcomes.

---

## 4. Sampling and data provenance

- **Old export was truncated on the outcome** (findings F04/F05): sorted by
  status and capped at 5,000 against a "5000+" result, so `Expired` was cut. The
  15-quarter re-pull replaced it; the guards that detect this
  (`export_sorted_by_status`, terminal-date selection) are in place and silent.
- **Distressed sales are excluded** using the REO and Short Sale flags: 402 and
  80 rows respectively. A distressed sale is a different transaction — the price
  is set by a lender payoff and the timing by lienholder approval.
- **13 list and 2 close `$/sqft` values were nulled** as physically implausible
  (outside $100–$20,000/sqft). The rows are kept; only the derived price is
  dropped and marked. Left in, one corrupted area set the premium support to
  +109 and the extrapolation guard could flag nothing (finding F06).
- **`Active` and `Active With Contract` are right-censored at the export date.**
  A contract exists but the listing is still taking backups and the spell has not
  ended. Scoring these as sales would bias the hazard upward.
- **`Pending` is treated as sold** where a pending date exists — the demand event
  is the buyer committing, not the deed recording. Flagged as
  `pending_treated_as_sold` so the choice is auditable.
- **`Last Status` is mapped but never codes an outcome.** It is the *previous*
  status: it matches a closing date on 1 of 16,014 sales where `Status` matches
  all of them. Coding from it would censor nearly every sale.

---

## 5. Scope, and the thing that is not a modelling caveat

### Not built, and stated as such

- **No authentication and no rate limiting on any endpoint**, including
  `/api/simulate`, which runs 10,000 optimizer evaluations per request.
  Acceptable on localhost. **Critical the moment the service is exposed.** This
  is a hard gate, not a backlog item.
- No backtest against realized outcomes (§3.5).
- No cross-elasticity in the optimizer (§2, caveat [5]).
- No price forecasting (§2, caveat [1]).
- No re-fitting between phases (§2, caveat [9]).
- The Bucharest pipeline is a config stub.

### The open decision

**Licensed MIAMI Association of Realtors MLS data — 48,206 listings including
street addresses, obtained under a broker's membership — is committed and pushed
to `origin/main`.** Roughly 62 MB, via commit `ab0d20d`. `.env` is not tracked;
no secrets leaked.

This is not a modelling caveat, it is a licensing and privacy exposure, and it
is unresolved by explicit decision. Options and detail in
[`FIXES_AND_USABILITY.md`](FIXES_AND_USABILITY.md) §7 and `PLAN_FIX.md` item 0.

---

## 6. If you quote one paragraph

> The own-price elasticity of demand, estimated on 44,591 Miami-Dade
> condominium listings from Q1 2023 to Q3 2026, is **−0.431** (95% CI −0.480 to
> −0.382; the two-stage bootstrap that accounts for the identifying variable
> being itself estimated widens the standard error by a factor of 1.21). The
> identifying variable is the residual from a hedonic regression of log asking
> price per square foot on unit characteristics, submarket and month, so it
> measures the seller's pricing decision rather than which unit is being sold:
> 3.1% of its variation is explained by unit characteristics, against 37.2% for
> the submarket-month median it replaces. The sign is robust to how cancelled
> listings are coded; the magnitude is not, moving by a factor of 1.8 across
> defensible codings. The estimate is attenuated by measurement error, so the
> true elasticity is larger in magnitude — by an unknown amount. It is estimated
> on resale listings by individual sellers and applied to new-construction
> developer inventory, a transfer this data cannot measure.
