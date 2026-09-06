# PLAN_FIX.md — open defects and how to close them

State as of `ab0d20d`. Every item below was verified against the code, not
carried over from a doc. Ordered by what I'd do first, not by severity.

**Blocked on a decision from Rom:** item 0. Everything else I can execute.

| # | Item | Sev | Effort |
|---|---|---|---|
| 0 | Licensed MLS data pushed to GitHub | **CRITICAL** | decision pending |
| 1 | ~~`view_description` as a 1,139-level categorical in the Cox~~ **DONE** | HIGH | 45 min |
| 2 | ~~`premium_model` lost silently via `frame.attrs`~~ **DONE** | HIGH | 45 min |
| 3 | ~~Premium reference vs comps band incomparable (R2)~~ **DONE — resolved by items 1+2** | HIGH | — |
| 4 | CANCELED treatment hardcoded, never swept | MEDIUM | 1 h |
| 5 | ~~Stale `PRICING VARIATION TOO WIDE` explanation~~ **DONE** | MEDIUM | 15 min |
| 6 | `Last Status` unmapped, semantics undocumented | MEDIUM | 30 min |
| 7 | ~~Dead HOA code + misleading report line~~ **DONE** | LOW | 15 min |
| 8 | ~~First-stage hedonic absent from bundle metadata~~ **DONE** | LOW | 20 min |
| 9 | Building-count reporting in ingest | LOW | 20 min |
| 10 | ~~pandas `FutureWarning` at `clean.py:162`~~ **DONE** | LOW | 2 min |
| 11 | ~~`REMEDIATION_REPORT.md` §4 paragraph now false~~ **DONE** | LOW | 5 min |

---

## 0. Licensed MLS data on GitHub — needs Rom's decision first

`ab0d20d` ("Track MLS and pipeline data") removed the `/data/` ignore rule I
added in `3d47069` and committed ~62 MB: `data/` (15 CSVs), an identical copy
in `backend/data/raw/mls/` (15), and `backend/data/raw/mls_old/PRICING_MODEL_export.csv`.
Pushed to `origin/main`. 48,206 MIAMI Association of Realtors listings with
street addresses, obtained under a broker's membership. `AUDIT_BRIEF.md` §7.4.

`.env` is **not** tracked — no secrets leaked.

**Options, Rom picks one:**
- **(a) Purge.** `git filter-repo --path data/ --path backend/data/raw/ --invert-paths`,
  restore the ignore rules, force-push, then ask GitHub Support to expire cached
  refs. Rewrites shared history.
- **(b) Make the repo private** and leave history alone. Fastest; the data stays
  in history forever.
- **(c) Accept**, if the broker's terms actually permit redistribution.

Do not proceed on my own judgement — this is his data and his relationship.
Separately, the same commit changed `!.env.example` to `.env.example`, so the
template is now untracked; restore it whichever option is chosen.

---

## 1. `view_description` enters the Cox as a raw 1,139-level categorical

Measured: **420 of 443 design columns**, 694 levels pooled into `__rare__`,
while the 16 tokenised `view_*` indicators built in `features.py` go unused.
Same defect class already fixed in `premium.py`. On the `--controls` path that
`CALIBRATION_READY.md` §3 tells the operator to run.

- `survival.py`: drop `view_description` from `CONTROLLED_COVARIATES` and
  `CONTROLLED_CATEGORICALS`; add the `view_*` token columns instead.
- `available_covariates` must select token columns by prefix, since they are
  generated at feature-build time rather than named in a constant.
- Verify: design columns ~443 → ~40; report β_price before/after.
- Test: assert **no single source field contributes more than 20 design
  columns** — guards the class, not this instance.

## 2. `premium_model` travels via `frame.attrs`, which `pd.concat` drops

Verified: survives copy/slice/`reset_index`/`iloc`, **not** `concat`. If it is
lost, `build_revenue_tensor` silently falls back to the submarket median while
`β_price` is a coefficient on a hedonic residual — wrong units, plausible
numbers, no warning.

- Return it on `FeatureResult` (already a dataclass) and pass explicitly:
  `CoxDemandModel(premium_model=...)`.
- Keep `attrs` only as a fallback.
- **Guard:** in `CoxDemandModel.fit`, if the frame reports
  `rel_price_premium_spec == "hedonic_residual"` and no premium model is
  present, **raise** `IdentificationError`. Silent fallback is the failure mode.
- Touches `features.py`, `survival.py`, `fit.py`, `api/services.py`.
- Test: build features → `pd.concat` → fit must raise, not fall back.

## 3. Premium reference vs comps band do not overlap (R2) — **RESOLVED**

Closed as a side effect of items 1 and 2, without needing approach (a). The
cause was narrower than diagnosed: the demand model had been fitted with
`wf_`/`park_`/`rest_` amenity blocks the inventory cannot supply, so scoring an
inventory unit held them at fit-time means and dragged the reference down.
Restricting the demand model's tokens to `view_` — the one block
`inventory_scoring_frame` can produce — and tokenising `view` in that shim
fixed it. Measured: band overlap **0/60 → 60/60 units**, median reference
$605 → $1,213/sqft against a ladder of $965–$1,457, and the plan now reports
*"All prices imply a rel_price_premium within [-0.280, +0.381], well inside the
fitted range."* R² of the first stage is untouched, so nothing was traded away.

The original plan is kept below for the record.

### Original plan (not executed)

The invariant broken: *the premium must be computed the same way at fit time
and at scoring time.* The price ladder comes from the close-price hedonic
(covariates the inventory carries); the reference comes from the list-price
hedonic (MLS amenity vocabulary the inventory lacks), so scoring an inventory
unit falls back to fit-time means. On the example inventory the bands do not
overlap **for any unit**, the clamp stands down, and all 60 units land on their
price floor.

**Approach (a) — restrict the premium hedonic to inventory-supplied covariates.**
- Declare the scorable set explicitly, matching what `inventory_scoring_frame`
  can produce: `log_living_area`, `log_floor`, `beds`, `baths_full`,
  `hoa_per_sqft`, `building_age_years`, `total_stories`, `submarket`,
  `list_month`, plus view tokens **only when the inventory supplies a view**.
- Fit one surface on that set. Reference and residual then come from the same
  fit by construction.
- **Must report:** R² before/after (63.0% now), residual sd (0.331 now),
  β_price before/after (−0.4233 naive / −0.4510 controlled now), and the
  overlap count (0/60 now).
- Expect R² to fall and β to shrink somewhat — that is the honest cost of
  measuring the premium against something the inventory can actually be scored
  on. Report it; do not tune to avoid it.
- **Escalate rather than choose** if β drops below roughly −0.30: at that point
  the trade-off is a specification decision (accept a weaker coefficient, or
  extend the inventory schema so developers supply view/waterfront/parking).
- Test: assert reference and ladder bands overlap for ≥90% of a standard
  inventory, and that the clamp actually engages.

## 4. CANCELED is hardcoded as censored and never swept

12,894 rows — 27% of the sample — bucketed as censored at `ingest_mls.py:232-237`
with no config knob anywhere. Unlike EXPIRED, a cancellation is ambiguous: many
are relists or agent changes, not a unit the market refused. The audit made me
quantify the PENDING choice; nobody made the same argument here.

- Add `cancelled_treatment: censored | excluded | event` to `config/miami.yaml`
  under `defaults`, default `censored` (current behaviour, no silent change).
- Thread into the event coding in `normalize.py` where `event_sold` is set.
- Have `diagnostics.py` report β_price + CI under **all three** so the choice is
  visible rather than assumed.
- Test: the three settings produce three different event counts and the default
  reproduces today's β_price exactly.

## 5. `PRICING VARIATION TOO WIDE` now gives a false explanation

Still fires (IQR 0.377 > 0.35) saying *"A (submarket, month) cell this dispersed
is not holding the unit fixed"* — but the variable is a hedonic residual now.
There is no cell. Correct number, wrong reason.

- Suppress under `rel_price_premium_spec == "hedonic_residual"`; the correct
  diagnostic for the new variable is `quality_explained_share` (5.1%), already
  computed and already reported.
- Keep the warning intact for the cell-median spec.

## 6. `Last Status` unmapped

Confirmed absent from `src/` entirely; falls into `_unmapped` with no semantics.
- Cross-tab against `Closing Date` / `Expiration Date` presence to establish
  whether it is authoritative for outcome coding.
- Then either map it to a canonical field or write down in `MLS_SCHEMA.md` why
  it is deliberately ignored. Either outcome is fine; silence is not.

## 7. Dead HOA code

`_HOA_PSF_MONTHLY_BOUNDS` (`features.py:61`), `hoa_units_suspect` (`:92, :639`),
the `HOA UNITS SUSPECT` warning (`:611-612`) and the report line (`:673`) can
no longer fire since the band was removed. The report prints
`hoa_per_sqft implausible (nulled): 0` every run, which reads like a
measurement and is an unreachable branch. Delete; keep a `negative` counter.

## 8. First-stage hedonic not in bundle metadata

Coefficient covariance is persisted; the premium fit's R² and residual sd are
not, so a saved bundle cannot say what the identifying variable's first stage
looked like. Add a `premium_hedonic` block to `build_metadata` from the
`FeatureReport.hedonic_premium` dict that already exists.

## 9. Building-count reporting

Given D7 (median 1 listing/building on the old export, 94% of towers pooled),
report `unit_key`/`building_name` cardinality and the count of buildings with
≥5 listings in the ingest report, so it is visible whether quarterly pooling
actually improved within-building support — i.e. whether `--building-fe` is
worth running at all now.

## 10. pandas `FutureWarning`

`clean.py:162`: `flag.fillna(False).astype(bool)` → `flag.eq(True)`, already
boolean. Will become an error in a future pandas.

## 11. Correct the stale paragraph

`audit/REMEDIATION_REPORT.md` §4 claims `.gitignore` excludes the repo-root
`/data/`. True when written at `3d47069`, false since `ab0d20d`. Rewrite to
state what happened and what was decided in item 0.

---

## Not in scope here

- **β_price = −0.451**, below the −0.5 gate. Not a bug. The next real gain is
  repeat-listing identification — within-unit price variation with quality held
  exactly fixed, the one uncontaminated source in this data.
- **F07 remainder** — only β_price's dispersion is propagated in the Monte
  Carlo. The bootstrap covers the dominant term; carrying the design tensor for
  Cholesky draws of the full vector is refinement, not correction.
- **No auth or rate limiting** on any endpoint including `/api/simulate`.
  Out of scope per `PROJECT_BRIEF.md` §7, critical the moment it is exposed.

## Execution order

Batch A (quick, no interaction): 10, 7, 5, 11, 8
Batch B (real fixes): 1, 2 — then full suite + `--inspect`
Batch C (changes β_price): 4, 6, then **3**, each with before/after reported
Item 0 whenever Rom decides; independent of the rest.

Re-run after every batch: `pytest -q`, `python -m src.data.ingest_mls --inspect`,
`python ../audit/a12_headline_numbers.py`. Confirm `export_sorted_by_status`
and the terminal-date-selection warning still do **not** fire — they are the
regression guard for the sampling defects the re-pull fixed.
