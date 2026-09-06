# MLS_SCHEMA.md — Canonical schema and normalization contract

Read this before writing `ingest_mls.py` or `normalize.py`.

Broker MLS exports are not standardized. The same field is `Orig List Price` in one
export, `OriginalListPrice` in another, `ORIG_LIST_PRICE` in a third. The normalizer's
job is to map whatever arrives onto the canonical names below, coerce types, and report
honestly on what was missing.

---

## 1. Canonical schema

`REQUIRED` = elasticity cannot be estimated without it. Fail loudly if absent.
`IMPORTANT` = model quality degrades materially without it; warn.
`OPTIONAL` = use if present.

| Canonical name | Type | Tier | Notes |
|---|---|---|---|
| `mls_number` | str | REQUIRED | Primary key for a listing *episode*. |
| `status` | enum | REQUIRED | `SOLD \| EXPIRED \| WITHDRAWN \| CANCELED \| ACTIVE \| PENDING` |
| `original_list_price` | float | REQUIRED | The **first** price the unit was listed at. Not the last. |
| `list_date` | date | REQUIRED | Start of the listing episode. |
| `living_area_sqft` | float | REQUIRED | Interior area. Needed for `$/sqft`. |
| `zip_code` | str | REQUIRED | Drives submarket assignment. Keep as string — leading zeros and ZIP+4. |
| `close_price` | float | IMPORTANT | Final sale price. Null for non-`SOLD`. |
| `close_date` | date | IMPORTANT | Null for non-`SOLD`. |
| `last_list_price` | float | IMPORTANT | Price at sale/termination. Reveals price cuts. |
| `days_on_market` | int | IMPORTANT | MLS-calculated. Used as fallback for `duration_days`. |
| `off_market_date` | date | IMPORTANT | Terminal date for `EXPIRED`/`WITHDRAWN`/`CANCELED`. |
| `beds` | int | IMPORTANT | |
| `baths_full` | float | IMPORTANT | |
| `street_address` | str | IMPORTANT | For `unit_key` when `building_name` is absent. |
| `unit_number` | str | IMPORTANT | Source for floor parsing. |
| `building_name` | str | IMPORTANT | Best grouping key for condo comps. |
| `property_type` | str | IMPORTANT | Filter to condo/co-op. |
| `sale_type` | str | IMPORTANT | Filter out distressed. |
| `unit_floor` | int | OPTIONAL | Reported floor. Frequently blank — hence parsing. |
| `cumulative_days_on_market` | int | OPTIONAL | Spans relists. |
| `pending_date` | date | OPTIONAL | Better sale-timing signal than `close_date`. |
| `baths_half` | float | OPTIONAL | |
| `year_built` | int | OPTIONAL | |
| `new_construction` | bool | OPTIONAL | |
| `total_stories` | int | OPTIONAL | Building height; contextualizes floor. |
| `hoa_monthly` | float | OPTIONAL | Material in FL — insurance-driven. |
| `hoa_frequency` | str | OPTIONAL | Normalize to monthly. |
| `tax_annual` | float | OPTIONAL | |
| `subdivision` | str | OPTIONAL | Neighborhood label. |
| `city` | str | OPTIONAL | |
| `list_agent_name` | str | OPTIONAL | Helps separate builder-direct from resale. |
| `list_office_name` | str | OPTIONAL | Same. |
| `waterfront` | bool | OPTIONAL | View proxy when no view field exists. |
| `view_description` | str | OPTIONAL | **Multi-valued**, comma separated. See below. |
| `status_change_date` | date | OPTIONAL | Last record movement. Its maximum is the export snapshot. Never a terminal date for a live listing. |
| `waterfront_description` | str | OPTIONAL | Multi-valued. `Ocean Front`, `Bay Front`, … |
| `restrictions` | str | OPTIONAL | Multi-valued. Rental rules drive Miami condo value. |
| `parking_description` | str | OPTIONAL | Multi-valued. |
| `amenities` | str | OPTIONAL | Multi-valued, very high cardinality; building-level. |
| `min_lease_days` | int | OPTIONAL | Minimum lease term; a legal attribute of the unit. |
| `furnished_info` | str | OPTIONAL | Part of what is being sold. |
| `special_assessment` | bool | OPTIONAL | Material in FL post-Surfside. |
| `association_type` | str | OPTIONAL | |
| `is_reo` | bool | OPTIONAL | Bank-owned. **Excluded from the fit.** |
| `is_short_sale` | bool | OPTIONAL | **Excluded from the fit.** Blank stays blank. |
| `terms_considered` | str | OPTIONAL | Captured but **deliberately not a hedonic control** — see below. |
| `occupancy_information` | str | OPTIONAL | Same. |
| `special_information` | str | OPTIONAL | Same. |

### Multi-valued fields

`Unit View` and its relatives arrive as comma-separated lists — `"Bay, Skyline
View, Water View"` is three facts about the unit, not one categorical level. As a
category it has 1,144 distinct values on the 2023–2026 pull and is useless; split
into atoms it has **17**, and `Direct Ocean` (10.7% of listings) is a different and
far more valuable thing from `Ocean View` (29.9%). Split on commas, keep one
indicator per token above ~1% frequency, and encode "no value at all" as its own
indicator rather than dropping the row.

### What may not enter the hedonic

The identification variable is the residual from a hedonic on log asking price, so
the hedonic's right-hand side decides what is left in it. Three rules:

1. **Nothing post-listing** — DOM, CDOM, Current Price, Sale Price, and every
   terminal date are functions of the outcome.
2. **Nothing about the seller** — `Terms Considered`, `Occupancy Information` and
   `Special Information` are available at listing time and are not outcome
   variables, but they describe the seller's situation rather than the asset. The
   residual is *supposed* to contain seller behaviour; a control that absorbs seller
   motivation strips out the exact variation `β_price` measures. Excluding them is
   deliberate, and adding them would look like an improvement while making the
   estimate worse.
3. **Fit on every listing, not just sold ones** — this models what sellers ask, not
   what buyers pay. Restricting to closed sales selects on the outcome.

### Derived (computed, never read from source)

| Name | Rule |
|---|---|
| `unit_key` | `slug(building_name or street_address) + "|" + slug(unit_number)` — stable ID for the physical unit across relists |
| `floor` | `unit_floor` if present, else parsed from `unit_number` (§4) |
| `floor_source` | `"reported" \| "parsed" \| "missing"` |
| `is_penthouse` | `unit_number` matches `PH`/`Penthouse` pattern |
| `list_ppsf` | `original_list_price / living_area_sqft` |
| `close_ppsf` | `close_price / living_area_sqft` |
| `submarket` | ZIP → submarket via `config/miami.yaml` |
| `duration_days` | §5 |
| `event_sold` | `1` if `status == SOLD` else `0` |
| `rel_price_premium` | see PROJECT_BRIEF §Phase 2 — the identification variable |

---

## 2. Column alias table

Match **case-insensitively**, ignoring spaces, underscores, hyphens, and periods. So
normalize both source headers and these aliases with
`re.sub(r'[^a-z0-9]', '', s.lower())` before comparing.

This list is a starting set covering RESO standard names and common Matrix/Stellar export
labels. **Extend it when you see the real file** — the last alias in each row is a
reminder that this table is expected to grow.

| Canonical | Aliases |
|---|---|
| `mls_number` | `ListingId`, `MLS #`, `MLS Number`, `MLSNum`, `ML#`, `Listing ID`, `ListingKey`, `ML Number` |
| `status` | `Status`, `MlsStatus`, `StandardStatus`, `Listing Status`, `Stat` |
| `original_list_price` | `OriginalListPrice`, `Orig List Price`, `Original List Price`, `Original Price`, `Orig Price`, `OrigListPrice`, `Original LP` |
| `last_list_price` | `ListPrice`, `Current Price`, `List Price`, `LP`, `Last List Price`, `CurrentListPrice` |
| `close_price` | `ClosePrice`, `Sold Price`, `Sale Price`, `SP`, `Closed Price`, `SoldPrice` |
| `list_date` | `ListDate`, `Listing Date`, `List Date`, `OnMarketDate`, `Date Listed`, `LD`, `ListingContractDate` |
| `pending_date` | `PendingDate`, `Pending Date`, `Under Contract Date`, `PurchaseContractDate`, `Contract Date` |
| `close_date` | `CloseDate`, `Sold Date`, `Sale Date`, `Closing Date`, `SD`, `ClosedDate` |
| `off_market_date` | `OffMarketDate`, `Off Market Date`, `Expiration Date`, `ExpirationDate`, `Withdrawn Date`, `Cancel Date`, `StatusChangeDate`, `Status Change Date` |
| `days_on_market` | `DaysOnMarket`, `DOM`, `Days On Market`, `ADOM`, `AgentDaysOnMarket` |
| `cumulative_days_on_market` | `CumulativeDaysOnMarket`, `CDOM`, `Cumulative DOM`, `CDOM (Days on Market)` |
| `living_area_sqft` | `LivingArea`, `SqFt Living`, `SqFt Liv Area`, `Main Living Area`, `Total Living Area`, `Living Sq Ft`, `SqFtTotal`, `Adjusted Sq Ft`, `BuildingAreaTotal` |
| `beds` | `BedroomsTotal`, `Beds`, `#Beds`, `Bedrooms`, `BR`, `Total Bedrooms` |
| `baths_full` | `BathroomsFull`, `Full Baths`, `#FBaths`, `FB`, `Baths Full` |
| `baths_half` | `BathroomsHalf`, `Half Baths`, `#HBaths`, `HB`, `Baths Half` |
| `street_address` | `UnparsedAddress`, `Address`, `Address Line`, `Street Address`, `StreetName`, `Property Address` |
| `unit_number` | `UnitNumber`, `Unit #`, `Unit`, `Apt`, `Unit No` |
| `building_name` | `BuildingName`, `Condo Name`, `Complex Name`, `Development Name`, `Project Name` |
| `unit_floor` | `EntryLevel`, `Floor`, `Unit Floor`, `Unit Floor Location`, `Story`, `Floor Number`, `Level` |
| `total_stories` | `StoriesTotal`, `Total Floors`, `Total Floors In Building`, `Building Stories`, `Floors In Building`, `#Stories` |
| `year_built` | `YearBuilt`, `Yr Built`, `Year Built`, `Year` |
| `new_construction` | `NewConstructionYN`, `New Construction`, `NewConstruction` |
| `hoa_monthly` | `AssociationFee`, `Association Fee`, `HOA Fee`, `Maintenance Fee`, `Maintenance Charge/Month`, `Monthly Fee`, `Condo Fee`, `HOAFee` |
| `hoa_frequency` | `AssociationFeeFrequency`, `HOA Frequency`, `Fee Frequency` |
| `tax_annual` | `TaxAnnualAmount`, `Taxes`, `Annual Taxes`, `Tax Amount` |
| `zip_code` | `PostalCode`, `Zip`, `Zip Code`, `ZIP`, `Postal` |
| `city` | `City`, `CityName` |
| `subdivision` | `SubdivisionName`, `Subdivision`, `Subdivision/Complex/Bldg.`, `Neighborhood`, `Area`, `Community` |
| `property_type` | `PropertyType`, `PropertySubType`, `Type`, `Property Sub Type` |
| `sale_type` | `SpecialListingConditions`, `Sale Type`, `Terms`, `Short Sale`, `Special Conditions` |
| `list_agent_name` | `ListAgentFullName`, `List Agent`, `Listing Agent`, `LA Name` |
| `list_office_name` | `ListOfficeName`, `List Office`, `Listing Office`, `LO Name` |
| `waterfront` | `WaterfrontYN`, `Waterfront`, `Water Front`, `Waterfront Property (Y/N)` |
| `view_description` | `View`, `ViewDescription`, `Views` |

When several source columns normalize to the same canonical, prefer the more
specific header (e.g. `ML#` over bare `MLS` board name; `SqFt Liv Area` over
`Main Living Area`; `Off Market Date` over `Status Change Date`). Losers stay
in `_unmapped`.

**Unmatched columns:** do not discard them. Keep them in a `_unmapped` dict per row and
log the distinct unmapped header names once, sorted by frequency. That log is how the
alias table gets extended.

---

## 3. Status normalization

Map to the canonical enum. Match on the normalized (lowercased, punctuation-stripped)
value, and check prefixes since MLS systems abbreviate.

| Canonical | Source values seen |
|---|---|
| `SOLD` | `Sold`, `Closed`, `S`, `CLS`, `Sold/Closed` |
| `EXPIRED` | `Expired`, `X`, `EXP` |
| `WITHDRAWN` | `Withdrawn`, `W`, `WDN`, `Temporarily Off Market`, `Temp Off Market`, `Hold` |
| `CANCELED` | `Canceled`, `Cancelled`, `C`, `CAN`, `Terminated` |
| `ACTIVE` | `Active`, `A`, `ACT`, `Coming Soon`, `New`, `Active Under Contract`, `Active With Contract`, `Backup`, `Contingent` |
| `PENDING` | `Pending`, `P` |

**Handling by status:**
- `SOLD` → the event. `event_sold = 1`.
- `EXPIRED`, `WITHDRAWN`, `CANCELED` → **right-censored**. `event_sold = 0`. These are
  the records that make elasticity identifiable; never drop them.
- `PENDING` → treat as `SOLD` for survival purposes if `pending_date` exists (the
  demand event is the buyer committing, not the deed recording). Flag it so this choice
  is auditable.
- `ACTIVE` → right-censored **at the export date**, measured from the listing's own
  list date. Not at any status-change date: a status change on a live listing is a
  price cut or a re-list, not the end of the spell.
- `Active With Contract` / `Active Under Contract` / `Backup` / `Contingent` →
  **`ACTIVE`, not `PENDING`.** A contract exists but the listing is still on the
  market taking backups and the spell has not ended, so the honest treatment is
  right-censoring. Scoring these as sales converts still-open listings into
  completed ones and biases the hazard upward — 217 rows on the 2023–2026 pull.

### `Last Status` is mapped but never codes an outcome

The export carries a second status column, `Last Status`, filled on 87.7% of rows.
It is normalized to `last_status` so it stops landing in `_unmapped` with no
semantics, but **nothing reads it for event coding.** That is deliberate.

`Last Status` is the *previous* status, not a competing view of the current one.
Cross-tabbed against the hard evidence of a `Closing Date`
(`audit/r04_last_status.py`):

| Column | Agrees with `Closing Date` on a sale | Claims a sale with no closing date |
|---|---|---|
| `Status` | 16,014 / 16,014 | 0 |
| `Last Status` | 1 / 16,014 | 0 |

Every `Closed` row has a `Last Status` of `Pending` (10,744) or
`Active With Contract` (4,419) — the state the listing was in immediately before
it closed. Reading it as the outcome would score essentially every completed sale
as censored.

Its one genuine use is diagnostic, not structural: among `Cancelled` listings,
11,101 have a `Last Status` of `Active` (a plain withdrawal) while 338 reached
`Pending` (246) or `Active With Contract` (92) before cancelling — a contract that
fell through, which is a different event from a unit the market ignored. That
distinction is available to anyone investigating `defaults.cancelled_treatment`
(§3.1); it is not currently wired into the coding.

### `CANCELED` coding is a configured choice, not a constant

27% of the quarterly export is `Cancelled`, and the status is genuinely
ambiguous — a relist under a new agent, a brokerage change, and a seller giving
up all look the same. `defaults.cancelled_treatment` selects between
`censored` (the default and the historical behaviour), `excluded` (drop the rows
as administrative noise), and `event` (score them as off-market sales). An
unrecognised value raises rather than defaulting.

The choice is material and `python -m src.demand.diagnostics --cancelled-sweep`
refits under all three so it is visible rather than assumed. On the 2023–2026
pull:

| Treatment | Rows fitted | Events | `beta_price` | 95% CI |
|---|---|---|---|---|
| `censored` (default) | 44,591 | 16,176 | −0.4310 | [−0.4795, −0.3824] |
| `excluded` | 32,562 | 16,176 | −0.3994 | [−0.4474, −0.3513] |
| `event` | 44,591 | 29,070 | −0.2336 | [−0.2659, −0.2012] |

The sign and the exclusion of zero survive all three; the magnitude moves by a
factor of 1.8. Any statement about the *size* of the elasticity therefore has to
name the treatment it was computed under.

---

## 4. Floor parsing from `unit_number`

Only when `unit_floor` is absent. Apply in order, first match wins:

| Pattern | Example | Floor | Note |
|---|---|---|---|
| `PH`, `PH-\d+`, `Penthouse` | `PH2` | `null` | Set `is_penthouse = True`, `floor_source = "parsed"`, floor left null — penthouse level is not inferable |
| `TS`, `Townhouse`, `TH` | `TH-4` | `1` | Townhouse units are ground-level |
| `LPH`, `Lower Penthouse` | `LPH3` | `null` | `is_penthouse = True` |
| 4 digits | `2506` | `25` | First 2 digits = floor |
| 3 digits | `805` | `8` | First 1 digit = floor |
| 2 digits | `12` | `null` | **Ambiguous** — could be unit 12 on floor 1, or floor 12. `floor_source = "missing"` |
| Letter + digits | `A1203` | `12` | Strip leading letters, then apply digit rules |
| Digits + letter | `1203B` | `12` | Strip trailing letters |
| Contains `-` | `12-03` | `12` | Segment before hyphen |

**Sanity gate:** if parsed `floor > total_stories` (when `total_stories` is known), reject
the parse and set `floor_source = "missing"`. A 4-digit unit number in a 12-story building
is not floor 25 — the building's numbering convention is something else.

**Never** impute a floor from the type average or submarket mean. A wrong floor corrupts
the hedonic surface, which then corrupts the price bounds fed to the optimizer. Missing is
strictly better than wrong here; the model handles nulls by dropping `log_floor` for those
rows or using `floor_bucket = "unknown"` as its own category.

---

## 5. `duration_days` computation

Priority order — use the first that yields a positive, plausible value:

1. `SOLD` with both dates → `close_date − list_date`
2. `SOLD`/`PENDING` with `pending_date` → `pending_date − list_date` *(preferred when
   available: it measures time to buyer commitment, not to escrow close, and escrow
   length is a financing artifact rather than a demand signal)*
3. Censored with `off_market_date` → `off_market_date − list_date`
4. Any status → reported `days_on_market`
5. `ACTIVE` → `export_date − list_date`

### Recovering a missing `list_date`

Where the export leaves it blank, derive it as `terminal date − days_on_market` and
mark `list_date_source = "derived_from_dom"`. Blanks are rarely random: on the
2023–2026 pull `List Date` is null for 100% of six live and off-market statuses and
0% of the rest, so dropping those rows selects the panel on the outcome.

The anchor depends on whether the clock has stopped:

| listing state | anchor | validated exactness |
|---|---|---|
| sold / under contract | `pending_date` | 93.6% |
| terminated (cancelled, expired, withdrawn) | `off_market_date` | 90.0–90.3% |
| **still live** (`ACTIVE`) | **the export snapshot** | 97.6% in-quarter |

Never `close_date` (1.3% exact) — DOM stops at buyer commitment, not at closing, so
escrow is not time on market. Never `status_change_date` for a live listing (25.1%) —
that is when the record last moved, not when the listing started.

The export snapshot is `max(status_change_date)`: that column is 100% filled and
cannot postdate the pull. Do **not** use "the latest date anywhere in the file" —
`off_market_date` carries scheduled future terminations and runs months past the
actual pull, which would push every recovered date early and censor every live
listing late.

Where the export arrives as one file per quarter, validate every derived date against
its own file's quarter and leave outliers missing rather than carrying a
plausible-looking date into the wrong comps cell.

**Validation:** drop rows where `duration_days <= 0` or `> 1095` (3 years) and log the
count. Record which rule produced each value in a `duration_source` column so the
survival fit can be re-run on a restricted, higher-quality subset if needed.

---

## 6. Type coercion rules

- **Money** (`original_list_price`, `last_list_price`, `close_price`, `hoa_monthly`,
  `tax_annual`): strip `$`, `,`, whitespace; handle `(1,234)` as negative; empty string
  and `-` → null. Coerce to float.
- **Dates:** try ISO first, then `%m/%d/%Y`, `%m/%d/%y`, `%d-%b-%Y`, `%Y%m%d`. Ambiguous
  two-digit years: `>= 70` → 19xx, else 20xx. Log the format that succeeded per column,
  and if a column parses with more than one format, warn — mixed formats within a column
  usually mean concatenated exports.
- **Booleans:** `Y/N`, `Yes/No`, `T/F`, `True/False`, `1/0`, `X`/blank → bool. Blank → null,
  **not** `False`.
- **Area:** strip commas and units (`sqft`, `SF`, `ft²`). Reject `< 200` or `> 20000` as
  implausible for a condo and null it with a logged reason.
- **HOA frequency:** normalize to a monthly figure — `Annually` → `/12`,
  `Quarterly` → `/3`, `Semi-Annually` → `/6`, `Monthly` → as-is. Store both
  `hoa_monthly` (normalized) and the original frequency. Where no frequency column
  exists, **do not guess and do not null the tails**: on the 2023–2026 pull
  `Association Fee` overlaps `Maintenance Charge/Month` on 6,943 rows with a median
  ratio of exactly 1.00, which settles that the field is already monthly. A high fee
  in a full-service oceanfront tower is a real fee carrying real amenity signal, and
  discarding it on suspicion throws that signal away.
- **ZIP:** keep as **string**, zero-pad to 5, truncate ZIP+4 to the first 5.

---

## 7. Validation report

`ingest_mls.py --inspect` must print, and write to
`data/processed/miami/ingest_report.json`:

```
FILES
  files read, rows per file, total rows in

MAPPING
  canonical fields matched (count + list)
  REQUIRED fields missing            ← if non-empty, FAIL
  IMPORTANT fields missing           ← warn
  unmapped source columns (sorted by frequency)

TYPES
  per column: null count, null rate, dtype, date format used

FILTERS
  rows dropped by reason: non-standard sale type, wrong property type,
  duplicate mls_number, implausible area, bad duration

STATUS
  distribution across SOLD / EXPIRED / WITHDRAWN / CANCELED / ACTIVE / PENDING
  PENDING rows treated as sold (§3), and the resulting survival event count —
  status counts alone understate events, so both must be printed
  ⚠  if censored (non-SOLD, non-ACTIVE) count == 0:
     "ELASTICITY NOT IDENTIFIABLE — export contains only sold records.
      Request expired, withdrawn, and canceled listings from the broker."

FLOOR
  floor_source distribution (reported / parsed / missing)
  parses rejected by the total_stories sanity gate

SUBMARKET COVERAGE
  rows with / without a submarket, and the top unmapped ZIPs
  ⚠  if > 5% of rows carry a ZIP absent from the market config:
     "SUBMARKET COVERAGE GAP — these rows join no submarket×month cell and are
      excluded from identification. Extend the submarket map or accept the
      reduced sample."

IDENTIFICATION READINESS
  (submarket, list_month) cells with >= min_cell_listings
    rows missing submarket or list_date are excluded — an "unknown" bucket
    must never be counted as a cell
  rel_price_premium: mean, sd, IQR, p5, p95
  ⚠  if IQR < 0.03:
     "PRICING VARIATION TOO NARROW — sellers priced near-identically relative to comps.
      Elasticity will be weakly identified regardless of sample size."
  count of usable rows (non-null in all REQUIRED + duration + rel_price_premium)
```

Those two warnings are the whole point of the report. They are the difference between
discovering a data problem in five minutes and discovering it after fitting a model and
trusting its output.

---

## 8. Minimum viable export

If the broker can only deliver a subset, this is the floor for the project to function:

**Must have:** `mls_number`, `status` (including at least one non-`SOLD` status),
`original_list_price`, `list_date`, `living_area_sqft`, `zip_code`, and a terminal date
(`close_date` or `off_market_date`) **or** `days_on_market`.

**Should have:** `close_price`, `beds`, `unit_number`, `building_name`, `property_type`,
`sale_type`.

Everything else improves precision but is not load-bearing.

Two things are worth pushing back on the broker for specifically, because exports often
default to omitting them:

1. **`original_list_price`, not just the current/last list price.** The starting price is
   what reveals how far the seller had to come down. Without it, the price-cut signal and
   a large part of the identification are gone.
2. **Non-sold statuses.** An export of sold-only records has no outcome variation — every
   row is a success — and elasticity is unidentifiable from it no matter how many rows it
   contains. `EXPIRED`, `WITHDRAWN`, and `CANCELED` records are not optional extras; they
   are half the estimator.
