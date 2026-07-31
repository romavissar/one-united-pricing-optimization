"""Column mapping, type coercion, floor/status parsing for MLS exports.

Units:
- Prices: USD total (converted to $/sqft as list_ppsf / close_ppsf).
- Area: sqft (Miami).
- Duration: days.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import MarketConfig, load_market_config, zip_to_submarket
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

CANONICAL_REQUIRED = (
    "mls_number",
    "status",
    "original_list_price",
    "list_date",
    "living_area_sqft",
    "zip_code",
)

CANONICAL_IMPORTANT = (
    "close_price",
    "close_date",
    "last_list_price",
    "days_on_market",
    "off_market_date",
    "beds",
    "baths_full",
    "street_address",
    "unit_number",
    "building_name",
    "property_type",
    "sale_type",
)

CANONICAL_OPTIONAL = (
    "unit_floor",
    "cumulative_days_on_market",
    "pending_date",
    "status_change_date",
    "baths_half",
    "year_built",
    "new_construction",
    "total_stories",
    "hoa_monthly",
    "hoa_frequency",
    "tax_annual",
    "subdivision",
    "city",
    "list_agent_name",
    "list_office_name",
    "waterfront",
    "view_description",
    # Added by the 2023Q1-2026Q3 quarterly re-pull.
    "waterfront_description",
    "restrictions",
    "parking_description",
    "amenities",
    "terms_considered",
    "occupancy_information",
    "furnished_info",
    "special_assessment",
    "special_information",
    "min_lease_days",
    "association_type",
    "is_reo",
    "is_short_sale",
)

CANONICAL_ALL = CANONICAL_REQUIRED + CANONICAL_IMPORTANT + CANONICAL_OPTIONAL

MONEY_COLS = (
    "original_list_price",
    "last_list_price",
    "close_price",
    "hoa_monthly",
    "tax_annual",
)

DATE_COLS = (
    "list_date",
    "pending_date",
    "close_date",
    "off_market_date",
    "status_change_date",
)

BOOL_COLS = ("new_construction", "waterfront")

INT_COLS = (
    "days_on_market",
    "cumulative_days_on_market",
    "beds",
    "unit_floor",
    "year_built",
    "total_stories",
)

FLOAT_COLS = ("living_area_sqft", "baths_full", "baths_half")

# Alias table from MLS_SCHEMA.md §2, extended for Matrix/SE Florida exports.
# EntryLevel → unit_floor (not unit_number).
_ALIAS_TABLE: dict[str, tuple[str, ...]] = {
    "mls_number": (
        "ListingId",
        "MLS #",
        "MLS Number",
        "MLSNum",
        "ML#",
        "Listing ID",
        "ListingKey",
        "ML Number",
        "MLSNumber",
    ),
    "status": (
        "Status",
        "MlsStatus",
        "StandardStatus",
        "Listing Status",
        "Stat",
    ),
    "original_list_price": (
        "OriginalListPrice",
        "Orig List Price",
        "Original List Price",
        "Original Price",
        "Orig Price",
        "OrigListPrice",
        "Original LP",
    ),
    "last_list_price": (
        "ListPrice",
        "Current Price",
        "List Price",
        "LP",
        "Last List Price",
        "CurrentListPrice",
    ),
    "close_price": (
        "ClosePrice",
        "Sold Price",
        "Sale Price",
        "SP",
        "Closed Price",
        "SoldPrice",
    ),
    "list_date": (
        "ListDate",
        "Listing Date",
        "OnMarketDate",
        "Date Listed",
        "LD",
        "ListingContractDate",
        "List Date",
    ),
    "pending_date": (
        "PendingDate",
        "Under Contract Date",
        "PurchaseContractDate",
        "Contract Date",
        "Pending Date",
    ),
    "close_date": (
        "CloseDate",
        "Sold Date",
        "Sale Date",
        "Closing Date",
        "SD",
        "ClosedDate",
    ),
    "off_market_date": (
        "OffMarketDate",
        "Off Market Date",
        "Expiration Date",
        "ExpirationDate",
        "Withdrawn Date",
        "Cancel Date",
    ),
    "status_change_date": ("StatusChangeDate", "Status Change Date"),
    "days_on_market": (
        "DaysOnMarket",
        "DOM",
        "Days On Market",
        "ADOM",
        "AgentDaysOnMarket",
    ),
    "cumulative_days_on_market": (
        "CumulativeDaysOnMarket",
        "CDOM",
        "Cumulative DOM",
        "CDOM (Days on Market)",
        "CDOM Days on Market",
    ),
    "living_area_sqft": (
        "LivingArea",
        "SqFt Living",
        "SqFt Liv Area",
        "Sq Ft Liv Area",
        "Main Living Area",
        "Total Living Area",
        "Living Sq Ft",
        "SqFtTotal",
        "Adjusted Sq Ft",
        "BuildingAreaTotal",
        "Living Area SqFt",
        "LA SqFt",
    ),
    "beds": (
        "BedroomsTotal",
        "Beds",
        "Bedrooms",
        "BR",
        "Total Bedrooms",
        "#Beds",
        "# Beds",
    ),
    "baths_full": (
        "BathroomsFull",
        "Full Baths",
        "FB",
        "Baths Full",
        "#FBaths",
        "# FBaths",
        "FBaths",
    ),
    "baths_half": (
        "BathroomsHalf",
        "Half Baths",
        "HB",
        "Baths Half",
        "#HBaths",
        "# HBaths",
        "HBaths",
    ),
    "street_address": (
        "UnparsedAddress",
        "Address",
        "Street Address",
        "StreetName",
        "Property Address",
        "Address Line",
        "AddressLine",
        "Full Address",
    ),
    "unit_number": ("UnitNumber", "Unit #", "Unit", "Apt", "Unit No", "Unit Number"),
    "building_name": (
        "BuildingName",
        "Condo Name",
        "Complex Name",
        "Development Name",
        "Project Name",
    ),
    "unit_floor": (
        "EntryLevel",
        "Floor",
        "Unit Floor",
        "Story",
        "Floor Number",
        "Level",
        "Unit Floor Location",
        "Floor Location",
        "UnitFloorLocation",
    ),
    "total_stories": (
        "StoriesTotal",
        "Total Floors",
        "Building Stories",
        "Floors In Building",
        "Total Floors In Building",
        "#Stories",
        "# Stories",
        "Stories",
    ),
    "year_built": ("YearBuilt", "Yr Built", "Year", "Year Built"),
    "new_construction": (
        "NewConstructionYN",
        "New Construction",
        "NewConstruction",
    ),
    "hoa_monthly": (
        "AssociationFee",
        "Association Fee",
        "HOA Fee",
        "Maintenance Fee",
        "Maintenance Charge/Month",
        "Maintenance Charge / Month",
        "Monthly Fee",
        "Condo Fee",
        "HOAFee",
    ),
    "hoa_frequency": (
        "AssociationFeeFrequency",
        "HOA Frequency",
        "Fee Frequency",
    ),
    "tax_annual": ("TaxAnnualAmount", "Taxes", "Annual Taxes", "Tax Amount"),
    "zip_code": ("PostalCode", "Zip", "Zip Code", "ZIP", "Postal"),
    "city": ("City", "CityName"),
    "subdivision": (
        "SubdivisionName",
        "Subdivision",
        "Neighborhood",
        "Area",
        "Community",
        "Subdivision/Complex/Bldg.",
        "Subdivision/Complex/Bldg",
        "Subdivision Complex Bldg",
    ),
    "property_type": (
        "PropertyType",
        "PropertySubType",
        "Type",
        "Property Sub Type",
    ),
    "sale_type": (
        "SpecialListingConditions",
        "Sale Type",
        "Terms",
        "Short Sale",
        "Special Conditions",
    ),
    "list_agent_name": (
        "ListAgentFullName",
        "List Agent",
        "Listing Agent",
        "LA Name",
    ),
    "list_office_name": (
        "ListOfficeName",
        "List Office",
        "Listing Office",
        "LO Name",
    ),
    "waterfront": (
        "WaterfrontYN",
        "Waterfront",
        "Water Front",
        "Waterfront Property (Y/N)",
        "Waterfront Property Y/N",
        "Waterfront Property",
    ),
    "view_description": ("View", "ViewDescription", "Views", "Unit View"),
    # --- Fields the quarterly re-pull added -------------------------------
    # Multi-valued, comma-separated. `features.py` splits them into atomic
    # tokens and one indicator per token; none is used as a single categorical,
    # because "Bay, Skyline View, Water View" is three facts, not one level.
    "waterfront_description": (
        "WaterfrontDescription",
        "Waterfront Description",
        "Water Frontage",
    ),
    "restrictions": ("Restrictions", "Listing Restrictions"),
    "parking_description": (
        "ParkingDescription",
        "Parking Description",
        "Parking Info",
    ),
    "amenities": ("Amenities", "Association Amenities", "Building Amenities"),
    "terms_considered": ("TermsConsidered", "Terms Considered", "Terms Offered"),
    # Scalar categoricals and numerics.
    "occupancy_information": ("OccupancyInformation", "Occupancy Information", "Occupancy"),
    "furnished_info": (
        "FurnishedInfo",
        "Furnished Info (List)",
        "Furnished Info List",
        "Furnished",
    ),
    "special_assessment": (
        "SpecialAssessmentYN",
        "Special Assessment YN",
        "Special Assessment",
    ),
    "special_information": ("SpecialInformation", "Special Information"),
    "min_lease_days": (
        "MinimumDaysForLease",
        "Minimum # of Days for Lease",
        "Minimum Days for Lease",
        "Min Days Lease",
    ),
    "association_type": ("TypeOfAssociation", "Type of Association", "Association Type"),
    # Distressed-sale markers. The old export carried neither, so the
    # `sale_type` filter was a no-op on it; these make it real.
    "is_reo": ("REO", "REOYN", "Bank Owned", "Real Estate Owned"),
    "is_short_sale": ("ShortSale", "Short Sale", "ShortSaleYN"),
}

# When multiple source columns normalize to the same canonical, higher wins.
_HEADER_PRIORITY: dict[str, dict[str, int]] = {
    "mls_number": {
        "ml": 100,  # ML#
        "listingid": 100,
        "listingkey": 100,
        "mlsnumber": 90,
        "mlsnum": 90,
        "mlnumber": 90,
        "mls": 1,  # bare "MLS" is often the board name, not the listing id
    },
    "living_area_sqft": {
        "sqftlivarea": 100,
        "sqftliving": 95,
        "livingsqft": 90,
        "livingarea": 85,
        "mainlivingarea": 50,
    },
    "last_list_price": {
        "currentprice": 100,
        "lastlistprice": 95,
        "listprice": 80,
        "lp": 40,
    },
    "off_market_date": {
        "offmarketdate": 100,
        "withdrawndate": 70,
        "expirationdate": 70,
        "canceldate": 60,
    },
    "hoa_monthly": {
        "associationfee": 100,
        "hoafee": 95,
        "maintenancefee": 90,
        "maintenancechargemonth": 80,  # prefer when present alone; Association Fee is denser
        "monthlyfee": 70,
    },
    "total_stories": {
        "totalfloorsinbuilding": 100,
        "floorsinbuilding": 90,
        "totalfloors": 80,
        "storiestotal": 80,
        "stories": 40,
    },
    "unit_floor": {
        "unitfloorlocation": 100,
        "floorlocation": 90,
        "unitfloor": 80,
        "entrylevel": 70,
        "floor": 50,
    },
    "street_address": {
        "addressline": 100,
        "unparsedaddress": 95,
        "propertyaddress": 90,
        "streetaddress": 85,
        "address": 70,
    },
    "subdivision": {
        "subdivisioncomplexbldg": 100,
        "subdivisionname": 80,
        "subdivision": 70,
    },
}

# Bare headers that collide after punctuation stripping and must not claim the field.
_REJECT_BARE_HEADERS: dict[str, set[str]] = {
    # "MLS" (board/region) normalizes the same as alias "MLS #".
    "mls_number": {"mls"},
}

STATUS_CANONICAL = (
    "SOLD",
    "EXPIRED",
    "WITHDRAWN",
    "CANCELED",
    "ACTIVE",
    "PENDING",
)

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d-%b-%Y",
    "%Y%m%d",
)


@dataclass
class FloorResult:
    """Parsed floor fields for one unit.

    floor: story number (1-indexed), or None when unknown / penthouse-only.
    floor_source: \"reported\" | \"parsed\" | \"missing\".
    is_penthouse: True when unit_number matches PH / LPH / Penthouse patterns.
    rejected_by_stories: True when a candidate floor exceeded total_stories and
        was discarded rather than kept (MLS_SCHEMA §4 sanity gate).
    """

    floor: int | None
    floor_source: str
    is_penthouse: bool = False
    rejected_by_stories: bool = False
    # Which branch of `resolve_floor` produced this, for the ingest report.
    resolution: str = ""


@dataclass
class MappingReport:
    """Result of mapping source headers onto canonical names."""

    matched: list[str] = field(default_factory=list)
    required_missing: list[str] = field(default_factory=list)
    important_missing: list[str] = field(default_factory=list)
    unmapped_headers: dict[str, int] = field(default_factory=dict)
    source_to_canonical: dict[str, str] = field(default_factory=dict)


@dataclass
class CoercionResult:
    """Typed frame plus the date format tally per date column."""

    frame: pd.DataFrame
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class NormalizeResult:
    """Normalized frame plus the diagnostics the ingest report is built from."""

    frame: pd.DataFrame
    mapping: MappingReport
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)
    floor_rejected_by_stories: int = 0
    list_date_recovered: int = 0
    list_date_rejected_outside_quarter: int = 0
    floor_resolution: dict[str, int] = field(default_factory=dict)


def normalize_header(name: str) -> str:
    """Normalize a header for alias matching: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in _ALIAS_TABLE.items():
        lookup[normalize_header(canonical)] = canonical
        for alias in aliases:
            lookup[normalize_header(alias)] = canonical
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()


def _header_priority(canonical: str, header: str) -> int:
    """Specificity score when several source columns claim the same canonical."""
    key = normalize_header(header)
    return _HEADER_PRIORITY.get(canonical, {}).get(key, 50)


def _reject_ambiguous_header(canonical: str, header: str) -> bool:
    """True when a bare header is known to collide with a different concept."""
    key = normalize_header(header)
    rejected = _REJECT_BARE_HEADERS.get(canonical, set())
    if key not in rejected:
        return False
    # Allow if the raw header still carries a number/# marker (e.g. "MLS #").
    raw = str(header)
    if "#" in raw or re.search(r"(?i)num|number|id|key", raw):
        return False
    return True


def is_missing(value: Any) -> bool:
    """True for every null this codebase can encounter.

    `isinstance(value, float) and pd.isna(value)` — the idiom this replaces —
    misses `pd.NA`, which is an `NAType`, not a float. Whole columns arrive as
    `pd.NA` whenever `map_columns` materialises a canonical field the export
    omitted, so the old guard fell through and stringified the sentinel: `slug`
    returned `"na"` for a missing building name (so `build_unit_key` never fell
    back to the street address, and every such unit collided in one namespace),
    and `hoa_frequency` came out as the literal string `"<NA>"`.
    """
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    # Array-like input: only a scalar can be "missing" here.
    return False


def slug(value: str | None) -> str:
    """Slug for unit_key components: lowercase alphanumerics only."""
    if is_missing(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def build_unit_key(
    building_name: str | None,
    street_address: str | None,
    unit_number: str | None,
) -> str:
    """Stable physical-unit id: slug(building|street) + '|' + slug(unit).

    Returns empty string when both location and unit are missing.
    """
    loc = slug(building_name) or slug(street_address)
    unit = slug(unit_number)
    if not loc and not unit:
        return ""
    return f"{loc}|{unit}"


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, MappingReport]:
    """Map source columns to canonical names; stash unmatched in `_unmapped`.

    Matching is case-insensitive and ignores spaces/underscores/hyphens/periods.
    When multiple source columns claim the same canonical, the higher-priority
    header wins; losers are kept in `_unmapped`.
    """
    report = MappingReport()
    candidates: dict[str, list[tuple[int, str]]] = {}
    unmapped_cols: list[str] = []

    for col in df.columns:
        header = str(col)
        key = normalize_header(header)
        canonical = _ALIAS_LOOKUP.get(key)
        if canonical is None or _reject_ambiguous_header(canonical, header):
            unmapped_cols.append(header)
            report.unmapped_headers[header] = int(df[col].notna().sum())
            continue
        candidates.setdefault(canonical, []).append(
            (_header_priority(canonical, header), header)
        )

    rename: dict[str, str] = {}
    for canonical, opts in candidates.items():
        opts_sorted = sorted(opts, key=lambda t: (-t[0], t[1]))
        _score, winner = opts_sorted[0]
        rename[winner] = canonical
        report.source_to_canonical[winner] = canonical
        report.matched.append(canonical)
        for _, loser in opts_sorted[1:]:
            unmapped_cols.append(loser)
            report.unmapped_headers[loser] = int(df[loser].notna().sum())

    out = df.rename(columns=rename).copy()

    if unmapped_cols:
        # Preserve original column order for unmapped payload.
        unmapped_cols = list(dict.fromkeys(unmapped_cols))
        unmapped_records: list[dict[str, Any]] = []
        for _, row in df[unmapped_cols].iterrows():
            unmapped_records.append(
                {c: (None if pd.isna(row[c]) else row[c]) for c in unmapped_cols}
            )
        out["_unmapped"] = unmapped_records
    else:
        out["_unmapped"] = [{} for _ in range(len(out))]

    for col in CANONICAL_ALL:
        if col not in out.columns:
            out[col] = pd.NA

    claimed = set(report.matched)
    report.required_missing = [c for c in CANONICAL_REQUIRED if c not in claimed]
    report.important_missing = [c for c in CANONICAL_IMPORTANT if c not in claimed]
    report.matched = sorted(claimed)
    report.unmapped_headers = dict(
        sorted(report.unmapped_headers.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return out, report


def coerce_money(value: Any) -> float | None:
    """Parse a money field to float USD. Empty / '-' → None."""
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip()
    if s == "" or s == "-":
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s == "" or s == "-":
        return None
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if negative else num


def coerce_bool(value: Any) -> bool | None:
    """Parse MLS boolean. Blank → None (not False)."""
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in {"y", "yes", "t", "true", "1", "x"}:
        return True
    if s in {"n", "no", "f", "false", "0"}:
        return False
    return None


def coerce_area(value: Any) -> float | None:
    """Parse living area in sqft. Rejects < 200 or > 20000 as implausible."""
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
    else:
        s = str(value).strip()
        s = re.sub(r"(?i)sq\.?\s*ft\.?|ft²|sf|sqft", "", s)
        s = s.replace(",", "").strip()
        if s == "" or s == "-":
            return None
        try:
            num = float(s)
        except ValueError:
            return None
    if num < 200 or num > 20000:
        # Counted in the ingest report rather than logged per row: at 48k rows a
        # per-row INFO buries every other diagnostic in the output.
        return None
    return num


def coerce_zip(value: Any) -> str | None:
    """Keep ZIP as string, zero-pad to 5, truncate ZIP+4 to first 5."""
    if is_missing(value):
        return None
    s = str(value).strip()
    if s.endswith(".0") and s.replace(".", "", 1).isdigit():
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if len(digits) >= 5:
        return digits[:5]
    return digits.zfill(5)


def coerce_date_with_format(value: Any) -> tuple[pd.Timestamp | None, str | None]:
    """Parse a date and report which format matched.

    Formats are tried in MLS_SCHEMA §6 order (ISO first, then US and abbreviated
    forms) so the returned label identifies the source convention. Pandas
    inference is a last resort and is labelled ``\"inferred\"``.
    """
    if is_missing(value):
        return None, None
    if isinstance(value, pd.Timestamp):
        return (value, "datetime") if not pd.isna(value) else (None, None)
    if isinstance(value, datetime):
        return pd.Timestamp(value), "datetime"
    if isinstance(value, date):
        return pd.Timestamp(value), "datetime"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel serial dates are handled upstream when reading xlsx; treat as YYYYMMDD if 8 digits.
        s = str(int(value))
        if len(s) == 8:
            try:
                return pd.Timestamp(datetime.strptime(s, "%Y%m%d")), "%Y%m%d"
            except ValueError:
                return None, None
        return None, None

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "nat", "-"}:
        return None, None

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if fmt == "%m/%d/%y":
            # Ambiguous two-digit years: >= 70 → 19xx, else 20xx.
            yy = int(s.rsplit("/", 1)[-1])
            dt = dt.replace(year=(1900 + yy) if yy >= 70 else (2000 + yy))
        return pd.Timestamp(dt), fmt

    try:
        ts = pd.to_datetime(s, utc=False)
    except (ValueError, TypeError):
        return None, None
    if pd.isna(ts):
        return None, None
    return pd.Timestamp(ts), "inferred"


def coerce_date(value: Any) -> pd.Timestamp | None:
    """Parse a date trying ISO then common MLS formats. Returns pandas Timestamp."""
    ts, _fmt = coerce_date_with_format(value)
    return ts


def normalize_hoa_frequency(freq: Any, hoa_value: float | None) -> tuple[float | None, str | None]:
    """Normalize HOA to a monthly USD figure.

    Returns (hoa_monthly, original_frequency_label).
    """
    if is_missing(hoa_value):
        label = None if is_missing(freq) else str(freq)
        return None, label

    if is_missing(freq):
        return float(hoa_value), None

    label = str(freq).strip()
    key = re.sub(r"[^a-z]", "", label.lower())
    monthly = float(hoa_value)
    if key in {"annually", "annual", "year", "yearly"}:
        monthly = monthly / 12.0
    elif key in {"quarterly", "quarter"}:
        monthly = monthly / 3.0
    elif key in {"semiannually", "semiannual", "biannually", "biannual"}:
        monthly = monthly / 6.0
    elif key in {"monthly", "month", "mo"}:
        pass
    return monthly, label


def normalize_status(raw: Any) -> str | None:
    """Map source status onto canonical enum. Prefix-aware for abbreviations."""
    if is_missing(raw):
        return None
    s = re.sub(r"[^a-z0-9]", "", str(raw).lower())
    if not s:
        return None

    exact = {
        "sold": "SOLD",
        "closed": "SOLD",
        "s": "SOLD",
        "cls": "SOLD",
        "soldclosed": "SOLD",
        "expired": "EXPIRED",
        "x": "EXPIRED",
        "exp": "EXPIRED",
        "withdrawn": "WITHDRAWN",
        "w": "WITHDRAWN",
        "wdn": "WITHDRAWN",
        "temporarilyoffmarket": "WITHDRAWN",
        "tempoffmarket": "WITHDRAWN",
        "temporaryoffmarket": "WITHDRAWN",
        "hold": "WITHDRAWN",
        "canceled": "CANCELED",
        "cancelled": "CANCELED",
        "c": "CANCELED",
        "can": "CANCELED",
        "terminated": "CANCELED",
        "active": "ACTIVE",
        "a": "ACTIVE",
        "act": "ACTIVE",
        "comingsoon": "ACTIVE",
        "new": "ACTIVE",
        # `Active With Contract` and `Active Under Contract` are LIVE listings.
        # A contract exists but the listing still accepts backups and the spell
        # has not ended, so the honest treatment is right-censoring at the export
        # date rather than scoring an event. The earlier mapping sent these to
        # PENDING, which counted them as sales; on the quarterly re-pull that
        # would convert 217 still-open listings into completed sales and bias
        # the hazard upward. `Pending` proper — where the listing has left the
        # active market — remains an event.
        "activeundercontract": "ACTIVE",
        "activewithcontract": "ACTIVE",
        "backup": "ACTIVE",
        "contingent": "ACTIVE",
        "pending": "PENDING",
        "p": "PENDING",
    }
    if s in exact:
        return exact[s]

    # Prefix checks for compound labels.
    if s.startswith("sold") or s.startswith("closed"):
        return "SOLD"
    if s.startswith("expired"):
        return "EXPIRED"
    if s.startswith("withdraw"):
        return "WITHDRAWN"
    if s.startswith("cancel") or s.startswith("termin"):
        return "CANCELED"
    # Order matters: "Active With Contract" starts with "active" and also
    # contains "contract", so the live-listing test has to run first.
    if s.startswith("active") or s.startswith("coming"):
        return "ACTIVE"
    if s.startswith("pending") or "undercontract" in s:
        return "PENDING"
    return None


_PH_RE = re.compile(r"(?i)^(ph|lph|phl|penthouse|lowerpenthouse)(-?[A-Za-z]?\d+)?$")
_TH_RE = re.compile(r"(?i)^(ts|th|townhouse)(-?\d+)?$")
_STREET_SUFFIXES = {
    "ave",
    "avenue",
    "st",
    "street",
    "rd",
    "road",
    "dr",
    "drive",
    "blvd",
    "boulevard",
    "ln",
    "lane",
    "ct",
    "court",
    "pl",
    "place",
    "ter",
    "terrace",
    "way",
    "hwy",
    "highway",
    "pkwy",
    "parkway",
    "cir",
    "circle",
}


def extract_unit_from_address(address: Any) -> str | None:
    """Pull unit number from an address string when unit_number column is blank.

    Handles explicit markers (Unit/Apt/#) and SE Florida Matrix style
    ``123 Main St 1204, CITY FL`` / ``123 Main St PH01, CITY FL``.
    """
    if is_missing(address):
        return None
    s = str(address).strip()
    if not s:
        return None

    patterns = (
        r"(?i)\b(?:unit|apt|suite|ste)\s*#?\s*([A-Za-z0-9-]+)\b",
        r"#\s*([A-Za-z0-9-]+)\b",
    )
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1).strip()

    # Matrix-style: unit token sits just before the city comma.
    left = s.split(",", 1)[0].strip()
    tokens = left.split()
    if len(tokens) < 2:
        return None
    last = tokens[-1].strip().lstrip("#")
    if not last:
        return None
    if last.lower().rstrip(".") in _STREET_SUFFIXES:
        return None
    if re.search(r"\d", last) or re.match(r"(?i)^(ph|lph|th|ts)", last):
        return last
    return None


def parse_floor(
    unit_number: Any,
    unit_floor: Any = None,
    total_stories: Any = None,
) -> FloorResult:
    """Derive floor from reported unit_floor or unit_number patterns.

    Never imputes from averages. Units: floor is story index (int).
    """
    # Reported floor wins when present.
    if not is_missing(unit_floor):
        try:
            reported = int(float(unit_floor))
            if reported > 0:
                stories = None
                if not is_missing(total_stories):
                    stories = int(float(total_stories))
                if stories is not None and reported > stories:
                    return FloorResult(
                        floor=None,
                        floor_source="missing",
                        is_penthouse=False,
                        rejected_by_stories=True,
                    )
                return FloorResult(floor=reported, floor_source="reported", is_penthouse=False)
        except (TypeError, ValueError):
            pass

    if is_missing(unit_number):
        return FloorResult(floor=None, floor_source="missing", is_penthouse=False)

    raw = str(unit_number).strip()
    if not raw:
        return FloorResult(floor=None, floor_source="missing", is_penthouse=False)

    compact = re.sub(r"[\s#]", "", raw)

    if _PH_RE.match(compact) or re.search(r"(?i)penthouse", compact):
        return FloorResult(floor=None, floor_source="parsed", is_penthouse=True)

    if _TH_RE.match(compact):
        return FloorResult(floor=1, floor_source="parsed", is_penthouse=False)

    # Strip leading/trailing letters for digit rules, keep hyphen form.
    floor: int | None = None
    source = "missing"

    if "-" in compact:
        left = compact.split("-", 1)[0]
        digits = re.sub(r"[^0-9]", "", left)
        if digits:
            try:
                floor = int(digits)
                source = "parsed"
            except ValueError:
                floor = None
    else:
        # Letter + digits or digits + letter
        m = re.match(r"^[A-Za-z]*(\d+)[A-Za-z]*$", compact)
        if m:
            digits = m.group(1)
            n = len(digits)
            if n == 2:
                # Ambiguous — unit 12 vs floor 12.
                return FloorResult(floor=None, floor_source="missing", is_penthouse=False)
            if n == 3:
                floor = int(digits[0])
                source = "parsed"
            elif n == 4:
                floor = int(digits[:2])
                source = "parsed"
            elif n > 4:
                # Take first 2 as floor for longer condo numbers.
                floor = int(digits[:2])
                source = "parsed"
            else:
                # Single digit — treat as floor.
                floor = int(digits)
                source = "parsed"

    stories = None
    if not is_missing(total_stories):
        try:
            stories = int(float(total_stories))
        except (TypeError, ValueError):
            stories = None

    if floor is not None and stories is not None and floor > stories:
        return FloorResult(
            floor=None,
            floor_source="missing",
            is_penthouse=False,
            rejected_by_stories=True,
        )

    if source == "parsed" and floor is not None:
        return FloorResult(floor=floor, floor_source="parsed", is_penthouse=False)
    return FloorResult(floor=None, floor_source="missing", is_penthouse=False)


# A floor above this is not a floor. The tallest building in Miami-Dade is under
# 90 storeys; the export carries values up to 373,737, which are unit numbers or
# keying errors sitting in the floor column.
_MAX_PLAUSIBLE_FLOOR = 150
# A condo listing reporting one storey or none has an unusable building height,
# not a townhouse — the field is blank-coded rather than measured.
_MIN_PLAUSIBLE_TOTAL_STORIES = 2


def resolve_floor(
    *,
    unit_number: Any,
    unit_floor: Any,
    total_stories: Any,
) -> FloorResult:
    """Decide a unit's floor when the reported fields disagree.

    On the quarterly export 1,417 rows (2.9%) report a floor above their
    building's height. The naive gate — null the floor whenever it exceeds
    `total_stories` — is wrong roughly half the time, because the conflict has
    three different causes and only one of them is a bad floor:

    * **the height is unusable** (652 rows report 0 or 1 total storeys for a
      condo). Nothing is wrong with the floor; the gate was discarding good data
      against a field that carries no information.
    * **the floor column holds something else** (551 rows report a floor above
      150, up to 373,737 — a unit number or a keying slip).
    * **a genuine two-field conflict** (223 rows, both values plausible).

    The unit number is an independent third signal, and it adjudicates: a unit
    numbered 3106 in a building reporting 29 storeys corroborates a reported
    floor of 31 and impeaches the height, while a unit numbered 106 in a
    building of 12 impeaches a reported floor of 106. Where the unit number
    corroborates neither, the floor is left missing — `AGENTS.md` §2, missing
    beats wrong, because a wrong floor corrupts the hedonic surface that sets
    the optimizer's price bounds.
    """
    reported: int | None = None
    if not is_missing(unit_floor):
        try:
            candidate = int(float(unit_floor))
            if 0 < candidate <= _MAX_PLAUSIBLE_FLOOR:
                reported = candidate
        except (TypeError, ValueError):
            reported = None

    stories: int | None = None
    if not is_missing(total_stories):
        try:
            candidate = int(float(total_stories))
            if _MIN_PLAUSIBLE_TOTAL_STORIES <= candidate <= _MAX_PLAUSIBLE_FLOOR:
                stories = candidate
        except (TypeError, ValueError):
            stories = None

    # What the unit number says, judged on its own.
    from_unit = parse_floor(unit_number, None, None)

    if reported is None:
        # No usable reported floor: fall back to the unit number, still gated on
        # the building height where that height is itself usable.
        result = parse_floor(unit_number, None, stories)
        result.resolution = (
            "from_unit_number" if result.floor is not None else "unresolved"
        )
        return result

    if stories is None or reported <= stories:
        return FloorResult(
            floor=reported,
            floor_source="reported",
            is_penthouse=from_unit.is_penthouse,
            resolution="reported_ok" if stories is not None else "reported_no_height",
        )

    # reported > stories, both plausible on their own.
    if from_unit.floor is not None and abs(from_unit.floor - reported) <= 1:
        # The unit number backs the floor, so the building height is the odd one.
        return FloorResult(
            floor=reported,
            floor_source="reported",
            is_penthouse=from_unit.is_penthouse,
            resolution="height_impeached_by_unit_number",
        )
    if from_unit.floor is not None and from_unit.floor <= stories:
        # The unit number backs the height, so the floor column is the odd one.
        return FloorResult(
            floor=from_unit.floor,
            floor_source="parsed",
            is_penthouse=from_unit.is_penthouse,
            resolution="floor_impeached_by_unit_number",
        )
    return FloorResult(
        floor=None,
        floor_source="missing",
        is_penthouse=from_unit.is_penthouse,
        rejected_by_stories=True,
        resolution="unresolved_conflict",
    )


def compute_duration_days(
    status: str | None,
    list_date: pd.Timestamp | None,
    close_date: pd.Timestamp | None,
    pending_date: pd.Timestamp | None,
    off_market_date: pd.Timestamp | None,
    days_on_market: Any,
    export_date: pd.Timestamp | None = None,
) -> tuple[float | None, str | None]:
    """Compute duration_days (days) and duration_source per MLS_SCHEMA §5."""

    def _diff(end: pd.Timestamp | None, start: pd.Timestamp | None) -> float | None:
        if end is None or start is None or pd.isna(end) or pd.isna(start):
            return None
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        return float(days)

    # Preferred: PENDING/SOLD with pending_date (buyer commitment).
    if status in {"SOLD", "PENDING"} and pending_date is not None and not pd.isna(pending_date):
        d = _diff(pending_date, list_date)
        if d is not None and d > 0:
            return d, "pending_date"

    if status == "SOLD" and close_date is not None and not pd.isna(close_date):
        d = _diff(close_date, list_date)
        if d is not None and d > 0:
            return d, "close_date"

    if status in {"EXPIRED", "WITHDRAWN", "CANCELED"} and off_market_date is not None:
        d = _diff(off_market_date, list_date)
        if d is not None and d > 0:
            return d, "off_market_date"

    # A live listing is right-censored at the export snapshot, and that must be
    # measured from its own list date rather than taken from any status-change
    # date: a status change on an Active listing is a price cut or a re-list,
    # not the end of the spell. This branch runs before the reported-DOM
    # fallback so an Active row is never censored at whatever clock DOM happens
    # to be on.
    if status == "ACTIVE":
        end = export_date if export_date is not None else pd.Timestamp(datetime.utcnow().date())
        d = _diff(end, list_date)
        if d is not None and d > 0:
            return d, "export_date"

    if not is_missing(days_on_market):
        try:
            dom = float(days_on_market)
            if dom > 0:
                return dom, "days_on_market"
        except (TypeError, ValueError):
            pass

    return None, None


def infer_export_snapshot(df: pd.DataFrame) -> pd.Timestamp | None:
    """When the export was pulled, read off the data rather than assumed.

    Two things depend on this instant: where a still-live listing is
    right-censored, and the anchor that recovers its missing list date. Both are
    wrong if the instant is.

    `status_change_date` is the field to read. It is 100% filled and records
    when the record last moved, so it cannot postdate the pull — its maximum
    *is* the pull. The obvious alternative, "the latest date anywhere in the
    file", is wrong: `off_market_date` carries scheduled future terminations
    and runs to 2026-10-30 on this export, three months past the actual pull of
    2026-07-31. Anchoring on that would push every recovered list date three
    months early and censor every live listing three months late.
    """
    for column in ("status_change_date", "close_date", "pending_date", "list_date"):
        if column in df.columns and df[column].notna().any():
            value = pd.to_datetime(df[column], errors="coerce").max()
            if pd.notna(value):
                return pd.Timestamp(value).normalize()
    return None


def reconstruct_list_date(
    df: pd.DataFrame,
    *,
    source_quarter: pd.Series | None = None,
    export_date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series, int, int]:
    """Recover `list_date` where the export leaves it blank, from terminal − DOM.

    Matrix populates `List Date` for some statuses and not others. In the Miami
    pull it is blank for **every** PENDING and **every** WITHDRAWN row and
    present for every SOLD, CANCELED, and EXPIRED row. Because
    `rel_price_premium` needs `list_month`, leaving it blank drops those rows —
    and that is a panel selected on the outcome: 100% of one event class and
    100% of one censored class vanish from identification.

    The value is recoverable. `days_on_market` counts from the listing date to
    the moment the listing left the market, so `list_date = terminal − DOM`.
    Validated on the statuses where all three fields are present:

    ==========================  =====================  ==========
    identity                    status                 exact
    ==========================  =====================  ==========
    pending_date − DOM          SOLD                   93.6%
    off_market_date − DOM       CANCELED               90.0%
    off_market_date − DOM       EXPIRED                90.3%
    close_date − DOM            SOLD                    0.9%
    ==========================  =====================  ==========

    So DOM stops at buyer commitment, not at closing: `pending_date` is the
    right terminal for an event and `off_market_date` for a termination.
    `close_date` is not used — the escrow period is not time on market.

    This is a derivation from two present columns, not an imputation: nothing is
    borrowed from another listing. Every recovered value is marked
    `list_date_source = "derived_from_dom"` so it can be excluded on demand, per
    `AGENTS.md` §2.

    Returns:
        (list_date, list_date_source, n_recovered). `list_date_source` is
        ``"reported" | "derived_from_dom" | "missing"``.
    """
    n = len(df)
    if "list_date" in df.columns:
        listed = pd.to_datetime(df["list_date"], errors="coerce")
    else:
        listed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    source = pd.Series("missing", index=df.index, dtype="object")
    source.loc[listed.notna()] = "reported"

    dom = (
        pd.to_numeric(df["days_on_market"], errors="coerce")
        if "days_on_market" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    # An event terminates at buyer commitment; a termination at going off market.
    pending = (
        pd.to_datetime(df["pending_date"], errors="coerce")
        if "pending_date" in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    off_market = (
        pd.to_datetime(df["off_market_date"], errors="coerce")
        if "off_market_date" in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    status_change = (
        pd.to_datetime(df["status_change_date"], errors="coerce")
        if "status_change_date" in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    status = df["status"] if "status" in df.columns else pd.Series(None, index=df.index)
    is_live = status.isin({"ACTIVE"})

    # The anchor depends on whether the clock has stopped, and getting this
    # wrong is what a naive single-anchor rule does. For a listing that has
    # terminated, DOM was measured to its own terminal date. For one that is
    # still on the market, DOM was measured to the moment the export was run —
    # so the anchor is the export snapshot, the same instant for every file,
    # and NOT the row's Status Change Date. A status change on a live listing
    # is a price cut or a re-list, so `status_change - DOM` lands wherever that
    # edit happened to fall: measured against the quarter each file is a query
    # for, it recovers 24% of live rows against 98% for the snapshot anchor
    # (`audit/r02_list_date_identity.py`).
    #
    # The snapshot is read off the data rather than written down: Status Change
    # Date is 100% filled and can never postdate the pull, so its maximum is the
    # pull. (Expiration Date runs years into the future and must not be used.)
    snapshot = export_date if export_date is not None else infer_export_snapshot(df)

    terminated_anchor = off_market.fillna(pending).fillna(status_change)
    anchor = terminated_anchor.copy()
    if pd.notna(snapshot):
        anchor.loc[is_live] = snapshot
    terminal = anchor

    recoverable = listed.isna() & terminal.notna() & dom.notna() & dom.ge(0)
    if not recoverable.any():
        return listed, source, 0, 0

    derived = terminal - pd.to_timedelta(dom.where(recoverable), unit="D")
    listed = listed.where(~recoverable, derived)
    # A derivation that lands on NaT (a bad terminal date) stays missing.
    recovered = recoverable & listed.notna()

    # Validate against the window the row's own quarterly file covers. Each file
    # is a query for listings whose list date falls inside that quarter, so a
    # derived date outside it is arithmetic that did not work — a DOM measured
    # on a different clock, or a status change that is not the end of the spell.
    # Those are dropped back to missing rather than carried as a plausible-
    # looking date that would place the listing in the wrong comps cell.
    rejected = 0
    if source_quarter is not None:
        quarters = pd.PeriodIndex(
            pd.Series(source_quarter, index=df.index).astype("string"), freq="Q"
        )
        derived_quarter = listed.dt.to_period("Q")
        outside = recovered & quarters.notna() & (derived_quarter != quarters)
        rejected = int(outside.sum())
        if rejected:
            listed = listed.where(~outside, pd.NaT)
            recovered = recovered & ~outside

    source.loc[recovered] = "derived_from_dom"
    count = int(recovered.sum())
    if count or rejected:
        logger.warning(
            "Recovered list_date for %d of %d rows (%.1f%%) from terminal date "
            "minus days_on_market, marked list_date_source='derived_from_dom'; "
            "%d further candidates landed outside their file's own quarter and "
            "were left missing. Without the derivation these rows carry no list "
            "month, so no comps cell and no premium, and they leave "
            "identification entirely — on this data they are exactly the live "
            "and off-market statuses, which is a panel selected on the outcome.",
            count, n, 100.0 * count / max(n, 1), rejected,
        )
    return listed, source, count, rejected


def coerce_types(df: pd.DataFrame) -> CoercionResult:
    """Coerce canonical columns to typed values. Money in USD; area in sqft.

    Also records which date format parsed each date column, so mixed-format
    columns (usually concatenated exports) can be warned about downstream.
    """
    out = df.copy()
    date_formats: dict[str, dict[str, int]] = {}

    for col in MONEY_COLS:
        if col in out.columns:
            out[col] = out[col].map(coerce_money)

    for col in DATE_COLS:
        if col not in out.columns:
            continue
        parsed = out[col].map(coerce_date_with_format)
        counts: dict[str, int] = {}
        for _ts, fmt in parsed:
            if fmt is None:
                continue
            counts[fmt] = counts.get(fmt, 0) + 1
        if counts:
            date_formats[col] = dict(
                sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        out[col] = pd.to_datetime([ts for ts, _ in parsed], errors="coerce")

    for col in BOOL_COLS:
        if col in out.columns:
            out[col] = out[col].map(coerce_bool)

    if "living_area_sqft" in out.columns:
        out["living_area_sqft"] = out["living_area_sqft"].map(coerce_area)

    if "zip_code" in out.columns:
        out["zip_code"] = out["zip_code"].map(coerce_zip)

    for col in INT_COLS:
        if col not in out.columns:
            continue

        def _to_int(v: Any) -> int | None:
            if is_missing(v):
                return None
            try:
                return int(float(str(v).replace(",", "").strip()))
            except (TypeError, ValueError):
                return None

        out[col] = out[col].map(_to_int)

    for col in FLOAT_COLS:
        if col == "living_area_sqft":
            continue
        if col not in out.columns:
            continue

        def _to_float(v: Any) -> float | None:
            if is_missing(v):
                return None
            try:
                return float(str(v).replace(",", "").strip())
            except (TypeError, ValueError):
                return None

        out[col] = out[col].map(_to_float)

    if "mls_number" in out.columns:
        out["mls_number"] = out["mls_number"].map(
            lambda v: None
            if is_missing(v)
            else str(v).strip()
        )

    # HOA frequency normalization after money coerce.
    if "hoa_monthly" in out.columns:
        freqs = out["hoa_frequency"] if "hoa_frequency" in out.columns else [None] * len(out)
        monthly_vals: list[float | None] = []
        freq_labels: list[str | None] = []
        for hoa, freq in zip(out["hoa_monthly"], freqs, strict=False):
            m, label = normalize_hoa_frequency(freq, hoa)
            monthly_vals.append(m)
            freq_labels.append(label)
        out["hoa_monthly"] = monthly_vals
        out["hoa_frequency"] = freq_labels

    for col, counts in date_formats.items():
        if len(counts) > 1:
            logger.warning(
                "Column %s parsed with %d different date formats %s — "
                "mixed formats usually mean concatenated exports",
                col,
                len(counts),
                counts,
            )

    return CoercionResult(frame=out, date_formats=date_formats)


def normalize_mls(
    df: pd.DataFrame,
    market: str = "miami",
    config: MarketConfig | None = None,
    export_date: pd.Timestamp | None = None,
    source_quarter: pd.Series | None = None,
) -> NormalizeResult:
    """Full normalize path: map → coerce → status/floor/unit_key/derived.

    Args:
        export_date: Censoring date for ACTIVE listings. Defaults to the latest
            date observed in the file, keeping ingestion deterministic.

    Raises SchemaError if any REQUIRED canonical field is unmatched in headers.
    """
    cfg = config if config is not None else load_market_config(market)
    mapped, report = map_columns(df)

    if report.required_missing:
        raise SchemaError(
            f"REQUIRED MLS fields missing from headers: {report.required_missing}"
        )

    coerced = coerce_types(mapped)
    typed = coerced.frame

    if export_date is None:
        export_date = infer_export_snapshot(typed)

    # Status
    typed["status"] = typed["status"].map(normalize_status)

    # List date, reconstructed where the export omits it (see the function).
    (
        typed["list_date"],
        typed["list_date_source"],
        list_date_recovered,
        list_date_rejected,
    ) = reconstruct_list_date(
        typed, source_quarter=source_quarter, export_date=export_date
    )

    # Fill unit_number from address when blank.
    units: list[Any] = []
    for _, row in typed.iterrows():
        u = row.get("unit_number")
        blank = (
            u is None
            or is_missing(u)
            or pd.isna(u)
            or str(u).strip() in {"", "<NA>", "nan", "None"}
        )
        if blank:
            units.append(extract_unit_from_address(row.get("street_address")))
        else:
            units.append(u)
    typed["unit_number"] = units

    floors: list[int | None] = []
    floor_sources: list[str] = []
    is_ph: list[bool] = []
    unit_keys: list[str] = []
    list_ppsf: list[float | None] = []
    close_ppsf: list[float | None] = []
    submarkets: list[str | None] = []
    durations: list[float | None] = []
    duration_sources: list[str | None] = []
    events: list[int] = []
    pending_as_sold: list[bool] = []
    floor_rejected = 0

    floor_resolution: dict[str, int] = {}

    for _, row in typed.iterrows():
        fr = resolve_floor(
            unit_number=row.get("unit_number"),
            unit_floor=row.get("unit_floor"),
            total_stories=row.get("total_stories"),
        )
        floors.append(fr.floor)
        floor_sources.append(fr.floor_source)
        is_ph.append(fr.is_penthouse)
        if fr.rejected_by_stories:
            floor_rejected += 1
        if fr.resolution:
            floor_resolution[fr.resolution] = floor_resolution.get(fr.resolution, 0) + 1

        unit_keys.append(
            build_unit_key(
                row.get("building_name"),
                row.get("street_address"),
                row.get("unit_number"),
            )
        )

        area = row.get("living_area_sqft")
        olp = row.get("original_list_price")
        cp = row.get("close_price")
        if (
            area is not None
            and not is_missing(area)
            and area > 0
            and olp is not None
            and not is_missing(olp)
        ):
            list_ppsf.append(float(olp) / float(area))
        else:
            list_ppsf.append(None)

        if (
            area is not None
            and not is_missing(area)
            and area > 0
            and cp is not None
            and not is_missing(cp)
        ):
            close_ppsf.append(float(cp) / float(area))
        else:
            close_ppsf.append(None)

        submarkets.append(zip_to_submarket(row.get("zip_code"), cfg))

        status = row.get("status")
        # PENDING with pending_date → treat as SOLD for survival, flag auditable.
        treat_sold = False
        pending_flag = False
        if status == "SOLD":
            treat_sold = True
        elif status == "PENDING":
            pd_ = row.get("pending_date")
            if pd_ is not None and not pd.isna(pd_):
                treat_sold = True
                pending_flag = True
        pending_as_sold.append(pending_flag)

        # For survival event flag: PENDING-as-sold counts as event.
        effective_status = "SOLD" if treat_sold else status
        events.append(1 if effective_status == "SOLD" else 0)

        dur, dur_src = compute_duration_days(
            status=status if isinstance(status, str) else None,
            list_date=row.get("list_date"),
            close_date=row.get("close_date"),
            pending_date=row.get("pending_date"),
            off_market_date=row.get("off_market_date"),
            days_on_market=row.get("days_on_market"),
            export_date=export_date,
        )
        durations.append(dur)
        duration_sources.append(dur_src)

    typed["floor"] = floors
    typed["floor_source"] = floor_sources
    typed["is_penthouse"] = is_ph
    typed["unit_key"] = unit_keys
    typed["list_ppsf"] = list_ppsf  # $/sqft
    typed["close_ppsf"] = close_ppsf  # $/sqft
    typed["submarket"] = submarkets
    typed["duration_days"] = durations  # days
    typed["duration_source"] = duration_sources
    typed["event_sold"] = events
    typed["pending_treated_as_sold"] = pending_as_sold

    if report.unmapped_headers:
        logger.info(
            "Unmapped source headers (by non-null count): %s",
            report.unmapped_headers,
        )
    if floor_rejected:
        logger.info(
            "Rejected %d floor parses that exceeded total_stories", floor_rejected
        )

    return NormalizeResult(
        frame=typed,
        mapping=report,
        date_formats=coerced.date_formats,
        floor_rejected_by_stories=floor_rejected,
        list_date_recovered=list_date_recovered,
        list_date_rejected_outside_quarter=list_date_rejected,
        floor_resolution=floor_resolution,
    )
