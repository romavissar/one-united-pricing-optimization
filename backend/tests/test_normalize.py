"""Phase 1 — normalize / clean / ingest tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_market_config, zip_to_submarket
from src.data.clean import clean_mls
from src.data.ingest_mls import format_inspect_report, ingest_mls
from src.data.normalize import (
    build_unit_key,
    coerce_money,
    coerce_zip,
    normalize_header,
    normalize_mls,
    normalize_status,
    parse_floor,
)
from src.exceptions import SchemaError

FIXTURE = Path(__file__).parent / "fixtures" / "messy_mls_20.csv"


def test_normalize_header_tolerant() -> None:
    from src.data.normalize import _ALIAS_LOOKUP

    assert normalize_header("Orig List Price") == "origlistprice"
    assert normalize_header("Orig-List-Price") == "origlistprice"
    assert _ALIAS_LOOKUP[normalize_header("Orig List Price")] == "original_list_price"
    assert _ALIAS_LOOKUP[normalize_header("ORIGINAL LIST PRICE")] == "original_list_price"
    assert _ALIAS_LOOKUP[normalize_header("original_list_price")] == "original_list_price"


def test_messy_fixture_produces_canonical_typed_frame() -> None:
    raw = pd.read_csv(FIXTURE, dtype=str)
    cfg = load_market_config("miami")
    result = normalize_mls(raw, config=cfg)
    frame, mapping = result.frame, result.mapping

    assert not mapping.required_missing
    for col in (
        "mls_number",
        "status",
        "original_list_price",
        "list_date",
        "living_area_sqft",
        "zip_code",
        "floor",
        "floor_source",
        "unit_key",
        "list_ppsf",
        "duration_days",
        "event_sold",
        "submarket",
    ):
        assert col in frame.columns

    assert frame["original_list_price"].dtype == object or pd.api.types.is_float_dtype(
        frame["original_list_price"]
    )
    assert frame["original_list_price"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(frame["list_date"]) or all(
        isinstance(v, pd.Timestamp) or v is None or pd.isna(v) for v in frame["list_date"]
    )
    assert frame["zip_code"].map(lambda z: isinstance(z, str) and len(z) == 5).all()
    assert set(frame["status"].dropna().unique()) <= {
        "SOLD",
        "EXPIRED",
        "WITHDRAWN",
        "CANCELED",
        "ACTIVE",
        "PENDING",
    }


def test_floor_parsing_rules() -> None:
    assert parse_floor("2506", None, 70).floor == 25
    assert parse_floor("2506", None, 70).floor_source == "parsed"

    assert parse_floor("805", None, 20).floor == 8

    amb = parse_floor("12", None, None)
    assert amb.floor is None and amb.floor_source == "missing"

    ph = parse_floor("PH2", None, 40)
    assert ph.is_penthouse and ph.floor is None and ph.floor_source == "parsed"

    th = parse_floor("TH-4", None, 3)
    assert th.floor == 1 and th.floor_source == "parsed"

    # Reported floor wins.
    assert parse_floor("2506", 10, 70).floor == 10
    assert parse_floor("2506", 10, 70).floor_source == "reported"

    # Sanity gate vs total_stories.
    rejected = parse_floor(None, 22, 12)
    assert rejected.floor is None and rejected.floor_source == "missing"
    assert rejected.rejected_by_stories is True
    assert parse_floor("2506", None, 12).rejected_by_stories is True
    assert parse_floor("2506", None, 70).rejected_by_stories is False

    letter = parse_floor("A1203", None, 50)
    assert letter.floor == 12

    hyphen = parse_floor("15-02", None, 40)
    assert hyphen.floor == 15


def test_status_and_money_zip() -> None:
    assert normalize_status("Sold/Closed") == "SOLD"
    assert normalize_status("WDN") == "WITHDRAWN"
    assert normalize_status("Active Under Contract") == "PENDING"
    assert coerce_money("($1,234)") == -1234.0
    assert coerce_money("$1,250,000") == 1250000.0
    assert coerce_zip("33131-1234") == "33131"
    assert coerce_zip(33131) == "33131"


def test_unit_key_and_submarket() -> None:
    key = build_unit_key("Brickell Flatiron", None, "2506")
    assert key == "brickellflatiron|2506"
    cfg = load_market_config("miami")
    assert zip_to_submarket("33131", cfg) == "brickell"
    assert zip_to_submarket("00000", cfg) is None


def test_fixture_floor_sources_and_non_sold() -> None:
    raw = pd.read_csv(FIXTURE, dtype=str)
    frame = normalize_mls(raw, config=load_market_config("miami")).frame

    # Ambiguous unit 12 → missing
    row12 = frame.loc[frame["mls_number"] == "A1004"].iloc[0]
    assert row12["floor_source"] == "missing"

    # PH → penthouse parsed
    row_ph = frame.loc[frame["mls_number"] == "A1003"].iloc[0]
    assert bool(row_ph["is_penthouse"]) is True
    assert row_ph["floor_source"] == "parsed"

    # Reported floor > stories → missing (A1014)
    row_gate = frame.loc[frame["mls_number"] == "A1014"].iloc[0]
    assert row_gate["floor_source"] == "missing"

    non_sold = int((frame["status"] != "SOLD").sum())
    assert non_sold > 0


def test_clean_drops_distressed_and_dedup() -> None:
    raw = pd.read_csv(FIXTURE, dtype=str)
    cfg = load_market_config("miami")
    frame = normalize_mls(raw, config=cfg).frame

    # Inject distressed + duplicate.
    extra = frame.iloc[[0]].copy()
    extra["mls_number"] = frame.iloc[0]["mls_number"]
    distressed = frame.iloc[[1]].copy()
    distressed["mls_number"] = "DISTRESS1"
    distressed["sale_type"] = "Short Sale"
    combined = pd.concat([frame, extra, distressed], ignore_index=True)

    cleaned, report = clean_mls(combined, cfg)
    assert report.dropped_by_reason.get("duplicate_mls_number", 0) >= 1
    assert report.dropped_by_reason.get("distressed_sale_type", 0) >= 1
    assert "DISTRESS1" not in set(cleaned["mls_number"])


def test_ingest_inspect_report_and_elasticity_warning() -> None:
    result = ingest_mls(paths=FIXTURE, market="miami")
    assert result.report.rows_in == 20
    assert result.report.rows_out <= 20
    ident = result.report.identification
    assert ident["non_sold_count"] > 0

    # Status counts must reconcile with the survival event flag: every row is
    # either an event or censored, and PENDING-as-sold is stated explicitly.
    frame = result.frame
    assert ident["event_sold_count"] == int(frame["event_sold"].sum())
    assert ident["event_sold_count"] == ident["sold_count"] + ident["pending_treated_as_sold"]
    assert ident["sold_count"] + ident["non_sold_count"] == len(frame)

    text = format_inspect_report(result.report)
    for section in ("STATUS", "FLOOR", "SUBMARKET COVERAGE", "IDENTIFICATION READINESS"):
        assert section in text
    assert "non-SOLD count" in text
    assert "survival events" in text
    assert "parses rejected by total_stories sanity gate" in text

    # Sold-only export → loud warning.
    sold_only = result.frame.loc[result.frame["status"] == "SOLD"].copy()
    # Re-ingest via normalize/clean path by writing temp is heavy; build report directly.
    from src.data.ingest_mls import build_ingest_report
    from src.data.normalize import MappingReport
    from src.data.clean import CleanReport

    mapping = MappingReport(matched=list(sold_only.columns))
    clean = CleanReport(rows_in=len(sold_only), rows_out=len(sold_only))
    report = build_ingest_report(
        frame=sold_only,
        mapping=mapping,
        clean=clean,
        file_stats=[],
        rows_in=len(sold_only),
        config=load_market_config("miami"),
    )
    assert any("ELASTICITY NOT IDENTIFIABLE" in w for w in report.warnings)


def test_missing_required_raises() -> None:
    df = pd.DataFrame({"MLS #": ["X1"], "Status": ["Sold"]})
    with pytest.raises(SchemaError, match="REQUIRED"):
        normalize_mls(df, config=load_market_config("miami"))


def test_pricing_model_export_headers_map() -> None:
    """Regression: SE Florida Matrix export headers from PRICING_MODEL_export."""
    from src.data.normalize import extract_unit_from_address, map_columns

    headers = [
        "ML#",
        "MLS",
        "Original List Price",
        "List Price",
        "Current Price",
        "Status",
        "Sale Price",
        "SqFt Liv Area",
        "Main Living Area",
        "#Beds",
        "#FBaths",
        "#HBaths",
        "#Units",
        "Address Line",
        "Complex Name",
        "Zip Code",
        "#Stories",
        "Year Built",
        "Association Fee",
        "CDOM (Days on Market)",
        "DOM",
        "Subdivision/Complex/Bldg.",
        "Maintenance Charge/Month",
        "Waterfront Property (Y/N)",
        "Total Floors In Building",
        "Unit Floor Location",
        "List Date",
        "Closing Date",
        "Pending Date",
        "Off Market Date",
        "Withdrawn Date",
        "Expiration Date",
        "Status Change Date",
        "Last Status",
    ]
    df = pd.DataFrame([{h: "1" for h in headers}])
    mapped, report = map_columns(df)
    assert report.required_missing == []
    assert "living_area_sqft" in report.matched
    assert "baths_full" in report.matched
    assert "street_address" in report.matched
    assert "unit_floor" in report.matched
    assert "total_stories" in report.matched
    assert "cumulative_days_on_market" in report.matched
    assert "waterfront" in report.matched
    # ML# wins over board-name column "MLS"
    assert report.source_to_canonical.get("ML#") == "mls_number"
    assert "MLS" not in report.source_to_canonical
    # Prefer SqFt Liv Area over Main Living Area
    assert report.source_to_canonical.get("SqFt Liv Area") == "living_area_sqft"
    assert normalize_status("Closed") == "SOLD"
    assert normalize_status("Temp Off Market") == "WITHDRAWN"
    assert normalize_status("Cancelled") == "CANCELED"
    assert extract_unit_from_address("17901 COLLINS AVE PH01, SUNNYISL FL") == "PH01"
    assert extract_unit_from_address("1228 West Ave 911, MIAMIBCH FL") == "911"
    assert extract_unit_from_address("5 Grove Isle Dr #PHL1, MIAMI FL 33133") == "PHL1"


def test_date_formats_tracked_and_mixed_formats_warn() -> None:
    from src.data.normalize import coerce_date_with_format

    assert coerce_date_with_format("2024-01-15")[1] == "%Y-%m-%d"
    assert coerce_date_with_format("2026-03-23 17:26:12")[1] == "%Y-%m-%d %H:%M:%S"
    assert coerce_date_with_format("01/15/2024")[1] == "%m/%d/%Y"
    assert coerce_date_with_format("")[1] is None

    # Two-digit years: >= 70 → 19xx, else 20xx.
    assert coerce_date_with_format("01/15/98")[0].year == 1998
    assert coerce_date_with_format("01/15/24")[0].year == 2024

    raw = pd.read_csv(FIXTURE, dtype=str)
    result = normalize_mls(raw, config=load_market_config("miami"))
    # The fixture deliberately mixes ISO and US dates in list_date.
    assert len(result.date_formats["list_date"]) > 1

    report = ingest_mls(paths=FIXTURE, market="miami").report
    assert any("MIXED DATE FORMATS" in w for w in report.warnings)
    assert "list_date" in report.date_formats


def test_submarket_coverage_and_cell_counts_exclude_unknowns() -> None:
    """Rows without a submarket or list_date must not inflate readiness counts."""
    from src.data.clean import CleanReport
    from src.data.ingest_mls import build_ingest_report
    from src.data.normalize import MappingReport

    cfg = load_market_config("miami")
    frame = ingest_mls(paths=FIXTURE, market="miami").frame.copy()

    # Ten rows in an unmapped ZIP would form an "unknown" cell above the
    # min_cell_listings threshold if nulls were counted as a real submarket.
    orphan = pd.concat([frame.iloc[[0]]] * 10, ignore_index=True)
    orphan["submarket"] = None
    orphan["zip_code"] = "99999"
    orphan["mls_number"] = [f"ORPHAN{i}" for i in range(len(orphan))]
    combined = pd.concat([frame, orphan], ignore_index=True)

    report = build_ingest_report(
        frame=combined,
        mapping=MappingReport(matched=list(combined.columns)),
        clean=CleanReport(rows_in=len(combined), rows_out=len(combined)),
        file_stats=[],
        rows_in=len(combined),
        config=cfg,
    )

    keyed = combined.loc[combined["submarket"].notna() & combined["list_date"].notna()]
    expected = int(
        (
            keyed.groupby([keyed["submarket"], keyed["list_date"].dt.to_period("M")]).size()
            >= report.identification["min_cell_listings"]
        ).sum()
    )
    assert report.identification["submarket_month_cells_ge_min"] == expected

    cov = report.submarket_coverage
    assert cov["rows_without_submarket"] == 10
    assert cov["top_unmapped_zips"]["99999"] == 10
    assert any("SUBMARKET COVERAGE GAP" in w for w in report.warnings)