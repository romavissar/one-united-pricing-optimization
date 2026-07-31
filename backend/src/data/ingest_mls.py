"""Read raw MLS CSV/XLSX exports → normalized + cleaned frames.

Side effects (file I/O) live here and in --inspect report writing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MarketConfig, load_market_config
from src.data.clean import CleanReport, clean_mls
from src.data.normalize import (
    CANONICAL_ALL,
    CANONICAL_REQUIRED,
    MappingReport,
    normalize_mls,
)
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Share of rows with a ZIP absent from the market config above which the export
# is flagged: those rows cannot join a submarket×month cell and drop out of
# identification entirely.
_SUBMARKET_COVERAGE_WARN_RATE = 0.05


@dataclass
class FileReadStat:
    path: str
    rows: int


@dataclass
class IngestReport:
    """Validation / identification-readiness report for --inspect."""

    files: list[FileReadStat] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    mapping: dict[str, Any] = field(default_factory=dict)
    types: dict[str, Any] = field(default_factory=dict)
    date_formats: dict[str, dict[str, int]] = field(default_factory=dict)
    filters: dict[str, int] = field(default_factory=dict)
    status_distribution: dict[str, int] = field(default_factory=dict)
    floor_source_distribution: dict[str, int] = field(default_factory=dict)
    floor_rejected_by_stories: int = 0
    submarket_coverage: dict[str, Any] = field(default_factory=dict)
    identification: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    outlier_flagged: int = 0
    list_ppsf_implausible: int = 0
    list_date_recovered: int = 0


@dataclass
class IngestResult:
    """Normalized + cleaned MLS frame plus reports."""

    frame: pd.DataFrame
    mapping: MappingReport
    clean: CleanReport
    report: IngestReport


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str, keep_default_na=True)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise SchemaError(f"Unsupported MLS file type: {path}")


_QUARTER_FILENAME = re.compile(r"(?i)^~?\s*Q([1-4])[-_ ]?(\d{4})")


def quarter_from_filename(name: str) -> str | None:
    """Read the calendar quarter a quarterly export covers off its filename.

    The broker delivers one file per quarter (`Q1-2023.csv` … `~Q3-2026.csv`,
    the tilde marking the partial current quarter), and every listing inside a
    file has a list date inside that quarter. That is the only external check
    available on a *derived* list date, so it is worth reading rather than
    assuming. Returns None for a filename that carries no quarter, in which case
    the validation simply does not run.
    """
    match = _QUARTER_FILENAME.match(str(name).strip())
    if not match:
        return None
    return f"{match.group(2)}Q{match.group(1)}"


def resolve_mls_paths(paths: list[Path] | Path | None, raw_dir: Path | None = None) -> list[Path]:
    """Resolve input files: explicit paths, or all CSV/XLSX under raw_dir."""
    if paths is not None:
        if isinstance(paths, Path):
            candidates = [paths]
        else:
            candidates = list(paths)
        missing = [p for p in candidates if not p.is_file()]
        if missing:
            raise SchemaError(f"MLS files not found: {missing}")
        return candidates

    directory = raw_dir or (_BACKEND_ROOT / "data" / "raw" / "mls")
    if not directory.is_dir():
        raise SchemaError(f"MLS raw directory not found: {directory}")
    found = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".xlsx", ".xls"}
    )
    if not found:
        raise SchemaError(f"No CSV/XLSX files in {directory}")
    return found


def ingest_mls(
    paths: list[Path] | Path | None = None,
    *,
    market: str = "miami",
    config: MarketConfig | None = None,
    raw_dir: Path | None = None,
    export_date: pd.Timestamp | None = None,
) -> IngestResult:
    """Ingest one or more MLS exports into a normalized, cleaned frame.

    Accepts multiple files and concatenates (broker exports are often capped).
    """
    cfg = config if config is not None else load_market_config(market)
    file_paths = resolve_mls_paths(paths, raw_dir=raw_dir)

    frames: list[pd.DataFrame] = []
    file_stats: list[FileReadStat] = []
    for path in file_paths:
        part = _read_one(path)
        file_stats.append(FileReadStat(path=str(path), rows=len(part)))
        frames.append(part)

    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    rows_in = len(raw)

    # Each quarterly file is a query for listings whose list date falls inside
    # that quarter, so the filename carries a fact about every row in it. That
    # fact is what lets a derived list date be validated rather than trusted.
    source_quarter = pd.concat(
        [
            pd.Series(
                [quarter_from_filename(path.name)] * len(part), index=range(len(part))
            )
            for path, part in zip(file_paths, frames, strict=True)
        ],
        ignore_index=True,
    ) if len(frames) > 1 else pd.Series(
        [quarter_from_filename(file_paths[0].name)] * len(raw)
    )

    normalized = normalize_mls(
        raw,
        market=market,
        config=cfg,
        export_date=export_date,
        source_quarter=source_quarter,
    )
    cleaned, clean_report = clean_mls(normalized.frame, cfg)

    # Raw, unsorted status column: the sampling diagnostic reads the file's own
    # row order, which cleaning and reindexing would destroy.
    raw_status = None
    for column in raw.columns:
        if str(column).strip().lower() in {"status", "mlsstatus", "standardstatus"}:
            raw_status = raw[column]
            break

    report = build_ingest_report(
        frame=cleaned,
        mapping=normalized.mapping,
        clean=clean_report,
        file_stats=file_stats,
        rows_in=rows_in,
        config=cfg,
        date_formats=normalized.date_formats,
        floor_rejected_by_stories=normalized.floor_rejected_by_stories,
        list_date_recovered=normalized.list_date_recovered,
        raw_status=raw_status,
    )
    return IngestResult(
        frame=cleaned,
        mapping=normalized.mapping,
        clean=clean_report,
        report=report,
    )


def build_ingest_report(
    frame: pd.DataFrame,
    mapping: MappingReport,
    clean: CleanReport,
    file_stats: list[FileReadStat],
    rows_in: int,
    config: MarketConfig,
    date_formats: dict[str, dict[str, int]] | None = None,
    floor_rejected_by_stories: int = 0,
    list_date_recovered: int = 0,
    raw_status: pd.Series | None = None,
) -> IngestReport:
    """Assemble the --inspect report structure (features may be partial pre-Phase 2)."""
    warnings: list[str] = []

    status_dist: dict[str, int] = {}
    if "status" in frame.columns:
        status_dist = {
            str(k): int(v)
            for k, v in frame["status"].value_counts(dropna=False).items()
        }

    sold = int(status_dist.get("SOLD", 0))
    active = int(status_dist.get("ACTIVE", 0))
    # Censored for identification: EXPIRED / WITHDRAWN / CANCELED (and PENDING without sale treatment)
    censored = (
        int(status_dist.get("EXPIRED", 0))
        + int(status_dist.get("WITHDRAWN", 0))
        + int(status_dist.get("CANCELED", 0))
    )
    non_sold = len(frame) - sold

    if censored == 0:
        msg = (
            "ELASTICITY NOT IDENTIFIABLE — export contains only sold records "
            "(no EXPIRED / WITHDRAWN / CANCELED). "
            "Request expired, withdrawn, and canceled listings from the broker."
        )
        warnings.append(msg)

    floor_src = {}
    if "floor_source" in frame.columns:
        floor_src = {
            str(k): int(v)
            for k, v in frame["floor_source"].value_counts(dropna=False).items()
        }

    types_info: dict[str, Any] = {}
    for col in list(CANONICAL_ALL) + [
        "floor",
        "floor_source",
        "list_ppsf",
        "duration_days",
        "submarket",
        "event_sold",
    ]:
        if col not in frame.columns:
            continue
        series = frame[col]
        null_count = int(series.isna().sum())
        types_info[col] = {
            "null_count": null_count,
            "null_rate": round(null_count / len(frame), 4) if len(frame) else None,
            "dtype": str(series.dtype),
        }

    for col, counts in (date_formats or {}).items():
        if len(counts) > 1:
            warnings.append(
                f"MIXED DATE FORMATS in '{col}': {counts}. "
                "A column parsing with more than one format usually means "
                "concatenated exports — verify the dates before trusting durations."
            )

    # Submarket coverage: rows whose ZIP is absent from the market config cannot
    # join a (submarket, list_month) cell, so they contribute nothing to
    # identification even though they survived cleaning.
    coverage: dict[str, Any] = {}
    if "submarket" in frame.columns and len(frame):
        unmapped_mask = frame["submarket"].isna()
        unmapped_rows = int(unmapped_mask.sum())
        rate = unmapped_rows / len(frame)
        top_zips: dict[str, int] = {}
        if "zip_code" in frame.columns and unmapped_rows:
            top_zips = {
                str(z): int(n)
                for z, n in frame.loc[unmapped_mask, "zip_code"]
                .value_counts()
                .head(10)
                .items()
            }
        coverage = {
            "rows_with_submarket": len(frame) - unmapped_rows,
            "rows_without_submarket": unmapped_rows,
            "unmapped_rate": round(rate, 4),
            "top_unmapped_zips": top_zips,
        }
        if rate > _SUBMARKET_COVERAGE_WARN_RATE:
            warnings.append(
                f"SUBMARKET COVERAGE GAP — {unmapped_rows} rows ({rate:.1%}) have a ZIP "
                f"not listed in config/{config.get('market')}.yaml and are excluded from "
                f"every submarket×month cell. Top ZIPs: {top_zips}. "
                "Extend the submarket map or accept the reduced sample."
            )

    min_cell = int((config.get("defaults") or {}).get("min_cell_listings") or 8)
    event_sold_count = (
        int(frame["event_sold"].sum()) if "event_sold" in frame.columns else 0
    )
    pending_as_sold = (
        int(frame["pending_treated_as_sold"].sum())
        if "pending_treated_as_sold" in frame.columns
        else 0
    )
    identification: dict[str, Any] = {
        "sold_count": sold,
        "non_sold_count": non_sold,
        "censored_expired_withdrawn_canceled": censored,
        "active_count": active,
        "pending_count": int(status_dist.get("PENDING", 0)),
        "pending_treated_as_sold": pending_as_sold,
        "event_sold_count": event_sold_count,
        "min_cell_listings": min_cell,
        "rel_price_premium": None,
        "note": (
            "rel_price_premium is computed in Phase 2 (features). "
            "Identification IQR check runs after feature engineering."
        ),
    }

    # Cell counts on submarket × list_month. Rows missing either key are excluded:
    # counting them would inflate the readiness figure with an "unknown" bucket.
    cells_ge_min = 0
    if "list_date" in frame.columns and "submarket" in frame.columns and len(frame):
        keyed = frame.loc[frame["submarket"].notna() & frame["list_date"].notna()]
        if len(keyed):
            counts = keyed.groupby(
                [keyed["submarket"], keyed["list_date"].dt.to_period("M")]
            ).size()
            cells_ge_min = int((counts >= min_cell).sum())
    identification["submarket_month_cells_ge_min"] = cells_ge_min

    usable = 0
    if len(frame):
        req_ok = frame[list(CANONICAL_REQUIRED)].notna().all(axis=1)
        dur_ok = frame["duration_days"].notna() if "duration_days" in frame.columns else False
        sub_ok = frame["submarket"].notna() if "submarket" in frame.columns else False
        usable = int((req_ok & dur_ok & sub_ok).sum())
    identification["usable_rows_required_duration_submarket"] = usable

    sampling, sampling_warnings = build_sampling_report(frame, rows_in, raw_status)
    warnings.extend(sampling_warnings)

    if list_date_recovered:
        warnings.append(
            f"LIST DATE RECOVERED FOR {list_date_recovered} ROWS — the export left "
            "`list_date` blank on these and it was derived as terminal date minus "
            "days_on_market, marked `list_date_source='derived_from_dom'`. Without "
            "the derivation these rows carry no list month, so no submarket-month "
            "median, so no rel_price_premium, and they leave identification "
            "entirely. Check the status breakdown of `list_date_source`: if the "
            "blanks fall on whole statuses, the panel would otherwise have been "
            "selected on the outcome."
        )
    if clean.list_ppsf_implausible:
        warnings.append(
            f"IMPLAUSIBLE $/SQFT NULLED — {clean.list_ppsf_implausible} listings had a "
            "derived list $/sqft outside the physical band and were marked "
            "`list_ppsf_source='implausible'`. These are almost always a wrong "
            "living_area_sqft against a correct price. Left in, one of them sets the "
            "upper end of the fitted rel_price_premium support and the extrapolation "
            "guard can then flag nothing."
        )

    return IngestReport(
        files=file_stats,
        rows_in=rows_in,
        rows_out=len(frame),
        mapping={
            "matched": mapping.matched,
            "matched_count": len(mapping.matched),
            "required_missing": mapping.required_missing,
            "important_missing": mapping.important_missing,
            "unmapped_source_columns": mapping.unmapped_headers,
            "source_to_canonical": mapping.source_to_canonical,
        },
        types=types_info,
        date_formats=dict(date_formats or {}),
        filters=dict(clean.dropped_by_reason),
        status_distribution=status_dist,
        floor_source_distribution=floor_src,
        floor_rejected_by_stories=floor_rejected_by_stories,
        submarket_coverage=coverage,
        identification=identification,
        sampling=sampling,
        warnings=warnings,
        outlier_flagged=clean.outlier_flagged,
        list_ppsf_implausible=clean.list_ppsf_implausible,
        list_date_recovered=list_date_recovered,
    )


def build_sampling_report(
    frame: pd.DataFrame, rows_in: int, raw_status: pd.Series | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Detect selection built into how the export itself was drawn.

    Two failure modes, neither of which any per-column null check can see, and
    both of which change what `beta_price` is an estimate *of*.

    **Stock sampling on the terminal date.** If every listing in the file went
    off market inside a window much shorter than the span of list dates, the
    export is a sample of *terminations* in that window, not of listings that
    started in it. A listing that began early is then only present if it lasted
    long enough to survive into the window, and one that is still on the market
    at the export date is absent altogether. Since the listings still unsold are
    disproportionately the over-priced ones, their absence removes exactly the
    high-premium / long-duration observations that identify the price response,
    and attenuates `beta_price` toward zero.

    **Truncation on status.** A row-capped export drawn from a status-sorted
    result truncates whichever status sorts last. Status is the outcome
    variable, so this is truncation on the outcome — the surviving mix of sold
    to unsold is not the market's mix.

    Returns:
        (sampling dict, warnings).
    """
    warnings: list[str] = []
    out: dict[str, Any] = {}

    terminal_cols = [c for c in ("close_date", "pending_date", "off_market_date")
                     if c in frame.columns]
    if "list_date" in frame.columns and terminal_cols and len(frame):
        listed = pd.to_datetime(frame["list_date"], errors="coerce")
        terminal = frame[terminal_cols].max(axis=1)
        both = listed.notna() & terminal.notna()
        if both.any():
            list_span = int((listed.max() - listed.min()).days)
            term_span = int((terminal.max() - terminal.min()).days)
            active = int((frame["status"] == "ACTIVE").sum()) if "status" in frame else 0
            started_before = int((listed < terminal.min()).sum())
            out = {
                "list_date_min": str(listed.min().date()),
                "list_date_max": str(listed.max().date()),
                "list_date_span_days": list_span,
                "terminal_date_min": str(terminal.min().date()),
                "terminal_date_max": str(terminal.max().date()),
                "terminal_date_span_days": term_span,
                "active_listings": active,
                "rows_started_before_terminal_window": started_before,
                "completed_spell_share": round(float(both.mean()), 4),
            }
            # A genuine listing-date sample has terminations spread at least as
            # widely as its starts, plus a tail of still-active listings.
            if list_span > 0 and term_span < 0.75 * list_span and active == 0:
                warnings.append(
                    "SAMPLE SELECTED ON THE TERMINAL DATE — every listing in this "
                    f"export went off market between {out['terminal_date_min']} and "
                    f"{out['terminal_date_max']} ({term_span} days), while list dates "
                    f"span {list_span} days, and no listing is still ACTIVE. This is a "
                    "sample of terminations in a window, not of listings that started "
                    "in one: a listing begun before the window is present only if it "
                    "lasted into it, and a listing still unsold at the export date is "
                    "absent entirely. Slow, over-priced listings are exactly the ones "
                    "missing, so beta_price is attenuated toward zero and should be "
                    "read as a lower bound on the true elasticity in magnitude. "
                    "Request an export drawn on LIST DATE, including still-active "
                    "listings, before treating the coefficient as an estimate."
                )

    if raw_status is not None and len(raw_status):
        labels = raw_status.astype(str).to_numpy()
        runs: list[tuple[str, int]] = []
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                runs.append((labels[start], i - start))
                start = i
        out["status_runs"] = [{"status": s, "n": n} for s, n in runs]
        distinct = {s for s, _ in runs}
        sorted_by_status = len(runs) == len(distinct) and len(runs) > 1
        out["export_sorted_by_status"] = sorted_by_status
        if sorted_by_status:
            last_status, last_n = runs[-1]
            out["last_status_block"] = {"status": last_status, "n": last_n}
            warnings.append(
                f"EXPORT SORTED BY STATUS — the file arrives in {len(runs)} contiguous "
                f"status blocks ending with {last_status!r} (n={last_n}). If the export "
                "hit a row cap (a search reporting more matches than rows delivered), "
                "the cap truncated that last block, and status is the outcome variable. "
                "The surviving sold-to-unsold mix is then not the market's mix. Confirm "
                "the total the search reported against the rows delivered; if they "
                "differ, re-pull sorted on a field unrelated to the outcome, such as "
                "the listing id."
            )

    return out, warnings


def format_inspect_report(report: IngestReport) -> str:
    """Human-readable --inspect report (stdout deliverable)."""
    lines: list[str] = []
    lines.append("FILES")
    for f in report.files:
        lines.append(f"  {f.path}: {f.rows} rows")
    lines.append(f"  total rows in: {report.rows_in}")
    lines.append(f"  rows out: {report.rows_out}")
    lines.append("")
    lines.append("MAPPING")
    lines.append(f"  canonical fields matched ({report.mapping.get('matched_count', 0)}):")
    lines.append(f"    {', '.join(report.mapping.get('matched') or [])}")
    req_miss = report.mapping.get("required_missing") or []
    lines.append(f"  REQUIRED fields missing: {req_miss if req_miss else 'none'}")
    if req_miss:
        lines.append("  FAIL — required fields absent")
    imp_miss = report.mapping.get("important_missing") or []
    lines.append(f"  IMPORTANT fields missing: {imp_miss if imp_miss else 'none'}")
    unmapped = report.mapping.get("unmapped_source_columns") or {}
    lines.append("  unmapped source columns (by non-null frequency):")
    if unmapped:
        for name, cnt in unmapped.items():
            lines.append(f"    {name}: {cnt}")
    else:
        lines.append("    (none)")
    lines.append("")
    lines.append("TYPES")
    for col, info in (report.types or {}).items():
        fmts = (report.date_formats or {}).get(col)
        fmt_note = f" formats={fmts}" if fmts else ""
        lines.append(
            f"  {col}: null={info['null_count']} ({info['null_rate']}) "
            f"dtype={info['dtype']}{fmt_note}"
        )
    lines.append("")
    lines.append("FILTERS")
    if report.filters:
        for reason, n in report.filters.items():
            lines.append(f"  dropped {n} — {reason}")
    else:
        lines.append("  (none dropped)")
    lines.append(f"  list_ppsf outliers flagged (kept): {report.outlier_flagged}")
    lines.append("")
    lines.append("STATUS")
    for k, v in report.status_distribution.items():
        lines.append(f"  {k}: {v}")
    ident = report.identification
    lines.append(f"  non-SOLD count: {ident.get('non_sold_count', 0)}")
    lines.append(
        f"  PENDING treated as sold (pending_date present): "
        f"{ident.get('pending_treated_as_sold', 0)} of {ident.get('pending_count', 0)}"
    )
    lines.append(f"  survival events (event_sold == 1): {ident.get('event_sold_count', 0)}")
    lines.append("")
    lines.append("FLOOR")
    for k, v in report.floor_source_distribution.items():
        lines.append(f"  floor_source {k}: {v}")
    lines.append(
        f"  parses rejected by total_stories sanity gate: {report.floor_rejected_by_stories}"
    )
    lines.append("")
    lines.append("SUBMARKET COVERAGE")
    cov = report.submarket_coverage or {}
    if cov:
        lines.append(f"  rows with submarket: {cov.get('rows_with_submarket')}")
        lines.append(
            f"  rows without submarket: {cov.get('rows_without_submarket')} "
            f"({cov.get('unmapped_rate')})"
        )
        for z, n in (cov.get("top_unmapped_zips") or {}).items():
            lines.append(f"    unmapped zip {z}: {n}")
    else:
        lines.append("  (not computed)")
    lines.append("")
    lines.append("SAMPLING")
    samp = report.sampling or {}
    if samp:
        lines.append(
            f"  list dates    {samp.get('list_date_min')} .. {samp.get('list_date_max')}"
            f"  ({samp.get('list_date_span_days')} days)"
        )
        lines.append(
            f"  terminal dates {samp.get('terminal_date_min')} .. "
            f"{samp.get('terminal_date_max')}  ({samp.get('terminal_date_span_days')} days)"
        )
        lines.append(f"  still ACTIVE at export: {samp.get('active_listings')}")
        if samp.get("export_sorted_by_status"):
            blocks = ", ".join(
                f"{b['status']}={b['n']}" for b in samp.get("status_runs", [])
            )
            lines.append(f"  export sorted by status: {blocks}")
    else:
        lines.append("  (not computed)")
    lines.append(f"  list_date recovered from DOM: {report.list_date_recovered}")
    lines.append(f"  implausible $/sqft nulled:    {report.list_ppsf_implausible}")
    lines.append("")
    lines.append("IDENTIFICATION READINESS")
    lines.append(f"  SOLD: {ident.get('sold_count')}")
    lines.append(
        f"  censored (EXPIRED/WITHDRAWN/CANCELED): "
        f"{ident.get('censored_expired_withdrawn_canceled')}"
    )
    lines.append(
        f"  (submarket, list_month) cells >= {ident.get('min_cell_listings')}: "
        f"{ident.get('submarket_month_cells_ge_min')}"
    )
    lines.append(
        f"  usable rows (REQUIRED + duration + submarket): "
        f"{ident.get('usable_rows_required_duration_submarket')}"
    )
    lines.append(f"  note: {ident.get('note')}")
    lines.append("")
    if report.warnings:
        lines.append("WARNINGS")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")
    return "\n".join(lines)


def write_ingest_report(
    report: IngestReport, market: str = "miami", name: str = "ingest_report.json"
) -> Path:
    """Write the report under data/processed/{market}/.

    `name` is namespaced by source directory at the CLI so that inspecting a
    synthetic export cannot overwrite the report for the real one.
    """
    out_dir = _BACKEND_ROOT / "data" / "processed" / market
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest MLS exports")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print validation report and write ingest_report.json",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of MLS CSV/XLSX files (default: data/raw/mls)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        default=None,
        help="Explicit MLS file (repeatable)",
    )
    parser.add_argument("--market", default="miami")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        result = ingest_mls(
            paths=args.file,
            market=args.market,
            raw_dir=args.dir,
        )
    except SchemaError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.inspect:
        text = format_inspect_report(result.report)
        print(text)
        name = (
            f"ingest_report_{args.dir.name}.json" if args.dir else "ingest_report.json"
        )
        out = write_ingest_report(result.report, market=args.market, name=name)
        print(f"\nWrote {out}")
        if result.report.warnings:
            return 2
    else:
        print(f"ingested rows_in={result.report.rows_in} rows_out={result.report.rows_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
