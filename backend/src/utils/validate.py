"""Inventory schema validation.

The inventory is developer-supplied, usually a spreadsheet, and it is the one
input to this system that nobody has cleaned. A bad `living_area_sqft` produces
a plausible-looking total price that is wrong by a factor; a bad
`cost_basis_ppsf` produces a price floor that quietly makes a unit unsellable.

So validation is row-level and it reports rather than repairs. Bad rows are
dropped from the returned frame and every one of them comes back as a
`RowError` naming the row number, the unit, the field, and what was wrong —
which is exactly what `POST /api/inventory/validate` has to render in Phase 6.
Nothing is coerced into plausibility.

Units:
- `living_area_sqft`: sqft. `cost_basis_ppsf`: $/sqft, USD, all-in basis.
- `completion_date`: date the building is physically complete.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import MarketConfig
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "floor",
    "living_area_sqft",
    "beds",
    "unit_type",
    "submarket",
    "completion_date",
    "cost_basis_ppsf",
)
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "baths",
    "view",
    "building",
    "hoa_monthly",
)

# Sanity bands, not estimates. Anything outside is a data-entry error rather
# than an unusual unit, and the point is to catch a decimal in the wrong place.
_AREA_BOUNDS_SQFT = (200.0, 20_000.0)
_FLOOR_BOUNDS = (1, 150)
_BEDS_BOUNDS = (0, 12)
_COST_BASIS_BOUNDS_PPSF = (50.0, 5_000.0)


@dataclass(frozen=True)
class RowError:
    """One thing wrong with one row, addressed so a user can go fix it."""

    row_number: int
    unit_id: str | None
    field: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryValidation:
    """Valid rows, plus a full account of what was rejected and why."""

    frame: pd.DataFrame
    errors: list[RowError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_in: int = 0

    @property
    def rows_valid(self) -> int:
        return len(self.frame)

    @property
    def is_valid(self) -> bool:
        """True when at least one row was submitted and every one survived.

        An empty upload is not a clean upload. Reporting zero rows as valid
        would let a mis-parsed file sail through to an optimizer that then has
        nothing to plan and no reason it can name.
        """
        return bool(self.frame is not None and len(self.frame) > 0 and not self.errors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_valid": self.rows_valid,
            "is_valid": self.is_valid,
            "errors": [e.as_dict() for e in self.errors],
            "warnings": self.warnings,
        }


def _numeric(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def validate_inventory(
    frame: pd.DataFrame, config: MarketConfig | None = None
) -> InventoryValidation:
    """Validate a developer inventory frame row by row.

    Args:
        frame: raw inventory, one row per unit.
        config: market config; supplies the submarket whitelist. Without it,
            submarkets are accepted as given.

    Returns:
        InventoryValidation carrying only the rows that passed.

    Raises:
        SchemaError: when a required *column* is absent. That is a malformed
            file rather than a bad row, and there is nothing per-row to report.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"Inventory is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )

    known_submarkets = set((config or {}).get("submarkets") or {})
    errors: list[RowError] = []
    warnings: list[str] = []
    keep: list[bool] = []

    seen_ids: dict[str, int] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        # Row numbers are 1-based over data rows, matching what a spreadsheet
        # shows once the header is accounted for.
        row_number = position + 1
        raw_id = row.get("unit_id")
        unit_id = None if pd.isna(raw_id) else str(raw_id).strip()
        row_errors: list[RowError] = []

        def fail(field_name: str, message: str) -> None:
            row_errors.append(RowError(row_number, unit_id, field_name, message))

        if not unit_id:
            fail("unit_id", "missing; every unit needs an identifier")
        elif unit_id in seen_ids:
            fail("unit_id", f"duplicate of row {seen_ids[unit_id]}")
        elif unit_id:
            seen_ids[unit_id] = row_number

        area = _numeric(row.get("living_area_sqft"))
        if area is None:
            fail("living_area_sqft", "missing or not a number")
        elif not _AREA_BOUNDS_SQFT[0] <= area <= _AREA_BOUNDS_SQFT[1]:
            fail(
                "living_area_sqft",
                f"{area:g} sqft is outside {_AREA_BOUNDS_SQFT[0]:g}-"
                f"{_AREA_BOUNDS_SQFT[1]:g}; check the units",
            )

        floor = _numeric(row.get("floor"))
        if floor is None:
            fail("floor", "missing or not a number")
        elif not _FLOOR_BOUNDS[0] <= floor <= _FLOOR_BOUNDS[1]:
            fail("floor", f"{floor:g} is outside {_FLOOR_BOUNDS[0]}-{_FLOOR_BOUNDS[1]}")

        beds = _numeric(row.get("beds"))
        if beds is None:
            fail("beds", "missing or not a number")
        elif not _BEDS_BOUNDS[0] <= beds <= _BEDS_BOUNDS[1]:
            fail("beds", f"{beds:g} is outside {_BEDS_BOUNDS[0]}-{_BEDS_BOUNDS[1]}")

        cost = _numeric(row.get("cost_basis_ppsf"))
        if cost is None:
            fail("cost_basis_ppsf", "missing or not a number; the price floor depends on it")
        elif not _COST_BASIS_BOUNDS_PPSF[0] <= cost <= _COST_BASIS_BOUNDS_PPSF[1]:
            fail(
                "cost_basis_ppsf",
                f"${cost:,.0f}/sqft is outside ${_COST_BASIS_BOUNDS_PPSF[0]:,.0f}-"
                f"${_COST_BASIS_BOUNDS_PPSF[1]:,.0f}; this should be $/sqft, not a total",
            )

        completion = pd.to_datetime(row.get("completion_date"), errors="coerce")
        if pd.isna(completion):
            fail(
                "completion_date",
                "missing or unparseable; the construction gate cannot be applied without it",
            )

        unit_type = row.get("unit_type")
        if pd.isna(unit_type) or not str(unit_type).strip():
            fail("unit_type", "missing; needed for the diversity constraint")

        submarket = row.get("submarket")
        if pd.isna(submarket) or not str(submarket).strip():
            fail("submarket", "missing; demand is estimated per submarket")
        elif known_submarkets and str(submarket).strip() not in known_submarkets:
            fail(
                "submarket",
                f"{submarket!r} is not in the market config; known: "
                f"{sorted(known_submarkets)[:5]}",
            )

        errors.extend(row_errors)
        keep.append(not row_errors)

    valid = frame.loc[pd.Series(keep, index=frame.index)].copy()
    if not valid.empty:
        valid["unit_id"] = valid["unit_id"].astype(str).str.strip()
        valid["completion_date"] = pd.to_datetime(valid["completion_date"], errors="coerce")
        for column in ("living_area_sqft", "floor", "beds", "cost_basis_ppsf"):
            valid[column] = pd.to_numeric(valid[column], errors="coerce")

    if not len(frame):
        warnings.append(
            "The inventory has the right columns but no rows. Nothing to validate, "
            "and nothing to plan — check the upload parsed."
        )

    absent_optional = [c for c in OPTIONAL_COLUMNS if c not in frame.columns]
    if absent_optional:
        warnings.append(
            f"optional columns absent: {absent_optional}. The hedonic surface prices "
            "whatever it is given; anything missing stays unpriced rather than guessed."
        )
    if errors:
        warnings.append(
            f"{len(errors)} row-level errors across {len(frame) - len(valid)} rows. "
            "Those units are excluded from the plan entirely, not priced with defaults."
        )

    logger.info("Inventory validated: %d of %d rows usable", len(valid), len(frame))
    return InventoryValidation(
        frame=valid, errors=errors, warnings=warnings, rows_in=len(frame)
    )


def inventory_scoring_frame(units: pd.DataFrame) -> pd.DataFrame:
    """Derive the unit-level columns the fitted models were trained on.

    The inventory speaks the developer's vocabulary (`baths`, `view`,
    `hoa_monthly`); the demand and hedonic models speak the MLS feature
    vocabulary (`baths_full`, `view_description`, `hoa_per_sqft`). This is the
    shim, and it lives next to the schema so there is one place where the two
    vocabularies are reconciled.

    Only *deterministic* re-expressions happen here: a log, a division, a
    rename. Anything the inventory does not carry comes out null and stays null.
    Time-varying columns are the caller's job — they depend on which phase a
    unit is being scored in.

    Units: `hoa_per_sqft` is $/sqft/month; `log_living_area` is log(sqft).
    """
    out = units.copy()
    floor = pd.to_numeric(out.get("floor"), errors="coerce")
    area = pd.to_numeric(out.get("living_area_sqft"), errors="coerce")

    out["log_floor"] = np.where(floor >= 0, np.log(floor.to_numpy(dtype="float64") + 1.0), np.nan)
    out["log_living_area"] = np.where(area > 0, np.log(area.to_numpy(dtype="float64")), np.nan)

    if "baths" in out.columns:
        baths = pd.to_numeric(out["baths"], errors="coerce")
        out["baths_full"] = np.floor(baths)
        out["baths_half"] = ((baths - np.floor(baths)) >= 0.5).astype("float64")
    else:
        out["baths_full"] = np.nan
        out["baths_half"] = np.nan

    if "hoa_monthly" in out.columns:
        hoa = pd.to_numeric(out["hoa_monthly"], errors="coerce")
        out["hoa_per_sqft"] = np.where((area > 0) & hoa.notna(), hoa / area, np.nan)
    else:
        out["hoa_per_sqft"] = np.nan

    if "view" in out.columns and "view_description" not in out.columns:
        out["view_description"] = out["view"]

    # The demand model regresses on view *indicators*, not on the raw string, so
    # the shim has to speak the same vocabulary. Without this an inventory
    # carrying `view` is unscorable against a model fitted with view tokens —
    # `transform_to_design` raises "missing fitted covariates" and the whole
    # plan fails. Tokens the inventory does not exercise come out zero, which is
    # correct: a unit with no ocean view has no ocean-view indicator set.
    if "view_description" in out.columns:
        from src.data.features import tokenize_multivalue

        block = tokenize_multivalue(out["view_description"], "view")
        for column in block.columns:
            out[column] = block[column]

    # Every unit in a developer's release pipeline is new construction. This is
    # a definition, not an inference from year_built.
    out["is_new_construction"] = pd.Series(True, index=out.index, dtype="boolean")
    return out


def format_validation(result: InventoryValidation) -> str:
    """Human-readable validation summary."""
    lines = [
        f"INVENTORY  {result.rows_valid} of {result.rows_in} rows usable "
        f"({'clean' if result.is_valid else 'errors present'})"
    ]
    if result.errors:
        lines.append("")
        lines.append("ROW ERRORS")
        for error in result.errors[:50]:
            unit = error.unit_id or "?"
            lines.append(f"  row {error.row_number} [{unit}] {error.field}: {error.message}")
        if len(result.errors) > 50:
            lines.append(f"  ... and {len(result.errors) - 50} more")
    if result.warnings:
        lines.append("")
        lines.append("WARNINGS")
        for warning in result.warnings:
            lines.append(f"  !  {warning}")
    return "\n".join(lines)
