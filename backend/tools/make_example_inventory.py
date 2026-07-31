"""Regenerate `data/project_inputs/example_inventory.csv`.

A 60-unit tower stack, two buildings, three unit types, staggered completion so
the construction gate has something to bite on. Nothing here is estimated: it is
a made-up building used to exercise the optimizer, and the cost basis rises with
elevation the way a real stack's does only because a flat basis would make the
floor premium invisible in the price bands.

Run: `python -m tools.make_example_inventory`
"""

from __future__ import annotations

import csv
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "data" / "project_inputs" / "example_inventory.csv"

_BUILDINGS = (
    # name, submarket, floors, completion, base cost $/sqft
    ("Marina Tower", "brickell", range(6, 26), "2027-06-30", 520.0),
    ("Bayline North", "edgewater", range(4, 24), "2028-03-31", 470.0),
)

# per floor: (line, unit_type, sqft, beds, baths, view)
_STACK = (
    ("01", "one_bed", 780, 1, 1.0, "city_skyline"),
    ("02", "two_bed", 1180, 2, 2.0, "bay_partial"),
    ("03", "three_bed", 1720, 3, 3.0, "bay_direct"),
)
_PENTHOUSE = ("PH", "penthouse", 2650, 4, 4.5, "direct_ocean")

_FIELDS = (
    "unit_id", "floor", "living_area_sqft", "beds", "baths", "view", "unit_type",
    "building", "submarket", "completion_date", "cost_basis_ppsf", "hoa_monthly",
)


_UNITS_PER_BUILDING = 30


def rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for name, submarket, floors, completion, base_cost in _BUILDINGS:
        stack: list[dict[str, object]] = []
        prefix = "".join(word[0] for word in name.split()).upper()
        levels = list(floors)
        for step, floor in enumerate(levels[:-1]):
            for line, unit_type, sqft, beds, baths, view in _STACK:
                stack.append(
                    {
                        "unit_id": f"{prefix}-{floor:02d}{line}",
                        "floor": floor,
                        "living_area_sqft": sqft,
                        "beds": beds,
                        "baths": baths,
                        "view": view,
                        "unit_type": unit_type,
                        "building": name,
                        "submarket": submarket,
                        "completion_date": completion,
                        "cost_basis_ppsf": round(base_cost + 3.5 * step, 2),
                        "hoa_monthly": round(1.15 * sqft, 2),
                    }
                )
        top = levels[-1]
        line, unit_type, sqft, beds, baths, view = _PENTHOUSE
        # One penthouse per tower, kept whatever the trim: it is the unit whose
        # price band is most likely to be cost-bound, which is the case worth
        # exercising.
        stack = stack[: _UNITS_PER_BUILDING - 1]
        stack.append(
            {
                "unit_id": f"{prefix}-{top:02d}{line}",
                "floor": top,
                "living_area_sqft": sqft,
                "beds": beds,
                "baths": baths,
                "view": view,
                "unit_type": unit_type,
                "building": name,
                "submarket": submarket,
                "completion_date": completion,
                "cost_basis_ppsf": round(base_cost + 3.5 * len(levels), 2),
                "hoa_monthly": round(1.35 * sqft, 2),
            }
        )
        out.extend(stack)
    return out


def main() -> int:
    inventory = rows()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(inventory)
    print(f"wrote {len(inventory)} units to {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
