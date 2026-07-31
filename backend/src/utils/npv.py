"""Discounting. Small module, one unit convention, easy to get wrong.

The discount rate is **annual and decimal** everywhere in this codebase: `0.12`,
never `12`. Phase timing is in **months** from project start. The factor is
`1 / (1 + r) ** (months / 12)`, so a phase 18 months out at 12% is worth
`1 / 1.12 ** 1.5 = 0.844` of the same cash today.

A rate passed as a percentage silently makes every future phase worthless and
pushes the optimizer to release everything immediately, which looks like a
plausible answer rather than a units bug. `discount_factor` rejects rates that
are almost certainly percentages.
"""

from __future__ import annotations

import numpy as np

from src.exceptions import SchemaError

MONTHS_PER_YEAR = 12.0
# A rate at or above this is being passed as a percentage. Real annual discount
# rates for development capital sit around 0.08-0.25; 1.0 would be 100%.
_IMPLAUSIBLE_RATE = 1.0


def _validate_rate(annual_rate: float) -> float:
    rate = float(annual_rate)
    if rate <= -1.0:
        raise SchemaError(f"annual discount rate must exceed -1, got {rate}")
    if rate >= _IMPLAUSIBLE_RATE:
        raise SchemaError(
            f"annual discount rate {rate} looks like a percentage. This module "
            "takes an annual decimal rate: 0.12 means 12%."
        )
    return rate


def discount_factor(months: float, annual_rate: float) -> float:
    """Present-value factor for cash arriving `months` from now.

    Args:
        months: months from project start. May be zero; negative is rejected.
        annual_rate: annual decimal rate, e.g. `0.12` for 12%.

    Raises:
        SchemaError: on a negative horizon or a rate that looks like a percentage.
    """
    if months < 0:
        raise SchemaError(f"months must be non-negative, got {months}")
    rate = _validate_rate(annual_rate)
    return float(1.0 / (1.0 + rate) ** (float(months) / MONTHS_PER_YEAR))


def discount_factors(months: np.ndarray | list[float], annual_rate: float) -> np.ndarray:
    """Vectorised `discount_factor`, same units."""
    horizons = np.asarray(months, dtype="float64")
    if (horizons < 0).any():
        raise SchemaError("every horizon in months must be non-negative")
    rate = _validate_rate(annual_rate)
    return 1.0 / (1.0 + rate) ** (horizons / MONTHS_PER_YEAR)


def npv(cash_flows_usd: np.ndarray | list[float], months: np.ndarray | list[float],
        annual_rate: float) -> float:
    """Net present value in USD of cash flows arriving at the given months."""
    flows = np.asarray(cash_flows_usd, dtype="float64")
    horizons = np.asarray(months, dtype="float64")
    if flows.shape != horizons.shape:
        raise SchemaError(
            f"cash flows and horizons must align, got {flows.shape} and {horizons.shape}"
        )
    return float(np.sum(flows * discount_factors(horizons, annual_rate)))
