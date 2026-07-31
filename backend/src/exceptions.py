"""Domain exceptions for the pricing pipeline."""

from __future__ import annotations


class SchemaError(Exception):
    """Raised when input schema or market config is invalid."""


class IdentificationError(Exception):
    """Raised when demand identification fails (e.g. non-negative β_price)."""


class InfeasibleModelError(Exception):
    """Raised when the optimizer finds no feasible plan."""
