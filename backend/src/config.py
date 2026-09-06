"""Load and validate market configuration YAML.

Price units in config are market-specific ($/sqft for Miami, RON/sqm for Bucharest).
Discount rates are annual decimal (e.g. 0.12).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

import yaml

from src.exceptions import SchemaError

__all__ = [
    "MarketConfig",
    "SchemaError",
    "load_market_config",
    "validate_market_id",
    "zip_to_submarket",
]


class MarketConfig(TypedDict, total=False):
    market: str
    currency: str
    area_unit: str
    price_unit: str
    submarkets: dict[str, Any]
    view_categories: list[str]
    defaults: dict[str, Any]
    filters: dict[str, Any]
    notes: dict[str, Any]


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_REQUIRED_TOP_LEVEL = (
    "market",
    "currency",
    "area_unit",
    "price_unit",
    "submarkets",
    "view_categories",
    "defaults",
    "filters",
)


_MARKET_ID = re.compile(r"^[a-z0-9_]{1,32}$")


def validate_market_id(market: str) -> str:
    """Reject anything that is not a bare market id before it reaches a path.

    `market` arrives from the URL (`/api/config/{market}`) and from request
    bodies, and it is joined into filesystem paths in two places: the config
    file here, and the model bundle directory in `registry.model_dir`. The
    `.yaml` suffix and the `market:` field check below make traversal hard to
    exploit today, but "hard to exploit" is not a containment check — and
    `model_dir` has neither. One allowlist at the entry point is the fix.

    Raises:
        SchemaError: when the id contains anything but lowercase alphanumerics
            and underscores.
    """
    candidate = str(market)
    if not _MARKET_ID.match(candidate):
        raise SchemaError(
            f"Invalid market id {candidate!r}. A market id is lowercase letters, "
            "digits, and underscores — it names a config file and a model "
            "directory, so it is never a path."
        )
    return candidate


def load_market_config(market: str) -> MarketConfig:
    """Load `config/{market}.yaml` and validate required top-level keys.

    Args:
        market: Market id (e.g. ``\"miami\"``). Selects ``config/{market}.yaml``.

    Returns:
        Parsed market config dict.

    Raises:
        SchemaError: If the file is missing or required keys are absent.
    """
    market = validate_market_id(market)
    path = _CONFIG_DIR / f"{market}.yaml"
    if not path.is_file():
        raise SchemaError(f"Market config not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise SchemaError(f"Market config must be a mapping: {path}")

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise SchemaError(
            f"Market config {path.name} missing required keys: {missing}"
        )

    if raw.get("market") != market:
        raise SchemaError(
            f"Config market field {raw.get('market')!r} does not match "
            f"requested market {market!r}"
        )

    return raw  # type: ignore[return-value]


CANCELLED_TREATMENTS = ("censored", "excluded", "event")


def cancelled_treatment(config: MarketConfig | None) -> str:
    """How a CANCELED listing is coded in the survival panel.

    27% of the quarterly export is Cancelled, and what it means is genuinely
    ambiguous — which is why it is a setting rather than a constant.

    * `censored` (default, and the historical behaviour): the listing was
      offered, observed for its duration, and did not sell. Treats a
      cancellation like an expiry.
    * `excluded`: drops the rows. Appropriate if cancellations are mostly
      administrative — a relist under a new agent, a seller changing brokerage —
      in which case they are neither a sale nor a market rejection and carry no
      information about demand.
    * `event`: codes them as sales. Only defensible if cancellations are
      predominantly off-market transactions, which nothing in this data
      supports; provided so the assumption can be *tested* rather than argued
      about.

    Raises:
        SchemaError: on an unknown value, rather than silently defaulting —
            a typo here would quietly change every coefficient.
    """
    value = str(((config or {}).get("defaults") or {}).get("cancelled_treatment", "censored"))
    if value not in CANCELLED_TREATMENTS:
        raise SchemaError(
            f"cancelled_treatment must be one of {list(CANCELLED_TREATMENTS)}, "
            f"got {value!r}. This decides how 27% of the sample is coded, so an "
            "unrecognised value is not something to guess past."
        )
    return value


def zip_to_submarket(zip_code: str | None, config: MarketConfig) -> str | None:
    """Map a 5-digit ZIP string to a submarket id from market config.

    Returns None when zip is missing or not listed (never imputes).
    """
    if zip_code is None:
        return None
    z = str(zip_code).strip()
    if not z or z.lower() in {"nan", "none"}:
        return None
    z = z[:5]
    for name, meta in (config.get("submarkets") or {}).items():
        zips = meta.get("zips") if isinstance(meta, dict) else None
        if zips and z in zips:
            return str(name)
    return None
