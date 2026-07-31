"""HTTP routes for Phase 6.

Inventory uploads are JSON (or raw CSV/XLSX bodies). Phase 7 parses files in
the browser with Papa Parse / SheetJS, so multipart is not required.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.schemas import (
    DemandCurveRequest,
    FitBody,
    InventoryValidateRequest,
    OptimizeRequest,
    SensitivityRequest,
    SimulateRequest,
)
from src.api.services import (
    current_demand_metadata,
    inventory_frame_from_csv_bytes,
    inventory_frame_from_xlsx_bytes,
    market_config_payload,
    run_demand_curve,
    run_fit,
    run_optimize,
    run_sensitivity,
    run_simulate,
    validate_rows,
)
from src.exceptions import SchemaError

logger = logging.getLogger(__name__)

router = APIRouter()

# An inventory is a building's unit list: a few hundred rows, a few hundred KB.
# `await request.body()` buffers the whole payload before anything validates it,
# so without a cap a single request can exhaust memory. Both limits are
# operational bounds, not modelling choices.
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_MAX_INVENTORY_ROWS = 20_000


@router.get("/api/config/{market}")
def get_config(market: str) -> dict[str, Any]:
    """Submarkets, view categories, bounds, defaults."""
    return market_config_payload(market)


@router.post("/api/inventory/validate")
async def validate_inventory_route(request: Request) -> dict[str, Any]:
    """Validate developer inventory.

    Accepts:
    - `application/json` body `{market, rows}` (primary — Phase 7 path)
    - `text/csv` raw body with `?market=`
    - Excel body with an xlsx content-type and `?market=`
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    market = request.query_params.get("market", "miami")

    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_UPLOAD_BYTES:
        raise SchemaError(
            f"Inventory upload is {int(declared):,} bytes, over the "
            f"{_MAX_UPLOAD_BYTES:,} byte limit. An inventory is a unit list, not a "
            "dataset; split the file or check what was uploaded."
        )

    raw = b""
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise SchemaError(
                f"Inventory upload exceeds the {_MAX_UPLOAD_BYTES:,} byte limit "
                "(a chunked request understated its length). Split the file or "
                "check what was uploaded."
            )
    if not raw:
        raise SchemaError("Empty body. Send JSON {market, rows} or text/csv.")

    if content_type in {"application/json", "text/json", ""} and raw.lstrip()[:1] in (
        b"{",
        b"[",
    ):
        payload = InventoryValidateRequest.model_validate_json(raw)
        result = validate_rows(payload.rows, payload.market)
    elif "csv" in content_type or content_type in {"text/plain", ""}:
        result = validate_rows(inventory_frame_from_csv_bytes(raw), market)
    elif "spreadsheet" in content_type or content_type.endswith("sheet") or "excel" in content_type:
        result = validate_rows(inventory_frame_from_xlsx_bytes(raw), market)
    else:
        raise SchemaError(
            f"Unsupported content-type {content_type!r}. Send "
            "application/json {{market, rows}}, text/csv, or an xlsx body."
        )

    if result.rows_in > _MAX_INVENTORY_ROWS:
        raise SchemaError(
            f"Inventory has {result.rows_in:,} rows, over the "
            f"{_MAX_INVENTORY_ROWS:,} row limit. The optimizer builds one binary "
            "per unit-phase-price cell, so a file this size is a mis-parse rather "
            "than a project."
        )

    body = result.as_dict()
    body["units"] = result.frame.to_dict(orient="records")
    return body


@router.post("/api/demand/fit")
def demand_fit(body: FitBody) -> JSONResponse:
    """Fit on synthetic or (gated) real MLS. Never auto-flips calibrated flag."""
    payload = run_fit(body)
    # FAILED fits still return the diagnostic body — the caller must read it —
    # but under 422 so a UI does not treat them as a usable model.
    status = 422 if payload["status"] == "FAILED" else 200
    return JSONResponse(status_code=status, content=payload)


@router.get("/api/demand/current")
def demand_current(market: str = Query(default="miami")) -> dict[str, Any]:
    """Active fitted model metadata, including is_calibrated_on_real_data."""
    return current_demand_metadata(market)


@router.post("/api/optimize")
def optimize(body: OptimizeRequest) -> dict[str, Any]:
    """Inventory + constraints → release plan. Always includes provenance."""
    payload = run_optimize(body)
    if "provenance" not in payload:
        raise SchemaError("Internal error: optimize response missing provenance")
    return payload


@router.post("/api/simulate")
async def simulate(body: SimulateRequest) -> StreamingResponse:
    """Plan + uncertainty → revenue distribution, streamed as SSE."""

    async def events() -> AsyncIterator[bytes]:
        n = body.n_draws or 10_000
        yield _sse("progress", {"stage": "starting", "n_draws": n, "drawn": 0})
        # Simulation is synchronous and fast (closed-form). Progress pulses
        # bracket the work so the SSE contract is real without chunking numpy.
        yield _sse("progress", {"stage": "drawing", "n_draws": n, "drawn": 0})
        distribution = run_simulate(body)
        yield _sse(
            "progress",
            {
                "stage": "complete",
                "n_draws": distribution.n_draws,
                "drawn": distribution.n_draws,
            },
        )
        payload = distribution.as_dict()
        if "provenance" not in payload:
            raise SchemaError("Internal error: simulate response missing provenance")
        yield _sse("result", payload)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/api/sensitivity")
def sensitivity(body: SensitivityRequest) -> dict[str, Any]:
    """Tornado (±1σ) and optional LP shadow prices for a fixed plan."""
    tornado_report, shadows = run_sensitivity(body)
    return {
        "tornado": tornado_report.as_dict(),
        "shadow_prices": shadows.as_dict() if shadows else None,
        "provenance": tornado_report.provenance,
    }


@router.post("/api/demand/curve")
def demand_curve(body: DemandCurveRequest) -> dict[str, Any]:
    """P(sell within horizon) vs $/sqft for one unit — makes elasticity visible."""
    return run_demand_curve(body)


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode("utf-8")
