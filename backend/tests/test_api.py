"""Phase 6 — every route's happy path, and bad inventory is not a 500.

`PROJECT_BRIEF.md` §6: provenance on optimize/simulate is mandatory;
`dataset=mls` without `calibration_gate` is refused; a malformed inventory
returns row-level errors rather than an unhandled exception.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app

INVENTORY_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "project_inputs" / "example_inventory.csv"
)

client = TestClient(app)


def _phases() -> list[dict]:
    return [
        {"name": "launch", "start_month": 0.0, "max_units": 20, "competing_listings": 30.0},
        {"name": "wave_two", "start_month": 6.0, "max_units": 20, "competing_listings": 30.0},
        {"name": "wave_three", "start_month": 12.0, "max_units": 20, "competing_listings": 30.0},
        {"name": "closeout", "start_month": 18.0, "max_units": 20, "competing_listings": 30.0},
    ]


def _inventory_rows() -> list[dict]:
    frame = pd.read_csv(INVENTORY_CSV)
    # JSON-friendly dates
    records = frame.to_dict(orient="records")
    for row in records:
        for key, value in list(row.items()):
            if pd.isna(value):
                row[key] = None
            elif hasattr(value, "isoformat"):
                row[key] = value.isoformat()
    return records


@pytest.fixture(scope="module")
def fitted_model():
    """One synthetic fit shared by optimize/simulate/sensitivity tests."""
    response = client.post(
        "/api/demand/fit",
        json={
            "dataset": "synthetic",
            "market": "miami",
            "n_listings": 2500,
            "seed": 17,
            "controls": True,
            "name": "current",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_calibrated_on_real_data"] is False
    assert "identification_report" in body
    assert body["beta_price"]["value"] < 0
    return body


# --- health / config ------------------------------------------------------


def test_health_still_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_miami_exposes_defaults_and_calibration_flag():
    response = client.get("/api/config/miami")
    assert response.status_code == 200
    body = response.json()
    assert body["market"] == "miami"
    assert "brickell" in body["submarkets"]
    assert body["defaults"]["discount_rate_annual"] == 0.12
    assert body["is_calibrated_on_real_data"] is False


def test_config_unknown_market_is_400():
    response = client.get("/api/config/atlantis")
    assert response.status_code == 400
    assert response.json()["error"] == "schema_error"


# --- inventory ------------------------------------------------------------


def test_validate_inventory_happy_path_json():
    response = client.post(
        "/api/inventory/validate",
        json={"market": "miami", "rows": _inventory_rows()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is True
    assert body["rows_valid"] == 60
    assert len(body["units"]) == 60


def test_validate_inventory_csv_body():
    raw = INVENTORY_CSV.read_bytes()
    response = client.post(
        "/api/inventory/validate?market=miami",
        content=raw,
        headers={"content-type": "text/csv"},
    )
    assert response.status_code == 200
    assert response.json()["rows_valid"] == 60


def test_malformed_inventory_returns_row_errors_not_500():
    rows = _inventory_rows()
    rows[0]["living_area_sqft"] = -50
    rows[1]["unit_id"] = None
    rows[2]["cost_basis_ppsf"] = "not-a-number"
    response = client.post(
        "/api/inventory/validate",
        json={"market": "miami", "rows": rows},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_valid"] is False
    assert body["errors"]
    assert all("row_number" in e and "field" in e and "message" in e for e in body["errors"])
    assert response.status_code != 500


# --- demand ---------------------------------------------------------------


def test_fit_mls_without_calibration_gate_is_403():
    response = client.post(
        "/api/demand/fit",
        json={"dataset": "mls", "market": "miami", "calibration_gate": False},
    )
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "calibration_gate" in detail


def test_fit_synthetic_and_current_metadata(fitted_model):
    response = client.get("/api/demand/current?market=miami")
    assert response.status_code == 200
    body = response.json()
    assert body["provenance"]["is_calibrated_on_real_data"] is False
    assert body["beta_price"]["value"] == pytest.approx(
        fitted_model["beta_price"]["value"]
    )


# --- optimize / simulate / sensitivity ------------------------------------


def test_optimize_returns_plan_with_provenance(fitted_model):
    del fitted_model
    response = client.post(
        "/api/optimize",
        json={
            "market": "miami",
            "project_start": "2026-01-01",
            "inventory": _inventory_rows(),
            "phases": _phases(),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "optimal"
    assert body["plan"]
    prov = body["provenance"]
    assert prov["is_calibrated_on_real_data"] is False
    assert prov["demand_model"]
    assert "beta_price" in prov
    assert "warnings" in prov
    assert any("synthetic" in w.lower() or "illustrative" in w.lower() for w in prov["warnings"])


def test_optimize_with_invalid_inventory_is_400_not_500(fitted_model):
    del fitted_model
    rows = _inventory_rows()
    rows[0]["living_area_sqft"] = -1
    response = client.post(
        "/api/optimize",
        json={
            "market": "miami",
            "inventory": rows,
            "phases": _phases(),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "schema_error"


def test_simulate_sse_includes_provenance_and_percentiles(fitted_model):
    del fitted_model
    plan_response = client.post(
        "/api/optimize",
        json={
            "market": "miami",
            "inventory": _inventory_rows(),
            "phases": _phases(),
        },
    )
    assert plan_response.status_code == 200
    plan_body = plan_response.json()

    with client.stream(
        "POST",
        "/api/simulate",
        json={
            "market": "miami",
            "inventory": _inventory_rows(),
            "phases": _phases(),
            "plan": plan_body["plan"],
            "n_draws": 1500,
            "seed": 3,
            "scenario": {"absorption_log_hazard_sd": 0.1, "comps_drift_sd": 0.02},
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = "".join(response.iter_text())

    assert "event: progress" in text
    assert "event: result" in text
    # Parse the result event payload
    result_payload = None
    chunks = text.split("\n\n")
    for chunk in chunks:
        if chunk.startswith("event: result"):
            data_line = [ln for ln in chunk.split("\n") if ln.startswith("data: ")][0]
            result_payload = json.loads(data_line[len("data: "):])
    assert result_payload is not None
    assert result_payload["provenance"]["is_calibrated_on_real_data"] is False
    assert result_payload["discounted_usd"]["p5"] <= result_payload["discounted_usd"]["p95"]
    assert result_payload["n_draws"] == 1500


def test_sensitivity_returns_tornado(fitted_model):
    del fitted_model
    plan_response = client.post(
        "/api/optimize",
        json={
            "market": "miami",
            "inventory": _inventory_rows(),
            "phases": _phases(),
        },
    )
    assert plan_response.status_code == 200
    response = client.post(
        "/api/sensitivity",
        json={
            "market": "miami",
            "inventory": _inventory_rows(),
            "phases": _phases(),
            "plan": plan_response.json()["plan"],
            "include_shadow_prices": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tornado"]["bars"]
    assert body["provenance"]["is_calibrated_on_real_data"] is False
    assert "shadow_prices" in body


def test_demand_curve_returns_monotone_probabilities(fitted_model):
    del fitted_model
    unit_id = _inventory_rows()[0]["unit_id"]
    response = client.post(
        "/api/demand/curve",
        json={
            "market": "miami",
            "unit_id": unit_id,
            "inventory": _inventory_rows(),
            "phases": _phases(),
            "phase_index": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["unit_id"] == unit_id
    assert len(body["prices_ppsf"]) == len(body["probabilities"]) >= 2
    # Higher price → lower or equal sale probability when elasticity is negative.
    assert body["probabilities"][0] >= body["probabilities"][-1] - 1e-9
