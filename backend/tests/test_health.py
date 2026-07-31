"""Phase 0 health endpoint smoke test."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_load_miami_config() -> None:
    from src.config import load_market_config

    cfg = load_market_config("miami")
    assert cfg["market"] == "miami"
    assert cfg["area_unit"] == "sqft"
    assert cfg["currency"] == "USD"
    assert "brickell" in cfg["submarkets"]
    assert cfg["defaults"]["discount_rate_annual"] == 0.12
