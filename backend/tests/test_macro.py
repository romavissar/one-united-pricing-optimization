"""Macro derivation — the mappings are transparent, so test them as such.

No test here touches the network: a fake httpx-like client returns planted
series, and the assertions check that the derivation recovers what was planted
(a known volatility, a sign, a correlation) and that every degradation path
(missing key, missing series, offline switch) lands on the documented fallback
with the right provenance label.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data import macro as m
from src.data.macro import (
    MacroSeries,
    build_macro_snapshot,
    channel_dispersion,
    estimate_correlations,
    format_macro_snapshot,
)


# ── Fakes ──────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _monthly_dates(n: int, start_year: int = 2016) -> list[str]:
    dates = []
    year, month = start_year, 1
    for _ in range(n):
        dates.append(f"{year}-{month:02d}-01")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return dates


def _fred_payload(dates: list[str], values: list[float]) -> dict:
    return {
        "observations": [
            {"date": d, "value": ("." if v is None else f"{v}")}
            for d, v in zip(dates, values, strict=True)
        ]
    }


def _bls_payload(series_id: str, dates: list[str], values: list[float]) -> dict:
    data = []
    for d, v in zip(dates, values, strict=True):
        year, month, _ = d.split("-")
        data.append(
            {"year": year, "period": f"M{month}", "periodName": month, "value": f"{v}"}
        )
    return {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {"series": [{"seriesID": series_id, "data": list(reversed(data))}]},
    }


class _FakeClient:
    """Serves planted FRED (GET) and BLS (POST) responses by series id."""

    def __init__(self, fred: dict[str, dict], bls: dict[str, dict]) -> None:
        self._fred = fred  # series_id -> payload
        self._bls = bls  # series_id -> payload
        self.closed = False

    def get(self, url, params=None, timeout=None):  # noqa: ANN001
        sid = params["series_id"]
        return _FakeResponse(self._fred[sid])

    def post(self, url, json=None, timeout=None):  # noqa: ANN001
        ids = json["seriesid"]
        series = []
        for sid in ids:
            if sid in self._bls:
                series.append(self._bls[sid]["Results"]["series"][0])
        return _FakeResponse(
            {"status": "REQUEST_SUCCEEDED", "message": [], "Results": {"series": series}}
        )

    def close(self) -> None:
        self.closed = True


def _planted_client(seed: int = 0) -> _FakeClient:
    """A client whose series carry known volatilities and a known co-movement."""
    rng = np.random.default_rng(seed)
    n = 120
    dates = _monthly_dates(n)

    # Home price index: geometric random walk with monthly log-return sd 0.01.
    hpi_ret = rng.normal(0.003, 0.01, n - 1)
    hpi = 300.0 * np.exp(np.concatenate([[0.0], np.cumsum(hpi_ret)]))

    # Days on market: moves *opposite* to prices (faster market when prices
    # rise) so absorption (−Δlog DOM) correlates positively with comps_drift.
    dom_ret = -0.8 * hpi_ret + rng.normal(0.0, 0.02, n - 1)
    dom = 60.0 * np.exp(np.concatenate([[0.0], np.cumsum(dom_ret)]))

    # Active listings: independent-ish, monthly log-return sd ~0.05.
    lst_ret = rng.normal(0.0, 0.05, n - 1)
    listings = 12000.0 * np.exp(np.concatenate([[0.0], np.cumsum(lst_ret)]))

    rate = 6.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    cpi = 300.0 * np.exp(np.cumsum(rng.normal(0.002, 0.001, n)))
    unemp = 4.0 + rng.normal(0.0, 0.2, n)

    fred = {
        "MIXRSA": _fred_payload(dates, list(hpi)),
        "MEDDAYONMAR33100": _fred_payload(dates, list(dom)),
        "ACTLISCOU33100": _fred_payload(dates, list(listings)),
        "MORTGAGE30US": _fred_payload(dates, list(rate)),
    }
    bls = {
        "CUURS35CSA0": _bls_payload("CUURS35CSA0", dates, list(cpi)),
        "LAUMT123310000000003": _bls_payload("LAUMT123310000000003", dates, list(unemp)),
    }
    return _FakeClient(fred, bls)


# ── Pure computation ───────────────────────────────────────────────────────

def test_channel_dispersion_recovers_planted_volatility_and_scales_with_horizon():
    # A pure random walk with monthly log-return sd = 0.02.
    rng = np.random.default_rng(1)
    n = 400
    ret = rng.normal(0.0, 0.02, n - 1)
    level = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(ret)]))
    series = MacroSeries(
        series_id="X", source="fred", title="x",
        dates=tuple(_monthly_dates(n)), values=tuple(level), periods_per_year=12.0,
    )
    sd_1mo = channel_dispersion(series, "log_return", horizon_days=30)
    # ~0.02 over one month, within sampling error of 400 obs.
    assert 0.015 < sd_1mo < 0.025
    # √-time scaling: a 6-month σ is ~√6 the 1-month σ.
    sd_6mo = channel_dispersion(series, "log_return", horizon_days=180)
    assert sd_6mo == pytest.approx(sd_1mo * np.sqrt(6.0), rel=0.05)


def test_channel_dispersion_zero_when_too_short():
    series = MacroSeries(
        series_id="X", source="fred", title="x",
        dates=("2020-01-01",), values=(100.0,), periods_per_year=12.0,
    )
    assert channel_dispersion(series, "log_return", 180) == 0.0


def test_estimate_correlations_recovers_planted_sign():
    client = _planted_client(seed=3)
    # Reach into the fetch by building a snapshot and inspecting correlations.
    snap = build_macro_snapshot("miami", 180, use_cache=False, client=client)
    # comps_drift and absorption were planted to co-move positively.
    assert snap.correlations["comps_drift|absorption"] > 0.3


# ── Snapshot orchestration ─────────────────────────────────────────────────

def test_build_snapshot_live_shape_and_units(monkeypatch):
    monkeypatch.delenv("MACRO_DISABLE_NETWORK", raising=False)
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})
    client = _planted_client(seed=5)
    snap = build_macro_snapshot("miami", 180, use_cache=False, client=client)

    assert snap.source == "fred_bls"
    assert snap.is_live
    assert client.closed is False  # caller owns the injected client, we don't close it
    # Absolute channels feed spec fields directly; competing is relative.
    assert set(snap.dispersions) == {"comps_drift_sd", "absorption_log_hazard_sd"}
    assert set(snap.relative_dispersions) == {"competing_listings_sd"}
    assert snap.dispersions["comps_drift_sd"] > 0
    # BLS context is present (both APIs actually used).
    assert "home_price_appreciation_yoy_real" in snap.context
    assert "unemployment_rate_pct" in snap.context


def test_scenario_dispersions_scale_relative_channel_by_baseline(monkeypatch):
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})
    snap = build_macro_snapshot(
        "miami", 180, use_cache=False, client=_planted_client(seed=7)
    )
    rel = snap.relative_dispersions["competing_listings_sd"]
    resolved = snap.scenario_dispersions(competing_listings_baseline=40.0)
    assert resolved["competing_listings_sd"] == pytest.approx(rel * 40.0)
    # Absolute channels are unchanged by the baseline.
    assert resolved["comps_drift_sd"] == snap.dispersions["comps_drift_sd"]


def test_absorption_sign_is_inverse_of_days_on_market(monkeypatch):
    """A market that slows (DOM ↑) must be a *negative* absorption co-move."""
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})
    snap = build_macro_snapshot(
        "miami", 180, use_cache=False, client=_planted_client(seed=11)
    )
    # DOM was planted to fall as prices rise, so absorption (−Δlog DOM) rises
    # with prices: a positive correlation with comps_drift confirms the sign flip.
    assert snap.correlations["comps_drift|absorption"] > 0


# ── Degradation paths ──────────────────────────────────────────────────────

def test_offline_switch_returns_labelled_fallback(monkeypatch):
    monkeypatch.setenv("MACRO_DISABLE_NETWORK", "1")
    snap = build_macro_snapshot("miami", 180, use_cache=False)
    assert snap.source == "static_fallback"
    assert not snap.is_live
    assert snap.dispersions["absorption_log_hazard_sd"] == 0.15
    assert any("MACRO_DISABLE_NETWORK" in w for w in snap.warnings)


def test_missing_fred_key_falls_back(monkeypatch):
    monkeypatch.delenv("MACRO_DISABLE_NETWORK", raising=False)
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": None, "bls": None})
    snap = build_macro_snapshot("miami", 180, use_cache=False)
    assert snap.source == "static_fallback"
    assert any("FRED_API_KEY" in w for w in snap.warnings)


def test_missing_required_series_falls_back(monkeypatch):
    monkeypatch.delenv("MACRO_DISABLE_NETWORK", raising=False)
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})

    class _BrokenClient(_FakeClient):
        def get(self, url, params=None, timeout=None):  # noqa: ANN001
            raise RuntimeError("network down")

    snap = build_macro_snapshot(
        "miami", 180, use_cache=False, client=_BrokenClient({}, {})
    )
    assert snap.source == "static_fallback"
    assert any("required FRED series missing" in w for w in snap.warnings)


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("MACRO_DISABLE_NETWORK", raising=False)
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})
    monkeypatch.setattr(m, "_cache_path", lambda market: tmp_path / f"{market}.json")

    first = build_macro_snapshot(
        "miami", 180, use_cache=True, client=_planted_client(seed=2)
    )
    assert first.source == "fred_bls"
    # A second call with cache on must not need a client at all.
    second = build_macro_snapshot("miami", 180, use_cache=True, client=None)
    assert second.source == "cache"
    assert second.dispersions == first.dispersions
    # A different horizon invalidates the cache (a σ is horizon-specific).
    cached_other = m._load_cache("miami", 360, max_age_days=7)
    assert cached_other is None


def test_format_snapshot_is_stringable(monkeypatch):
    monkeypatch.setenv("MACRO_DISABLE_NETWORK", "1")
    snap = build_macro_snapshot("miami", 180, use_cache=False)
    text = format_macro_snapshot(snap)
    assert "MACRO" in text and "DISPERSIONS" in text


# ── Scenario-spec wiring (data-driven default, custom override) ─────────────

from types import SimpleNamespace

from src.api.schemas import PhaseInput, ScenarioInput
from src.api.services import _competing_listings_baseline, _scenario_spec


def _coef(name: str, value: float, se: float = 0.1):
    return SimpleNamespace(name=name, value=value, std_error=se)


def _fake_bundle(*, with_inventory: bool):
    coeffs = {"rel_price_premium": _coef("rel_price_premium", -1.6)}
    if with_inventory:
        coeffs["inventory_competition"] = _coef("inventory_competition", -0.02)
    cox = SimpleNamespace(
        beta_price=_coef("rel_price_premium", -1.6, 0.11),
        coefficients=coeffs,
    )
    return SimpleNamespace(cox_result=cox)


def _live_snapshot(monkeypatch):
    monkeypatch.setattr(m, "_read_keys", lambda: {"fred": "k", "bls": "k"})
    return build_macro_snapshot(
        "miami", 180, use_cache=False, client=_planted_client(seed=9)
    )


def test_scenario_spec_uses_data_when_no_override(monkeypatch):
    snap = _live_snapshot(monkeypatch)
    spec, resolution = _scenario_spec(
        _fake_bundle(with_inventory=True),
        ScenarioInput(),  # all None → data-driven
        snap,
        competing_baseline=30.0,
    )
    assert resolution["comps_drift_sd"]["source"] == "fred_bls"
    assert spec.comps_drift_sd == pytest.approx(snap.dispersions["comps_drift_sd"])
    assert spec.absorption_log_hazard_sd == pytest.approx(
        snap.dispersions["absorption_log_hazard_sd"]
    )
    # beta_price SE stays the fitted value.
    assert spec.beta_price_se == pytest.approx(0.11)
    assert resolution["beta_price_se"]["source"] == "fitted"


def test_scenario_spec_honors_custom_override_per_channel(monkeypatch):
    snap = _live_snapshot(monkeypatch)
    spec, resolution = _scenario_spec(
        _fake_bundle(with_inventory=True),
        ScenarioInput(comps_drift_sd=0.99),  # only this channel is custom
        snap,
        competing_baseline=30.0,
    )
    assert spec.comps_drift_sd == 0.99
    assert resolution["comps_drift_sd"]["source"] == "custom"
    # The un-overridden channel is still data-driven.
    assert resolution["absorption_log_hazard_sd"]["source"] == "fred_bls"


def test_competing_listings_disabled_without_inventory_coefficient(monkeypatch):
    snap = _live_snapshot(monkeypatch)
    spec, resolution = _scenario_spec(
        _fake_bundle(with_inventory=False),
        ScenarioInput(),
        snap,
        competing_baseline=30.0,
    )
    assert spec.competing_listings_sd == 0.0
    assert resolution["competing_listings_sd"]["source"] == "disabled"


def test_competing_listings_active_with_inventory_coefficient(monkeypatch):
    snap = _live_snapshot(monkeypatch)
    spec, _ = _scenario_spec(
        _fake_bundle(with_inventory=True),
        ScenarioInput(),
        snap,
        competing_baseline=30.0,
    )
    expected = snap.relative_dispersions["competing_listings_sd"] * 30.0
    assert spec.competing_listings_sd == pytest.approx(expected)


def test_competing_listings_baseline_is_phase_median():
    phases = [
        PhaseInput(name="a", start_month=0.0, competing_listings=20.0),
        PhaseInput(name="b", start_month=6.0, competing_listings=40.0),
    ]
    assert _competing_listings_baseline(phases) == 30.0
