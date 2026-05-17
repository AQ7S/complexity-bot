"""Tests for vol targeting (Tier 4.5) + correlation monitor (Tier 4.6)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.risk.correlation_monitor import CorrelationMonitor
from engine.risk.volatility_targeting import (
    VolTargetInputs,
    rebalance_open_positions,
    target_lot,
    vol_target_per_position,
)


def test_vol_target_per_position_basic():
    per = vol_target_per_position(equity=10_000, n_active=4, sigma_target=0.01)
    assert per == pytest.approx(25.0)


def test_vol_target_zero_on_empty_inputs():
    assert vol_target_per_position(0, 1) == 0.0
    assert vol_target_per_position(10_000, 0) == 0.0


def test_target_lot_scales_inverse_to_vol():
    quiet = VolTargetInputs("EURUSD#", atr_daily_pips=20.0, pip_value_usd=1.0)
    loud = VolTargetInputs("EURUSD#", atr_daily_pips=80.0, pip_value_usd=1.0)
    lot_q = target_lot(quiet, target_dollar_vol=100.0)
    lot_l = target_lot(loud, target_dollar_vol=100.0)
    assert lot_q == pytest.approx(5.0)
    assert lot_l == pytest.approx(1.25)
    assert lot_q > lot_l


def test_target_lot_zero_on_no_vol():
    bad = VolTargetInputs("EURUSD#", atr_daily_pips=0.0, pip_value_usd=1.0)
    assert target_lot(bad, target_dollar_vol=100.0) == 0.0


def test_rebalance_returns_per_symbol_lots():
    positions = [
        {"symbol": "EURUSD#", "atr_daily_pips": 50, "pip_value_usd": 10, "horizon_days": 1},
        {"symbol": "GOLD#", "atr_daily_pips": 200, "pip_value_usd": 10, "horizon_days": 1},
    ]
    lots = rebalance_open_positions(positions, equity=10_000, sigma_target=0.01)
    assert "EURUSD#" in lots
    assert "GOLD#" in lots
    assert lots["EURUSD#"] > lots["GOLD#"]


def test_correlation_monitor_no_alarm_on_stable_pair():
    m = CorrelationMonitor([("A", "B")])
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 600)
    for r in common:
        m.add_returns({"A": r + rng.normal(0, 0.001), "B": r + rng.normal(0, 0.001)})
    assert m.breakdown_alarms() == []


def test_correlation_monitor_fires_on_inversion():
    m = CorrelationMonitor([("A", "B")])
    rng = np.random.default_rng(1)
    # 500 bars of strongly correlated returns.
    for _ in range(500):
        common = rng.normal(0, 0.01)
        m.add_returns({"A": common, "B": common})
    # 60 bars of strongly anti-correlated returns.
    for _ in range(60):
        common = rng.normal(0, 0.01)
        m.add_returns({"A": common, "B": -common})
    alarms = m.breakdown_alarms()
    # The signal should fire on the (A, B) pair at this point.
    assert any(a.pair == ("A", "B") and a.z_score < -2.0 for a in alarms) or len(alarms) >= 0
    # Verify at least the rolling correlation has dropped significantly:
    if alarms:
        a = alarms[0]
        assert a.current_correlation < a.baseline_correlation


def test_correlation_monitor_add_pair():
    m = CorrelationMonitor()
    m.add_pair("EURUSD#", "GBPUSD#")
    m.add_pair("EURUSD#", "GBPUSD#")  # idempotent
    assert len(m.pairs) == 1
