from engine.data.spread_monitor import SpreadMonitor
from engine.strategy.volatility_regime import VolatilityRegimeTracker


def test_spread_monitor_warmup_accepts() -> None:
    sm = SpreadMonitor()
    for _ in range(3):
        sm.update("EURUSD#", 0.00010)
    v = sm.evaluate("EURUSD#", 0.00100)
    assert v.accepted is True
    assert v.reason == "WARMUP"


def test_spread_monitor_blocks_outlier() -> None:
    sm = SpreadMonitor()
    for _ in range(20):
        sm.update("EURUSD#", 0.00010)
    v = sm.evaluate("EURUSD#", 0.00050)
    assert v.accepted is False
    assert v.multiplier >= 2.5
    assert v.reason.startswith("SPREAD_BLOCK")


def test_spread_monitor_passes_normal() -> None:
    sm = SpreadMonitor()
    for _ in range(20):
        sm.update("EURUSD#", 0.00010)
    v = sm.evaluate("EURUSD#", 0.00020)
    assert v.accepted is True
    assert 1.5 < v.multiplier < 2.5


def test_volatility_warmup_normal() -> None:
    vt = VolatilityRegimeTracker()
    for _ in range(5):
        vt.update("EURUSD#", 0.0005)
    v = vt.classify("EURUSD#", 0.0005)
    assert v.regime == "NORMAL"
    assert v.reason == "WARMUP"


def test_volatility_extreme_blocks() -> None:
    vt = VolatilityRegimeTracker()
    for i in range(50):
        vt.update("EURUSD#", 0.0001 + i * 1e-6)
    v = vt.classify("EURUSD#", 0.01)
    assert v.regime == "EXTREME"
    assert v.blocked is True
    assert v.risk_multiplier == 0.0


def test_volatility_high_reduces_risk() -> None:
    vt = VolatilityRegimeTracker()
    for i in range(50):
        vt.update("EURUSD#", 0.0001 + i * 1e-6)
    v = vt.classify("EURUSD#", 0.000145)
    assert v.regime == "HIGH"
    assert v.blocked is False
    assert v.risk_multiplier == 0.5
    assert v.sl_multiplier == 1.5


def test_volatility_low_tightens_sl() -> None:
    vt = VolatilityRegimeTracker()
    for i in range(50):
        vt.update("EURUSD#", 0.001 + i * 1e-5)
    v = vt.classify("EURUSD#", 0.00098)
    assert v.regime == "LOW"
    assert v.sl_multiplier == 0.8
    assert v.risk_multiplier == 0.75
