"""Tests for the 6 strategy classes (Tier 3.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy.base import StrategyContext
from engine.strategy.strategies import (
    BreakoutStrategy,
    CarryStrategy,
    DayTradingStrategy,
    MeanReversionStrategy,
    ScalpingStrategy,
    SwingStrategy,
    all_strategies,
)


def _bars_oscillating(n: int = 100, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 1.0 + 0.001 * np.sin(np.linspace(0, 8 * np.pi, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.0005, "low": close - 0.0005,
        "close": close, "volume": [1.0] * n,
    }, index=idx)


def _bars_breakout(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = np.concatenate([np.linspace(1.0, 1.001, n - 1), [1.010]])
    high = close + 0.0001
    low = close - 0.0001
    high[-1] = 1.010 + 0.0005
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": [1.0] * n}, index=idx)


def test_all_strategies_listed():
    strategies = all_strategies()
    assert len(strategies) == 7
    names = {s.name for s in strategies}
    assert names == {
        "scalping", "day_trading", "swing",
        "mean_reversion", "breakout", "carry",
        "pairs_trading",
    }


def test_scalping_rejects_non_whitelisted_symbol():
    s = ScalpingStrategy()
    ctx = StrategyContext(
        symbol="EURJPY#", timeframe="M1",
        bars=_bars_oscillating(n=80),
    )
    assert s.detect(ctx) is None


def test_scalping_emits_on_overstretched_z():
    s = ScalpingStrategy(z_threshold=1.5)
    bars = _bars_oscillating(n=80)
    # Push the last close hard up.
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("close")] = 1.05
    bars.iloc[-1, bars.columns.get_loc("high")] = 1.06
    ctx = StrategyContext(symbol="EURUSD#", timeframe="M1", bars=bars)
    sig = s.detect(ctx)
    assert sig is not None
    assert sig.direction in {"BUY", "SELL"}


def test_day_trading_returns_none_in_isolation():
    s = DayTradingStrategy()
    ctx = StrategyContext(symbol="EURUSD#", timeframe="M5")
    assert s.detect(ctx) is None


def test_swing_requires_bias():
    s = SwingStrategy()
    bars = _bars_oscillating(n=30, freq="60min")
    ctx = StrategyContext(symbol="EURUSD#", timeframe="H1", bars=bars, h4_bias="RANGING")
    assert s.detect(ctx) is None


def test_swing_emits_on_uptrend_bias():
    s = SwingStrategy()
    bars = _bars_oscillating(n=30, freq="60min")
    ctx = StrategyContext(symbol="EURUSD#", timeframe="H1", bars=bars, h4_bias="TRENDING_UP")
    sig = s.detect(ctx)
    assert sig is not None
    assert sig.direction == "BUY"


def test_mean_reversion_emits_on_high_z():
    s = MeanReversionStrategy(z_threshold=1.0)
    bars = _bars_oscillating(n=80, freq="15min")
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("close")] = 1.05
    ctx = StrategyContext(symbol="EURUSD#", timeframe="M15", bars=bars)
    sig = s.detect(ctx)
    assert sig is not None


def test_breakout_fires_on_range_break():
    s = BreakoutStrategy()
    bars = _bars_breakout()
    ctx = StrategyContext(symbol="EURUSD#", timeframe="M15", bars=bars)
    sig = s.detect(ctx)
    assert sig is not None
    assert sig.direction == "BUY"


def test_carry_only_on_whitelisted_pair():
    s = CarryStrategy()
    bars = pd.DataFrame({
        "open": [1.0] * 20, "high": [1.01] * 20, "low": [0.99] * 20,
        "close": np.linspace(1.0, 1.01, 20), "volume": [1.0] * 20,
    }, index=pd.date_range("2024-01-01", periods=20, freq="D"))
    ctx = StrategyContext(symbol="AUDUSD#", timeframe="D1", bars=bars)
    sig = s.detect(ctx)
    if sig is not None:
        assert sig.direction == "BUY"
    ctx2 = StrategyContext(symbol="UNKNOWN#", timeframe="D1", bars=bars)
    assert s.detect(ctx2) is None
