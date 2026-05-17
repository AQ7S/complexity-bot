"""Tests for tick arrival rate + trade intensity (Tier 2.4)
and Avellaneda-Stoikov mean reversion (Tier 2.5)."""
from __future__ import annotations

import pytest

from engine.features.order_flow import tick_arrival_rate, trade_intensity
from engine.strategy.mean_reversion import (
    avellaneda_stoikov_signal,
    compute_inventory_skew,
    target_price,
)


def _tick(ts_ms: int, bid: float = 1.0, ask: float = 1.001, last: float | None = None) -> dict:
    d = {"ts": ts_ms, "bid": bid, "ask": ask, "volume": 1.0}
    if last is not None:
        d["last"] = last
    return d


def test_arrival_rate_counts_per_second():
    # 60 ticks evenly spaced over 60 seconds → 1 tick/s.
    ticks = [_tick(i * 1000) for i in range(61)]
    rate = tick_arrival_rate(ticks, window_s=60, now_ts=60.0)
    assert rate == pytest.approx(61 / 60, abs=0.02)


def test_arrival_rate_zero_on_empty():
    assert tick_arrival_rate([], window_s=60) == 0.0


def test_trade_intensity_zero_on_flat_price():
    # All ticks same price → no price-changers → intensity = 0.
    ticks = [_tick(i * 1000, last=1.0) for i in range(50)]
    intensity = trade_intensity(ticks, half_life_s=30, now_ts=50.0)
    assert intensity == 0.0


def test_trade_intensity_positive_when_price_changes():
    ticks = [_tick(i * 1000, last=1.0 + i * 0.001) for i in range(50)]
    intensity = trade_intensity(ticks, half_life_s=30, now_ts=50.0)
    assert intensity > 0


def test_avellaneda_stoikov_hold_when_within_threshold():
    prices = [1.0] * 60
    sig = avellaneda_stoikov_signal(prices, lookback=60, z_threshold=2.5)
    assert sig.direction == "HOLD"


def test_avellaneda_stoikov_sell_on_overstretched_up():
    # 59 prices around 1.0, then one outlier at 1.10 → z >> 2.5
    prices = [1.0 + ((-1) ** i) * 0.001 for i in range(59)] + [1.10]
    sig = avellaneda_stoikov_signal(prices, lookback=60, z_threshold=2.5)
    assert sig.direction == "SELL"
    assert sig.z_score > 2.5


def test_avellaneda_stoikov_buy_on_overstretched_down():
    prices = [1.0 + ((-1) ** i) * 0.001 for i in range(59)] + [0.90]
    sig = avellaneda_stoikov_signal(prices, lookback=60, z_threshold=2.5)
    assert sig.direction == "BUY"
    assert sig.z_score < -2.5


def test_inventory_skew_signed_sum():
    positions = [
        {"direction": "BUY",  "lot": 0.3},
        {"direction": "SELL", "lot": 0.1},
        {"direction": "BUY",  "lot": 0.2},
    ]
    assert compute_inventory_skew(positions) == pytest.approx(0.4)


def test_target_price_for_short_above_mean():
    sig = avellaneda_stoikov_signal([1.0] * 30 + [1.10], lookback=30, z_threshold=2.0)
    tp = target_price(sig, z_target=0.5)
    assert tp is not None
    # Target should be above the rolling mean for a SELL (mean reversion down).
    assert tp > sig.rolling_mean


def test_target_price_none_for_hold():
    sig = avellaneda_stoikov_signal([1.0] * 30, lookback=30, z_threshold=2.5)
    assert target_price(sig) is None
