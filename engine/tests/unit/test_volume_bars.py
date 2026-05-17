"""Tests for volume bars (Tier 2.3)."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data.volume_bars import (
    VolumeBarBuilder,
    calibrate_target_volume,
    from_tick_history,
)


def test_builder_emits_bar_at_threshold():
    b = VolumeBarBuilder(target_volume=10.0)
    ts = pd.Timestamp("2024-01-01")
    completed = None
    for i in range(10):
        c = b.add_tick(ts + pd.Timedelta(seconds=i), price=1.0 + i * 0.01, volume=1.0)
        if c is not None:
            completed = c
    assert completed is not None
    assert completed.volume == pytest.approx(10.0)
    assert completed.open == pytest.approx(1.0)
    assert completed.tick_count == 10


def test_partial_fill_spills_into_next_bar():
    b = VolumeBarBuilder(target_volume=10.0)
    ts = pd.Timestamp("2024-01-01")
    # Single tick of 25 vol → fills 2 full bars + leaves 5 in next.
    c1 = b.add_tick(ts, price=1.0, volume=25.0)
    assert c1 is not None
    # The builder should hold one more completed bar internally.
    extras = b.pop_completed()
    assert len(extras) >= 1
    # And there should still be partial volume waiting.
    assert b._volume == pytest.approx(5.0)


def test_high_low_track_across_ticks():
    b = VolumeBarBuilder(target_volume=10.0)
    ts = pd.Timestamp("2024-01-01")
    prices = [1.0, 1.05, 0.95, 1.10, 1.02, 0.98, 1.03, 1.04, 1.01, 1.00]
    completed = None
    for i, p in enumerate(prices):
        c = b.add_tick(ts + pd.Timedelta(seconds=i), price=p, volume=1.0)
        if c is not None:
            completed = c
    assert completed is not None
    assert completed.high == 1.10
    assert completed.low == 0.95


def test_from_tick_history_aggregates_in_one_call():
    ticks = []
    for i in range(50):
        ticks.append({
            "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            "bid": 1.0, "ask": 1.001, "volume": 2.0,
        })
    bars = from_tick_history(ticks, target_volume=10.0)
    assert len(bars) == 10
    for bar in bars:
        assert bar.volume == pytest.approx(10.0)


def test_builder_rejects_invalid_target():
    with pytest.raises(ValueError):
        VolumeBarBuilder(target_volume=0)


def test_calibrate_target_volume_default():
    idx = pd.date_range("2024-01-01", periods=288 * 2, freq="5min")
    df = pd.DataFrame({"volume": [10.0] * len(idx)}, index=idx)
    # 2 days, 288 bars/day × 10 vol = 2880 vol/day, /288 = 10
    target = calibrate_target_volume(df, target_bars_per_day=288)
    assert target == pytest.approx(10.0)


def test_calibrate_target_volume_empty_returns_one():
    assert calibrate_target_volume(pd.DataFrame()) == 1.0
