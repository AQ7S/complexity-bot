"""Tests for triple-barrier labeling (Tier 1.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.learning.triple_barrier import (
    apply_triple_barrier,
    build_triple_barrier_dataset,
    compute_volatility,
)


def _flat_then_up(n: int = 100, up_after: int = 60) -> pd.Series:
    """Flat 1.0 until `up_after`, then linear ramp to 1.10."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    arr = np.full(n, 1.0)
    if up_after < n:
        arr[up_after:] = np.linspace(1.0, 1.10, n - up_after)
    return pd.Series(arr, index=idx, name="close")


def test_volatility_is_nonneg_and_zero_on_flat():
    s = pd.Series([1.0] * 100, index=pd.date_range("2024-01-01", periods=100, freq="5min"))
    vol = compute_volatility(s, span=20)
    assert (vol >= 0).all()
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_vertical_barrier_timeout_returns_hold():
    s = _flat_then_up(n=200, up_after=300)  # ramp never happens within data
    vol = pd.Series(np.full(len(s), 0.01), index=s.index)
    labels = apply_triple_barrier(
        s, np.array([10, 20, 30]), vol,
        pt_mult=2.0, sl_mult=1.0, max_h=20,
    )
    assert (labels == 2).all()


def test_upper_barrier_first_returns_buy():
    idx = pd.date_range("2024-01-01", periods=50, freq="5min")
    prices = np.full(50, 1.0)
    prices[5] = 1.05  # spike up shortly after t0=0
    s = pd.Series(prices, index=idx, name="close")
    vol = pd.Series(np.full(50, 0.02), index=idx)
    labels = apply_triple_barrier(
        s, np.array([0]), vol,
        pt_mult=1.0, sl_mult=1.0, max_h=10,
    )
    assert labels[0] == 0


def test_lower_barrier_first_returns_sell():
    idx = pd.date_range("2024-01-01", periods=50, freq="5min")
    prices = np.full(50, 1.0)
    prices[5] = 0.95
    s = pd.Series(prices, index=idx, name="close")
    vol = pd.Series(np.full(50, 0.02), index=idx)
    labels = apply_triple_barrier(
        s, np.array([0]), vol,
        pt_mult=1.0, sl_mult=1.0, max_h=10,
    )
    assert labels[0] == 1


def test_asymmetric_barriers_distinguish_outcomes():
    idx = pd.date_range("2024-01-01", periods=50, freq="5min")
    prices = np.full(50, 1.0)
    # 1.5% move up at index 3
    prices[3] = 1.015
    s = pd.Series(prices, index=idx, name="close")
    vol = pd.Series(np.full(50, 0.01), index=idx)
    # pt=2.0 → upper at +2% (1.02), sl=1.0 → lower at -1% (0.99).
    # 1.5% move hits neither → HOLD.
    labels = apply_triple_barrier(
        s, np.array([0]), vol, pt_mult=2.0, sl_mult=1.0, max_h=10,
    )
    assert labels[0] == 2
    # Tighten pt to 1.0 → upper at +1% (1.01). 1.5% move hits upper → BUY.
    labels2 = apply_triple_barrier(
        s, np.array([0]), vol, pt_mult=1.0, sl_mult=2.0, max_h=10,
    )
    assert labels2[0] == 0


def test_zero_volatility_skipped_to_hold():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min")
    s = pd.Series(np.full(20, 1.0), index=idx, name="close")
    vol = pd.Series(np.zeros(20), index=idx)
    labels = apply_triple_barrier(s, np.array([0, 5]), vol)
    assert (labels == 2).all()


def test_out_of_range_t0_safely_handled():
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    s = pd.Series(np.full(10, 1.0), index=idx, name="close")
    vol = pd.Series(np.full(10, 0.01), index=idx)
    labels = apply_triple_barrier(s, np.array([-1, 100, 9]), vol)
    assert labels.shape == (3,)
    assert (labels == 2).all()


def test_build_dataset_returns_aligned_shapes():
    idx = pd.date_range("2024-01-01", periods=400, freq="5min")
    rng = np.random.default_rng(0)
    prices = 1.0 + np.cumsum(rng.normal(0, 0.001, 400))
    bars = pd.DataFrame({"close": prices}, index=idx)
    t0, labels, times = build_triple_barrier_dataset(
        bars, pt_mult=2.0, sl_mult=1.0, max_h=20, vol_span=20, skip_first=50,
    )
    assert len(t0) == len(labels) == len(times)
    assert len(t0) > 0
    assert labels.min() >= 0 and labels.max() <= 2
