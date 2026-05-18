"""Tests for cointegration + pairs trading primitives (Tier 8.3)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.strategy.cointegration import (
    augmented_dickey_fuller,
    engle_granger_cointegration,
    pairs_trade_signal,
)


def test_adf_rejects_unit_root_on_stationary_series():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    res = augmented_dickey_fuller(x, lags=1)
    assert res.is_stationary


def test_adf_accepts_unit_root_on_random_walk():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0.0, 1.0, 500))
    res = augmented_dickey_fuller(rw, lags=1)
    assert not res.is_stationary


def test_adf_short_series_safe():
    res = augmented_dickey_fuller(np.array([1.0, 2.0, 3.0]))
    assert not res.is_stationary


def test_engle_granger_cointegrated_pair():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0.0, 1.0, 600))
    noise = rng.normal(0.0, 0.5, 600)
    y = 1.5 * x + 10.0 + noise
    res = engle_granger_cointegration(y, x)
    assert res.is_cointegrated
    assert res.hedge_ratio == pytest.approx(1.5, abs=0.2)
    assert res.intercept == pytest.approx(10.0, abs=2.0)


def test_engle_granger_non_stationary_residuals_not_cointegrated():
    # Construct a y that is x + a SEPARATE random walk — residuals are themselves
    # a random walk → not cointegrated.
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(0.0, 1.0, 600))
    rw = np.cumsum(rng.normal(0.0, 1.0, 600))
    y = 1.5 * x + rw
    res = engle_granger_cointegration(y, x)
    assert not res.is_cointegrated


def test_pairs_signal_flat_when_not_cointegrated():
    from engine.strategy.cointegration import CointegrationResult
    coint = CointegrationResult(
        is_cointegrated=False, hedge_ratio=0.0, intercept=0.0,
        spread_mean=0.0, spread_std=0.0, adf_stat=0.0,
        adf_critical_5pct=-1.95, n=0,
    )
    sig = pairs_trade_signal(np.array([1.0]), np.array([1.0]), coint)
    assert sig.side == "FLAT"


def test_pairs_signal_short_when_overstretched_up():
    from engine.strategy.cointegration import CointegrationResult
    coint = CointegrationResult(
        is_cointegrated=True, hedge_ratio=1.0, intercept=0.0,
        spread_mean=0.0, spread_std=1.0, adf_stat=-3.0,
        adf_critical_5pct=-1.95, n=500,
    )
    sig = pairs_trade_signal(np.array([2.5]), np.array([0.0]), coint, z_entry=2.0)
    assert sig.side == "SHORT_Y_LONG_X"
    assert sig.z_score > 2.0


def test_pairs_signal_long_when_overstretched_down():
    from engine.strategy.cointegration import CointegrationResult
    coint = CointegrationResult(
        is_cointegrated=True, hedge_ratio=1.0, intercept=0.0,
        spread_mean=0.0, spread_std=1.0, adf_stat=-3.0,
        adf_critical_5pct=-1.95, n=500,
    )
    sig = pairs_trade_signal(np.array([-2.5]), np.array([0.0]), coint, z_entry=2.0)
    assert sig.side == "LONG_Y_SHORT_X"


def test_pairs_signal_stop_on_extreme_z():
    from engine.strategy.cointegration import CointegrationResult
    coint = CointegrationResult(
        is_cointegrated=True, hedge_ratio=1.0, intercept=0.0,
        spread_mean=0.0, spread_std=1.0, adf_stat=-3.0,
        adf_critical_5pct=-1.95, n=500,
    )
    sig = pairs_trade_signal(np.array([4.0]), np.array([0.0]), coint,
                              z_entry=2.0, z_stop=3.5)
    assert sig.side == "FLAT"
    assert "stop" in sig.notes
