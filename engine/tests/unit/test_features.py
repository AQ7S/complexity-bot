"""Phase 4 unit tests — feature pipeline, SMC, regime, correlation.

Plan asserts:
- tensor shape (60, 50)
- no NaN after 200-bar warmup
- SMC returns ≥1 zone on EURUSD M5 sample fixture
- regime ∈ 4 classes
- correlation matrix is 13×13 symmetric, diag=1
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.config.symbols import SYMBOL_NAMES
from engine.features import correlation, feature_pipeline, regime, smc


def synthetic_ohlcv(n: int = 800, seed: int = 42, base: float = 1.07) -> pd.DataFrame:
    """Random-walk OHLCV with M5 cadence and intraday volatility."""
    rng = np.random.default_rng(seed)
    # Trend + noise so SMC has something to find.
    drift = np.linspace(0, 0.01, n)
    steps = rng.normal(0, 0.0005, n) + drift / n
    close = base + np.cumsum(steps)
    open_ = np.roll(close, 1); open_[0] = base
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0003, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0003, n))
    volume = rng.integers(80, 500, n).astype(float)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def m5_bars():
    return synthetic_ohlcv()


def test_tensor_shape_and_no_nan(m5_bars):
    tensor = feature_pipeline.build_tensor(m5_bars, kill_zone_flag=True)
    assert tensor.shape == (60, 50)
    assert not np.isnan(tensor).any(), "NaNs survived the 60-bar window"
    assert np.isfinite(tensor).all()


def test_features_have_no_nan_after_warmup(m5_bars):
    """Plan: no NaN after 200-bar warmup."""
    feats = feature_pipeline.build_features(m5_bars, kill_zone_flag=False, sequence_len=400)
    # Drop the warmup region (first 200 bars) and assert clean.
    tail = feats.iloc[200:]
    assert not tail.isna().any().any(), (
        f"NaNs after warmup: {tail.isna().sum()[tail.isna().sum() > 0].to_dict()}"
    )


def test_smc_returns_at_least_one_zone(m5_bars):
    zones = smc.detect_zones(m5_bars)
    assert "ob" in zones and "fvg" in zones
    # Count rows where any zone column is non-NaN.
    ob_active = zones["ob"].dropna(how="all")
    fvg_active = zones["fvg"].dropna(how="all")
    assert len(ob_active) + len(fvg_active) >= 1, "no SMC zones detected on synthetic M5 fixture"


def test_smc_get_signal_returns_valid_shape(m5_bars):
    h4 = m5_bars.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    m15 = m5_bars.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    sig = smc.get_signal(h4, m15, m5_bars)
    assert sig.signal in ("BUY", "SELL", "HOLD")
    assert sig.zone_type in ("OB", "FVG", "NONE")


def test_regime_in_four_classes(m5_bars):
    snap = regime.classify(m5_bars)
    assert snap.regime in regime.ALL_REGIMES


def test_correlation_matrix_13x13_symmetric_diag_one():
    rng = np.random.default_rng(7)
    closes = {}
    base_walk = np.cumsum(rng.normal(0, 0.0005, 300))
    for i, sym in enumerate(SYMBOL_NAMES):
        # Each symbol is a slightly perturbed version of the base walk so
        # correlations are non-degenerate.
        noise = rng.normal(0, 0.0003, 300) * (i + 1) * 0.1
        prices = 1.0 + base_walk + noise
        closes[sym] = pd.Series(prices, index=pd.date_range("2025-01-01", periods=300, freq="15min"))

    corr = correlation.correlation_matrix(closes, window=200)
    assert corr.shape == (13, 13)
    assert list(corr.columns) == list(SYMBOL_NAMES)
    np.testing.assert_allclose(np.diag(corr.values), np.ones(13), atol=1e-9)
    # Symmetry within float tolerance.
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)
    assert ((corr.values >= -1.0) & (corr.values <= 1.0 + 1e-9)).all()
