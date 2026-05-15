"""Build (60, 50) sliding-window training tensors from a long OHLCV history.

Optimised so that indicator computation happens *once* over the full history,
then we slice with a stride-trick for the sequence dimension. This cuts a
26k-bar dataset from minutes (per-window indicator recompute) to seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features import indicators
from engine.features.feature_pipeline import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LEN


def _vectorized_regime(feats: pd.DataFrame, *, percentile_window: int = 200) -> pd.DataFrame:
    """Per-bar regime one-hot using the bar's own ADX/ATR/EMA values."""
    adx = feats["adx_14"]
    atr_pct = feats["atr_pct"]
    ema21 = feats["ema_21"]
    ema50 = feats["ema_50"]

    # Rolling ATR percentile rank over `percentile_window` bars.
    atr_pctl = atr_pct.rolling(percentile_window, min_periods=20).rank(pct=True)
    is_high_vol = (atr_pctl >= 0.85) & (adx < 25.0)
    is_trend = adx >= 25.0
    is_up = is_trend & (ema21 > ema50)
    is_down = is_trend & (ema21 <= ema50)
    is_range = ~(is_high_vol | is_trend)

    return pd.DataFrame({
        "regime_trending_up":     is_up.astype(float),
        "regime_trending_down":   is_down.astype(float),
        "regime_ranging":         is_range.astype(float),
        "regime_high_volatility": is_high_vol.astype(float),
    }, index=feats.index).fillna(0.0)


def build_feature_frame(bars: pd.DataFrame, *, kill_zone_flag: pd.Series | None = None) -> pd.DataFrame:
    """Compute the full 50-column feature DataFrame across all of `bars`."""
    feats = indicators.compute_all(bars)
    regime_oh = _vectorized_regime(feats)
    feats = pd.concat([feats, regime_oh], axis=1)
    if kill_zone_flag is None:
        feats["kill_zone_flag"] = 0.0
    else:
        feats["kill_zone_flag"] = kill_zone_flag.reindex(feats.index).fillna(0.0).astype(float)
    return feats[list(FEATURE_COLUMNS)]


def make_labels(close: pd.Series, *, threshold_bps: float = 1.0) -> np.ndarray:
    """3-class next-bar direction.

    BUY=0 if next return > +threshold, SELL=1 if < -threshold, else HOLD=2.
    `threshold_bps` is in basis points (1 bp = 0.0001 = 1pip on EURUSD).
    """
    next_ret = close.shift(-1) / close - 1.0
    th = threshold_bps / 10_000.0
    labels = np.full(len(close), 2, dtype=np.int64)  # default HOLD
    labels[next_ret.values > th] = 0   # BUY
    labels[next_ret.values < -th] = 1  # SELL
    labels[-1] = -1  # last bar has no next ⇒ mark and drop
    return labels


def build_windows(
    bars: pd.DataFrame,
    *,
    sequence_len: int = SEQUENCE_LEN,
    warmup: int = 200,
    label_threshold_bps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X is (N, 60, 50) float32 and y is (N,) int64."""
    feats = build_feature_frame(bars)
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    feat_arr = feats.to_numpy(dtype=np.float32, copy=False)  # (T, 50)
    labels = make_labels(bars["close"], threshold_bps=label_threshold_bps)

    # Per-column z-score using stats from the warmup region only (no leakage).
    mu = feat_arr[warmup : warmup + sequence_len * 200].mean(axis=0, keepdims=True)
    sd = feat_arr[warmup : warmup + sequence_len * 200].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    feat_norm = (feat_arr - mu) / sd
    feat_norm = np.clip(feat_norm, -10.0, 10.0).astype(np.float32, copy=False)

    # Window indices: a sample at row `i` uses rows [i - sequence_len + 1, i]
    # and labels position `i`. Skip the warmup and the last row (no label).
    start = max(warmup, sequence_len - 1)
    end = len(feat_norm) - 1
    n = end - start
    if n <= 0:
        raise ValueError(f"not enough bars: T={len(feat_norm)} warmup={warmup} seq={sequence_len}")

    X = np.empty((n, sequence_len, N_FEATURES), dtype=np.float32)
    for k, i in enumerate(range(start, end)):
        X[k] = feat_norm[i - sequence_len + 1 : i + 1]
    y = labels[start:end].astype(np.int64)

    # Filter out any sentinel labels (-1).
    mask = y >= 0
    return X[mask], y[mask]
