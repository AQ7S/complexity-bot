"""Assemble the per-symbol (60, 50) feature tensor for CNN-LSTM input.

50 columns = 45 indicators (engine.features.indicators) + 4 regime one-hots
+ 1 kill-zone flag.

The pipeline expects an OHLCV bar history with at least 200+60 rows so the
indicator warm-up doesn't bleed NaNs into the final 60-row window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators, regime

REGIME_COLUMNS: tuple[str, ...] = (
    "regime_trending_up",
    "regime_trending_down",
    "regime_ranging",
    "regime_high_volatility",
)
EXTRA_COLUMNS: tuple[str, ...] = REGIME_COLUMNS + ("kill_zone_flag",)
FEATURE_COLUMNS: tuple[str, ...] = indicators.FEATURE_COLUMNS + EXTRA_COLUMNS

SEQUENCE_LEN = 60
N_FEATURES = 50

assert len(FEATURE_COLUMNS) == N_FEATURES, (
    f"feature schema mismatch: {len(FEATURE_COLUMNS)} != {N_FEATURES}"
)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (arr - mu) / sd


def build_features(
    bars: pd.DataFrame,
    *,
    kill_zone_flag: bool | pd.Series = False,
    sequence_len: int = SEQUENCE_LEN,
) -> pd.DataFrame:
    """Return a DataFrame of length ``sequence_len`` with the 50 feature columns.

    `bars` must have at least 200 rows of OHLCV history before the window we
    return; we take the last `sequence_len` rows after computing indicators.
    """
    if len(bars) < sequence_len + 50:
        raise ValueError(
            f"bars too short: have {len(bars)} rows, need ≥ {sequence_len + 50}"
        )

    feats = indicators.compute_all(bars)

    snap = regime.classify(bars)
    regime_one_hot = {
        "regime_trending_up":     1.0 if snap.regime == "TRENDING_UP" else 0.0,
        "regime_trending_down":   1.0 if snap.regime == "TRENDING_DOWN" else 0.0,
        "regime_ranging":         1.0 if snap.regime == "RANGING" else 0.0,
        "regime_high_volatility": 1.0 if snap.regime == "HIGH_VOLATILITY" else 0.0,
    }
    for k, v in regime_one_hot.items():
        feats[k] = v

    if isinstance(kill_zone_flag, pd.Series):
        feats["kill_zone_flag"] = kill_zone_flag.reindex(feats.index).fillna(0.0).astype(float)
    else:
        feats["kill_zone_flag"] = float(bool(kill_zone_flag))

    feats = feats[list(FEATURE_COLUMNS)]
    window = feats.tail(sequence_len).copy()
    # Forward/backward fill within the 60-bar window to absorb stragglers.
    window = window.ffill().bfill().fillna(0.0)
    return window


def build_tensor(bars: pd.DataFrame, **kwargs) -> np.ndarray:
    """Build features and return a normalized (60, 50) ndarray."""
    window = build_features(bars, **kwargs)
    arr = window.to_numpy(dtype=np.float64, copy=True)
    arr = _zscore(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape != (SEQUENCE_LEN, N_FEATURES):
        raise RuntimeError(f"unexpected tensor shape {arr.shape}")
    return arr
