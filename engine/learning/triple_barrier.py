"""Triple-barrier labeling (López de Prado, AFML ch. 3).

Each candidate entry bar gets labeled by which of three barriers is touched
first: upper (profit-target), lower (stop-loss), or vertical (timeout).

Labels:
    0 = BUY  (upper barrier hit first — long would have won)
    1 = SELL (lower barrier hit first — long would have lost; short would win)
    2 = HOLD (timeout — vertical barrier hit first; no clear directional outcome)

Compared to next-bar direction labels, this classifies samples by their
*actual trade outcome* over a realistic holding window. It removes label
noise from bars that have nothing to do with whether a trade would be
profitable, and typically lifts effective model accuracy by ~10pp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierParams:
    pt_mult: float = 2.0
    sl_mult: float = 1.0
    max_h: int = 48
    vol_span: int = 50
    min_ret: float = 1e-6


def compute_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """Rolling EMA of absolute returns — the bar-by-bar volatility scale.

    `span` of 50 gives a half-life of ~17 bars, which is the standard AFML
    choice for M5 trading. Returns a Series indexed identically to `close`.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    rets = close.pct_change().abs()
    vol = rets.ewm(span=span, adjust=False).mean()
    return vol.fillna(0.0)


def apply_triple_barrier(
    prices: pd.Series,
    t0_indices: np.ndarray | list[int],
    vol: pd.Series,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_h: int = 48,
    min_ret: float = 1e-6,
) -> np.ndarray:
    """Return label array for each entry index in `t0_indices`.

    For each `t0`:
        upper barrier = price[t0] * (1 + pt_mult * vol[t0])
        lower barrier = price[t0] * (1 - sl_mult * vol[t0])
        vertical barrier = t0 + max_h
    The first barrier touched in [t0+1, min(t0+max_h, T-1)] decides the label.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    arr = prices.to_numpy(dtype=np.float64, copy=False)
    vol_arr = vol.to_numpy(dtype=np.float64, copy=False)
    n = len(arr)
    t0 = np.asarray(t0_indices, dtype=np.int64)
    labels = np.full(len(t0), 2, dtype=np.int64)

    for k, i in enumerate(t0):
        if i < 0 or i >= n - 1:
            continue
        v = vol_arr[i]
        if not np.isfinite(v) or v < min_ret:
            continue
        p0 = arr[i]
        upper = p0 * (1.0 + pt_mult * v)
        lower = p0 * (1.0 - sl_mult * v)
        end = min(i + max_h, n - 1)
        # Walk forward; first touch wins.
        hit = 2
        for j in range(i + 1, end + 1):
            pj = arr[j]
            if pj >= upper:
                hit = 0
                break
            if pj <= lower:
                hit = 1
                break
        labels[k] = hit
    return labels


def build_triple_barrier_dataset(
    bars: pd.DataFrame,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_h: int = 48,
    vol_span: int = 50,
    skip_first: int = 200,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (t0_indices, labels, label_times) over `bars`.

    `label_times` is the time at which each label was determined (max(t0+max_h,
    first-touch-bar)). It is used by purged CV to enforce no-overlap with the
    test fold's time range.

    `skip_first` skips the warmup region used by indicators / volatility.
    """
    if "close" not in bars.columns:
        raise ValueError("bars must contain a 'close' column")
    close = bars["close"]
    vol = compute_volatility(close, span=vol_span)
    # Valid entry indices: after warmup and before max_h bars from the end.
    start = max(skip_first, vol_span)
    end = len(close) - max_h - 1
    if end <= start:
        raise ValueError(f"bars too short: T={len(close)} skip_first={skip_first} max_h={max_h}")
    t0 = np.arange(start, end, dtype=np.int64)
    labels = apply_triple_barrier(
        close, t0, vol,
        pt_mult=pt_mult, sl_mult=sl_mult, max_h=max_h,
    )
    # Label time = end of the trade horizon (conservative; overlapping windows
    # then must be purged from training when test contains any of these times).
    label_idx = np.minimum(t0 + max_h, len(close) - 1)
    label_times = close.index[label_idx]
    return t0, labels, pd.DatetimeIndex(label_times)
