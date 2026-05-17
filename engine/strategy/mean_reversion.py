"""Avellaneda–Stoikov style mean-reversion signal primitives.

The original Avellaneda–Stoikov (2008) framework is a market-making model
for optimal bid/ask quote placement; we adapt its core insight — that
short-horizon mid-price deviations from a rolling reference are
mean-reverting — into a directional signal generator for the scalping
strategy.

Algorithm:
    z = (price − rolling_mean(short)) / rolling_std(short)
    z > +z_threshold  → price overstretched up   → SELL (mean reversion)
    z < −z_threshold  → price overstretched down → BUY  (mean reversion)
    |z| ≤ z_threshold → HOLD

Default thresholds chosen for M1-equivalent volume bars; SL = 0.5× ATR,
target = +0.5σ. Holding window ≤30s by design — this is *scalping*.

The inventory-skew helper applies the Avellaneda–Stoikov optimal quote
adjustment when the strategy already holds inventory: positive inventory
biases the next signal toward selling (closing) and vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Vote = Literal["BUY", "SELL", "HOLD"]

DEFAULT_Z_THRESHOLD = 2.5
DEFAULT_LOOKBACK = 60
INVENTORY_PENALTY = 0.5


@dataclass(frozen=True)
class MeanReversionSignal:
    direction: Vote
    z_score: float
    rolling_mean: float
    rolling_std: float
    inventory_adjusted_z: float


def avellaneda_stoikov_signal(
    prices: pd.Series | np.ndarray,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    inventory: float = 0.0,
) -> MeanReversionSignal:
    """Return a directional vote based on the latest z-score.

    `inventory` (signed lots / units currently held) biases the z-score:
    long inventory increases the apparent over-extension on the up side,
    shifting the signal toward SELL even at a smaller absolute z.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size < 2:
        return MeanReversionSignal("HOLD", 0.0, float(arr[-1]) if arr.size else 0.0, 0.0, 0.0)
    window = arr[-lookback:] if arr.size > lookback else arr
    mu = float(np.mean(window))
    sd = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
    p = float(arr[-1])
    if sd < 1e-12:
        return MeanReversionSignal("HOLD", 0.0, mu, sd, 0.0)
    z = (p - mu) / sd
    z_adj = z + INVENTORY_PENALTY * float(inventory)
    if z_adj > z_threshold:
        direction: Vote = "SELL"
    elif z_adj < -z_threshold:
        direction = "BUY"
    else:
        direction = "HOLD"
    return MeanReversionSignal(
        direction=direction,
        z_score=z,
        rolling_mean=mu,
        rolling_std=sd,
        inventory_adjusted_z=z_adj,
    )


def compute_inventory_skew(open_positions: list[dict]) -> float:
    """Sum of signed lot sizes across open positions on a single symbol.

    BUY positions contribute +lot, SELL positions contribute -lot. The
    returned value feeds `inventory` in `avellaneda_stoikov_signal()`.
    """
    skew = 0.0
    for pos in open_positions:
        lot = float(pos.get("lot", pos.get("volume", 0.0)) or 0.0)
        side = str(pos.get("direction", pos.get("side", ""))).upper()
        if side == "BUY":
            skew += lot
        elif side == "SELL":
            skew -= lot
    return skew


def target_price(
    signal: MeanReversionSignal,
    *,
    direction: Vote | None = None,
    z_target: float = 0.5,
) -> float | None:
    """Compute a take-profit price at `z_target` standard deviations
    (toward the mean) from the current price.

    Returns None if the signal direction is HOLD or rolling_std is zero.
    """
    d = direction or signal.direction
    if d == "HOLD" or signal.rolling_std <= 0:
        return None
    if d == "SELL":
        return signal.rolling_mean + z_target * signal.rolling_std
    return signal.rolling_mean - z_target * signal.rolling_std
