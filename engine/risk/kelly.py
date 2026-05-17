"""Fractional Kelly per-strategy risk sizing.

The fixed 2% per-trade rule is a safe default but it ignores how much
*edge* a strategy actually has. Kelly sizing connects size to edge:
strategies with proven positive expectancy get more risk, ones that
break even get less.

Full Kelly:
    f* = (p × (b + 1) − 1) / b

where p = win probability, b = avg win / avg loss ratio. Full Kelly is
mathematically optimal but real edges are estimated, not known — applying
full Kelly to a noisy estimate is catastrophic. The academic safety
standard is FRACTIONAL Kelly, typically 1/4 Kelly, which:

  * keeps growth rate at ~75% of full-Kelly optimum
  * cuts variance by ~94%
  * is much more robust to overestimated edge

Output is clipped to [0.005, 0.02] (0.5%–2%) so a wildly positive estimate
cannot exceed our hard per-trade risk cap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KELLY_FRACTION = 0.25
RISK_FLOOR = 0.005
RISK_CAP = 0.02
MIN_TRADES_FOR_KELLY = 20


@dataclass(frozen=True)
class KellyEstimate:
    full_kelly: float
    fractional_kelly: float
    win_rate: float
    avg_win: float
    avg_loss: float
    n_samples: int
    used_floor: bool = False
    used_cap: bool = False


def compute_kelly_fraction(
    wins: int,
    losses: int,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Full Kelly fraction from (wins, losses, avg_win, |avg_loss|).

    Returns 0 when avg_loss is non-positive or sample is degenerate.
    """
    n = wins + losses
    if n <= 0 or avg_loss <= 0 or avg_win <= 0:
        return 0.0
    p = wins / n
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    return (p * (b + 1.0) - 1.0) / b


def fractional_kelly(
    p: float,
    b: float,
    *,
    fraction: float = KELLY_FRACTION,
    floor: float = RISK_FLOOR,
    cap: float = RISK_CAP,
) -> float:
    """Clipped fractional Kelly: max(floor, min(cap, fraction × full Kelly))."""
    if b <= 0:
        return floor
    full = (p * (b + 1.0) - 1.0) / b
    frac = fraction * full
    if frac < floor:
        return floor
    if frac > cap:
        return cap
    return frac


def kelly_from_pnl(
    pnls: list[float] | np.ndarray,
    *,
    fraction: float = KELLY_FRACTION,
    floor: float = RISK_FLOOR,
    cap: float = RISK_CAP,
) -> KellyEstimate:
    """Estimate Kelly fraction from a list of trade PnL values.

    Insufficient samples (<MIN_TRADES_FOR_KELLY) → floor. Pure-winner or
    pure-loser samples → floor (avoid divide-by-zero edge case).
    """
    arr = np.asarray(pnls, dtype=np.float64)
    if arr.size == 0:
        return KellyEstimate(0.0, floor, 0.0, 0.0, 0.0, 0, used_floor=True)
    wins_arr = arr[arr > 0]
    losses_arr = arr[arr < 0]
    n = arr.size
    if n < MIN_TRADES_FOR_KELLY or wins_arr.size == 0 or losses_arr.size == 0:
        return KellyEstimate(
            full_kelly=0.0, fractional_kelly=floor,
            win_rate=float(wins_arr.size) / max(n, 1),
            avg_win=float(wins_arr.mean()) if wins_arr.size else 0.0,
            avg_loss=float(-losses_arr.mean()) if losses_arr.size else 0.0,
            n_samples=n, used_floor=True,
        )
    p = wins_arr.size / n
    avg_win = float(wins_arr.mean())
    avg_loss = float(-losses_arr.mean())
    b = avg_win / avg_loss
    full = (p * (b + 1.0) - 1.0) / b
    frac = fraction * full
    used_floor = used_cap = False
    if frac < floor:
        frac = floor
        used_floor = True
    elif frac > cap:
        frac = cap
        used_cap = True
    return KellyEstimate(
        full_kelly=full, fractional_kelly=frac,
        win_rate=p, avg_win=avg_win, avg_loss=avg_loss,
        n_samples=n, used_floor=used_floor, used_cap=used_cap,
    )
