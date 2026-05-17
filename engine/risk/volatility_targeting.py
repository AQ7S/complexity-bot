"""Dollar volatility targeting.

Equal-risk-percent sizing (2% per trade) ignores that a 2% risk on a
quiet 0.5% ATR symbol contributes far less dollar variance than 2% on a
turbulent 3% ATR symbol. Vol targeting normalises by *dollar volatility
contribution* so each open position contributes roughly the same daily
dollar vol to the portfolio.

Target:
    dollar_vol_per_position = σ_target × equity / N_active

For position p on symbol s:
    position_dollar_vol = lot × pip_value × atr_daily_pips × √h

where h is the intended holding period in days. We invert for lot:
    lot = target_dollar_vol / (pip_value × atr_daily_pips × √h)
"""
from __future__ import annotations

from dataclasses import dataclass

import math


VOL_TARGET_DAILY_DEFAULT = 0.01    # 1% of equity per position per day


@dataclass(frozen=True)
class VolTargetInputs:
    symbol: str
    atr_daily_pips: float
    pip_value_usd: float
    horizon_days: float = 1.0


def target_lot(
    inputs: VolTargetInputs,
    *,
    target_dollar_vol: float,
) -> float:
    """Solve for the lot size that contributes `target_dollar_vol`.

    Returns 0 when ATR is non-positive (no measurable risk to size against).
    """
    if inputs.atr_daily_pips <= 0 or inputs.pip_value_usd <= 0:
        return 0.0
    h = max(inputs.horizon_days, 1e-6)
    denom = inputs.pip_value_usd * inputs.atr_daily_pips * math.sqrt(h)
    if denom <= 0:
        return 0.0
    return target_dollar_vol / denom


def vol_target_per_position(
    equity: float,
    n_active: int,
    *,
    sigma_target: float = VOL_TARGET_DAILY_DEFAULT,
) -> float:
    """Dollar-vol budget allotted to each of `n_active` open positions."""
    if equity <= 0 or n_active <= 0:
        return 0.0
    return (sigma_target * equity) / float(n_active)


def rebalance_open_positions(
    positions: list[dict],
    equity: float,
    *,
    sigma_target: float = VOL_TARGET_DAILY_DEFAULT,
) -> dict[str, float]:
    """For each open position, compute the lot size that would equalise
    the dollar-vol contribution across the portfolio.

    `positions` is a list of dicts with keys: symbol, atr_daily_pips,
    pip_value_usd, horizon_days. The returned mapping `symbol -> lot` is
    advisory — the order router decides whether to rebalance based on
    proximity to the suggested lot.
    """
    if equity <= 0 or not positions:
        return {}
    per = vol_target_per_position(equity, len(positions), sigma_target=sigma_target)
    out: dict[str, float] = {}
    for p in positions:
        inputs = VolTargetInputs(
            symbol=p["symbol"],
            atr_daily_pips=float(p.get("atr_daily_pips", 0.0)),
            pip_value_usd=float(p.get("pip_value_usd", 0.0)),
            horizon_days=float(p.get("horizon_days", 1.0)),
        )
        out[p["symbol"]] = target_lot(inputs, target_dollar_vol=per)
    return out
