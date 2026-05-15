"""4-class market regime classifier using ADX + ATR percentile + EMA structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

Regime = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY"]
ALL_REGIMES: tuple[Regime, ...] = ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY")

ADX_TREND_THRESHOLD = 25.0
ATR_PCTL_HIGH_VOL = 0.85


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: Regime
    adx: float
    atr_pct: float
    atr_percentile: float


def classify(df: pd.DataFrame, *, lookback: int = 200) -> RegimeSnapshot:
    """Return the regime as of the last bar of `df`."""
    h, l, c = df["high"], df["low"], df["close"]
    adx_df = ta.adx(h, l, c, length=14)
    atr = ta.atr(h, l, c, length=14)
    ema_fast = ta.ema(c, length=21)
    ema_slow = ta.ema(c, length=50)

    adx_now = float(adx_df.iloc[-1, 0]) if adx_df is not None and len(adx_df) else float("nan")
    atr_now = float(atr.iloc[-1])
    last_close = float(c.iloc[-1])
    atr_pct = atr_now / last_close if last_close else 0.0

    atr_window = atr.iloc[-lookback:].dropna()
    if len(atr_window) < 20:
        atr_pctl = 0.5
    else:
        atr_pctl = float((atr_window <= atr_now).mean())

    fast = float(ema_fast.iloc[-1])
    slow = float(ema_slow.iloc[-1])

    if not np.isnan(adx_now) and atr_pctl >= ATR_PCTL_HIGH_VOL and adx_now < ADX_TREND_THRESHOLD:
        regime: Regime = "HIGH_VOLATILITY"
    elif not np.isnan(adx_now) and adx_now >= ADX_TREND_THRESHOLD:
        regime = "TRENDING_UP" if fast > slow else "TRENDING_DOWN"
    else:
        regime = "RANGING"

    return RegimeSnapshot(regime=regime, adx=adx_now, atr_pct=atr_pct, atr_percentile=atr_pctl)
