"""Vectorized technical indicator pack (45 columns).

Built on pandas_ta_classic. All functions take a DataFrame with the lowercase
columns ``open, high, low, close, volume`` and return either a Series or a
DataFrame; ``compute_all()`` assembles them into one feature frame indexed
identically to the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

REQUIRED_COLS = ("open", "high", "low", "close", "volume")

# The 45 ordered feature names produced by compute_all().
FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1", "ret_5", "log_ret_1",
    "candle_body", "candle_upper_wick", "candle_lower_wick", "candle_range_pct",
    "sma_20",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "ema_9_21_diff", "ema_21_50_diff", "ema_50_200_diff",
    "vwap",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pctb",
    "atr_14", "atr_pct",
    "adx_14", "dmp_14", "dmn_14",
    "stoch_k", "stoch_d",
    "cci_20",
    "willr_14",
    "obv", "obv_slope",
    "mfi_14",
    "roc_10",
    "donchian_upper", "donchian_lower", "donchian_pct",
    "kama_30",
    "psar",
    "volume_z",
    "hour_of_day",
)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all 45 features, indexed like ``df``."""
    _validate(df)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    out = pd.DataFrame(index=df.index)

    # Returns + candle anatomy
    out["ret_1"] = c.pct_change()
    out["ret_5"] = c.pct_change(5)
    out["log_ret_1"] = np.log(c / c.shift(1))
    body = (c - o)
    rng = (h - l).replace(0, np.nan)
    out["candle_body"] = body / rng
    out["candle_upper_wick"] = (h - np.maximum(o, c)) / rng
    out["candle_lower_wick"] = (np.minimum(o, c) - l) / rng
    out["candle_range_pct"] = rng / c

    # Moving averages
    out["sma_20"] = ta.sma(c, length=20)
    out["ema_9"] = ta.ema(c, length=9)
    out["ema_21"] = ta.ema(c, length=21)
    out["ema_50"] = ta.ema(c, length=50)
    out["ema_200"] = ta.ema(c, length=200)
    out["ema_9_21_diff"] = (out["ema_9"] - out["ema_21"]) / c
    out["ema_21_50_diff"] = (out["ema_21"] - out["ema_50"]) / c
    out["ema_50_200_diff"] = (out["ema_50"] - out["ema_200"]) / c

    # VWAP — needs a DatetimeIndex; fall back to cumulative typical price if not present.
    try:
        out["vwap"] = ta.vwap(h, l, c, v)
    except Exception:
        tp = (h + l + c) / 3.0
        out["vwap"] = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)

    # Momentum
    out["rsi_14"] = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]
        out["macd_hist"] = macd.iloc[:, 1]
        out["macd_signal"] = macd.iloc[:, 2]
    else:
        out["macd"] = out["macd_hist"] = out["macd_signal"] = np.nan

    # Bollinger
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None and not bb.empty:
        out["bb_lower"] = bb.iloc[:, 0]
        out["bb_middle"] = bb.iloc[:, 1]
        out["bb_upper"] = bb.iloc[:, 2]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
        out["bb_pctb"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    else:
        for k in ("bb_lower","bb_middle","bb_upper","bb_width","bb_pctb"):
            out[k] = np.nan

    # Volatility
    out["atr_14"] = ta.atr(h, l, c, length=14)
    out["atr_pct"] = out["atr_14"] / c

    # Trend strength
    adx = ta.adx(h, l, c, length=14)
    if adx is not None and not adx.empty:
        out["adx_14"] = adx.iloc[:, 0]
        out["dmp_14"] = adx.iloc[:, 1]
        out["dmn_14"] = adx.iloc[:, 2]
    else:
        out["adx_14"] = out["dmp_14"] = out["dmn_14"] = np.nan

    # Stochastic
    st = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    if st is not None and not st.empty:
        out["stoch_k"] = st.iloc[:, 0]
        out["stoch_d"] = st.iloc[:, 1]
    else:
        out["stoch_k"] = out["stoch_d"] = np.nan

    out["cci_20"] = ta.cci(h, l, c, length=20)
    out["willr_14"] = ta.willr(h, l, c, length=14)

    # Volume
    out["obv"] = ta.obv(c, v)
    out["obv_slope"] = out["obv"].diff(5)
    out["mfi_14"] = ta.mfi(h, l, c, v, length=14)

    out["roc_10"] = ta.roc(c, length=10)

    # Donchian
    don = ta.donchian(h, l, lower_length=20, upper_length=20)
    if don is not None and not don.empty:
        out["donchian_lower"] = don.iloc[:, 0]
        out["donchian_upper"] = don.iloc[:, 2]
        out["donchian_pct"] = (c - out["donchian_lower"]) / (
            (out["donchian_upper"] - out["donchian_lower"]).replace(0, np.nan)
        )
    else:
        out["donchian_lower"] = out["donchian_upper"] = out["donchian_pct"] = np.nan

    out["kama_30"] = ta.kama(c, length=30)
    psar = ta.psar(h, l, c)
    if psar is not None and not psar.empty:
        long_col = next((col for col in psar.columns if col.startswith("PSARl_")), None)
        short_col = next((col for col in psar.columns if col.startswith("PSARs_")), None)
        long_v = psar[long_col] if long_col else pd.Series(np.nan, index=psar.index)
        short_v = psar[short_col] if short_col else pd.Series(np.nan, index=psar.index)
        out["psar"] = long_v.fillna(short_v)
    else:
        out["psar"] = np.nan

    vol_mean = v.rolling(50).mean()
    vol_std = v.rolling(50).std().replace(0, np.nan)
    out["volume_z"] = (v - vol_mean) / vol_std

    if isinstance(df.index, pd.DatetimeIndex):
        out["hour_of_day"] = df.index.hour.astype(float)
    else:
        out["hour_of_day"] = 0.0

    # Reorder to the canonical column list.
    return out[list(FEATURE_COLUMNS)]
