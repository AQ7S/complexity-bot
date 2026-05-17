from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta_ext
import talib


def _last_float(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    if pd.isna(val):
        return default
    return float(val)


def _last_int(arr: np.ndarray) -> int:
    if arr is None or len(arr) == 0:
        return 0
    val = arr[-1]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val)


def compute_supplementary_features(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        raise ValueError(f"need at least 30 bars, got {len(df)}")

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    supertrend = ta_ext.supertrend(
        pd.Series(h), pd.Series(l), pd.Series(c), length=7, multiplier=3.0
    )
    squeeze = ta_ext.squeeze(
        pd.Series(h), pd.Series(l), pd.Series(c), bb_length=20, kc_length=20
    )
    hull9 = ta_ext.hma(pd.Series(c), length=9)
    fisher = ta_ext.fisher(pd.Series(h), pd.Series(l), length=9)
    qqe = ta_ext.qqe(pd.Series(c), length=14, smooth=5)

    cdl_engulf = talib.CDLENGULFING(o, h, l, c)
    cdl_hammer = talib.CDLHAMMER(o, h, l, c)
    cdl_star = talib.CDLSHOOTINGSTAR(o, h, l, c)
    cdl_doji = talib.CDLDOJI(o, h, l, c)
    cdl_morning = talib.CDLMORNINGSTAR(o, h, l, c)
    cdl_evening = talib.CDLEVENINGSTAR(o, h, l, c)
    cdl_3white = talib.CDL3WHITESOLDIERS(o, h, l, c)
    cdl_3black = talib.CDL3BLACKCROWS(o, h, l, c)

    bullish_cdl = any(v[-1] > 0 for v in [cdl_engulf, cdl_hammer, cdl_morning, cdl_3white])
    bearish_cdl = any(v[-1] < 0 for v in [cdl_engulf, cdl_star, cdl_evening, cdl_3black])
    candle_vote = 1 if bullish_cdl else (-1 if bearish_cdl else 0)

    supertrend_dir = 0
    if supertrend is not None and not supertrend.empty:
        for col in supertrend.columns:
            if "SUPERTd" in col:
                val = supertrend[col].iloc[-1]
                supertrend_dir = int(val) if not pd.isna(val) else 0
                break

    squeeze_on = False
    if squeeze is not None and not squeeze.empty:
        for col in squeeze.columns:
            if col.startswith("SQZ_ON") or "SQZ_ON" in col:
                val = squeeze[col].iloc[-1]
                squeeze_on = bool(val) if not pd.isna(val) else False
                break

    fisher_val = 0.0
    if fisher is not None and not fisher.empty:
        for col in fisher.columns:
            if "FISHERT" in col and "s" not in col.split("_")[0].lower():
                fisher_val = _last_float(fisher[col])
                break

    qqe_val = 0.0
    if qqe is not None and not qqe.empty:
        for col in qqe.columns:
            if "QQE" in col and "_" in col and col.split("_")[0] == "QQE":
                qqe_val = _last_float(qqe[col])
                break

    hull_val = _last_float(hull9, default=float(c[-1]))

    return {
        "supertrend_dir":  supertrend_dir,
        "squeeze_coiling": squeeze_on,
        "hull_ma9":        hull_val,
        "fisher_val":      fisher_val,
        "qqe_signal":      qqe_val,
        "candle_vote":     candle_vote,
        "cdl_detail": {
            "engulfing":        _last_int(cdl_engulf),
            "hammer":           _last_int(cdl_hammer),
            "shooting_star":    _last_int(cdl_star),
            "doji":             _last_int(cdl_doji),
            "morning_star":     _last_int(cdl_morning),
            "evening_star":     _last_int(cdl_evening),
            "3_white_soldiers": _last_int(cdl_3white),
            "3_black_crows":    _last_int(cdl_3black),
        },
    }
