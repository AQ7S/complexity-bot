"""Smart Money Concepts wrapper — multi-timeframe signal aggregator.

Uses the `smartmoneyconcepts` package (joshyattridge/smart-money-concepts).
The package returns a DataFrame for each indicator (despite the type hints
saying Series), so we treat them as DataFrames defensively.

Public API:
    detect_zones(df) -> dict of indicator DataFrames
    get_signal(h4_df, m15_df, m5_df) -> dict with the keys
        signal, zone_type, strength, entry, sl, tp
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from smartmoneyconcepts import smc

REQUIRED_COLS = ("open", "high", "low", "close")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """smartmoneyconcepts requires lowercase OHLC columns."""
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        raise ValueError(f"OHLC missing columns: {missing}; have {list(df.columns)}")
    out = pd.DataFrame({k: df[cols[k]].astype(float) for k in REQUIRED_COLS})
    if "volume" in cols:
        out["volume"] = df[cols["volume"]].astype(float)
    out.index = df.index
    return out


def detect_zones(df: pd.DataFrame, *, swing_length: int = 50) -> dict[str, pd.DataFrame]:
    """Run every smc detector against `df`. Returns a dict of result frames."""
    o = _normalize(df)
    swings = smc.swing_highs_lows(o, swing_length=swing_length)
    return {
        "swings":     swings,
        "ob":         smc.ob(o, swings),
        "fvg":        smc.fvg(o),
        "bos_choch":  smc.bos_choch(o, swings),
        "liquidity":  smc.liquidity(o, swings),
    }


def _last_active(frame: pd.DataFrame | pd.Series, value_cols: list[str] | None = None) -> dict | None:
    """Return the last row that has a non-NaN value in `value_cols` (or any col)."""
    if frame is None or len(frame) == 0:
        return None
    f = frame if isinstance(frame, pd.DataFrame) else frame.to_frame()
    cols = value_cols or list(f.columns)
    mask = f[cols].notna().any(axis=1)
    if not mask.any():
        return None
    idx = mask[mask].index[-1]
    row = f.loc[idx].to_dict()
    row["_index"] = idx
    return row


def _h4_bias(h4_df: pd.DataFrame) -> str:
    """Determine higher-timeframe bias from latest BOS/CHoCH direction."""
    zones = detect_zones(h4_df, swing_length=20)
    bos = zones["bos_choch"]
    if bos is None or len(bos) == 0:
        return "NONE"
    last = _last_active(bos, value_cols=[c for c in bos.columns if c.lower() in ("bos", "choch")])
    if last is None:
        return "NONE"
    # Each BOS/CHoCH row has a direction column (1 = bullish, -1 = bearish).
    direction = None
    for k in ("Direction", "direction"):
        if k in last:
            direction = last[k]
            break
    if direction is None:
        return "NONE"
    return "BULL" if direction > 0 else "BEAR"


@dataclass(frozen=True)
class SmcSignal:
    signal: str        # BUY | SELL | HOLD
    zone_type: str     # OB | FVG | NONE
    strength: float
    entry: float | None
    sl: float | None
    tp: float | None


def get_signal(
    h4_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    *,
    rr: float = 2.0,
) -> SmcSignal:
    """Compose H4 bias → M15 setup zone → M5 entry into a single signal."""
    bias = _h4_bias(h4_df)
    if bias == "NONE":
        return SmcSignal("HOLD", "NONE", 0.0, None, None, None)

    m15_zones = detect_zones(m15_df, swing_length=20)
    ob = m15_zones["ob"]
    fvg = m15_zones["fvg"]

    # Prefer the freshest active OB; fall back to FVG.
    chosen = None
    zone_type = "NONE"
    if ob is not None and len(ob) > 0:
        last_ob = _last_active(ob, value_cols=[c for c in ob.columns if c.lower() == "ob"])
        if last_ob is not None:
            chosen, zone_type = last_ob, "OB"
    if chosen is None and fvg is not None and len(fvg) > 0:
        last_fvg = _last_active(fvg, value_cols=[c for c in fvg.columns if c.lower() == "fvg"])
        if last_fvg is not None:
            chosen, zone_type = last_fvg, "FVG"
    if chosen is None:
        return SmcSignal("HOLD", "NONE", 0.0, None, None, None)

    # Direction must agree with H4 bias.
    direction = chosen.get("Direction") or chosen.get("direction")
    if direction is None:
        return SmcSignal("HOLD", zone_type, 0.0, None, None, None)
    zone_dir = "BULL" if direction > 0 else "BEAR"
    if zone_dir != bias:
        return SmcSignal("HOLD", zone_type, 0.0, None, None, None)

    top = chosen.get("Top") or chosen.get("top")
    bot = chosen.get("Bottom") or chosen.get("bottom")
    if top is None or bot is None or np.isnan(top) or np.isnan(bot):
        return SmcSignal("HOLD", zone_type, 0.0, None, None, None)

    last_close = float(m5_df["close"].iloc[-1])
    if zone_dir == "BULL":
        entry = float(top)  # buy at zone top
        sl = float(bot)
        tp = entry + rr * (entry - sl)
        signal = "BUY"
    else:
        entry = float(bot)
        sl = float(top)
        tp = entry - rr * (sl - entry)
        signal = "SELL"

    # Strength: distance from current price scaled to ATR-ish window.
    rng = float(m5_df["close"].iloc[-50:].std() or 1e-9)
    proximity = max(0.0, 1.0 - abs(last_close - entry) / (rng * 5))
    strength = round(50 + 50 * proximity, 2)

    return SmcSignal(signal, zone_type, strength, entry, sl, tp)
