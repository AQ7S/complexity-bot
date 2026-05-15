from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["BUY", "SELL"]
PdState = Literal["DISCOUNT", "PREMIUM", "EQUILIBRIUM"]

EQUILIBRIUM_BAND_PCT = 0.05
OTE_HIGH_FIB = 0.62
OTE_LOW_FIB = 0.79
OB_FRESH_MAX_TOUCHES = 1
FVG_OB_TOLERANCE_PCT = 0.001
H4_RANGING_ADX_THRESHOLD = 20.0
H4_TREND_LOOKBACK = 3


@dataclass(frozen=True)
class Zone:
    high: float
    low: float


@dataclass(frozen=True)
class OBZone(Zone):
    touches: int = 0


@dataclass
class SmcFilterResult:
    allow: bool
    reason: str
    premium_discount: PdState
    ote_active: bool
    ob_freshness: str
    fvg_ob_confluence: bool
    h4_aligned: bool
    counter_trend_allowed: bool = False


def premium_discount_state(
    current_price: float, session_high: float, session_low: float,
) -> PdState:
    if session_high <= session_low:
        return "EQUILIBRIUM"
    equilibrium = (session_high + session_low) / 2.0
    band = (session_high - session_low) * EQUILIBRIUM_BAND_PCT
    if abs(current_price - equilibrium) <= band:
        return "EQUILIBRIUM"
    return "DISCOUNT" if current_price < equilibrium else "PREMIUM"


def compute_ote_zone(
    swing_low: float, swing_high: float, direction: Direction,
) -> tuple[float, float]:
    rng = swing_high - swing_low
    if direction == "BUY":
        return swing_high - OTE_LOW_FIB * rng, swing_high - OTE_HIGH_FIB * rng
    return swing_low + OTE_HIGH_FIB * rng, swing_low + OTE_LOW_FIB * rng


def price_in_ote(current: float, ote_low: float, ote_high: float) -> bool:
    return ote_low <= current <= ote_high


def ob_freshness_label(touches: int) -> str:
    if touches <= 0:
        return "FRESH"
    if touches == OB_FRESH_MAX_TOUCHES:
        return "TAPPED_ONCE"
    return "CONSUMED"


def ob_is_fresh(touches: int) -> bool:
    return touches <= OB_FRESH_MAX_TOUCHES


def fvg_ob_confluence(
    current_price: float,
    ob_zones: list[OBZone],
    fvg_zones: list[Zone],
    tolerance_pct: float = FVG_OB_TOLERANCE_PCT,
) -> bool:
    for ob in ob_zones:
        ob_mid = (ob.high + ob.low) / 2.0
        if ob_mid == 0:
            continue
        for fvg in fvg_zones:
            fvg_mid = (fvg.high + fvg.low) / 2.0
            if abs(ob_mid - fvg_mid) / ob_mid < tolerance_pct:
                if ob.low <= current_price <= ob.high:
                    return True
    return False


def h4_bias_aligned(
    direction: Direction,
    h4_highs: list[float],
    h4_lows: list[float],
    adx_h4: float = 25.0,
    ranging_threshold: float = H4_RANGING_ADX_THRESHOLD,
) -> tuple[bool, str, bool]:
    if len(h4_highs) < H4_TREND_LOOKBACK + 1 or len(h4_lows) < H4_TREND_LOOKBACK + 1:
        return False, "RANGING", False
    bullish = (
        h4_highs[-1] > h4_highs[-1 - H4_TREND_LOOKBACK]
        and h4_lows[-1]  > h4_lows[-1 - H4_TREND_LOOKBACK]
    )
    bearish = (
        h4_highs[-1] < h4_highs[-1 - H4_TREND_LOOKBACK]
        and h4_lows[-1]  < h4_lows[-1 - H4_TREND_LOOKBACK]
    )
    if not bullish and not bearish:
        bias = "RANGING"
    else:
        bias = "BULLISH" if bullish else "BEARISH"
    counter_trend_allowed = adx_h4 < ranging_threshold
    if direction == "BUY" and bullish:
        return True, bias, counter_trend_allowed
    if direction == "SELL" and bearish:
        return True, bias, counter_trend_allowed
    return False, bias, counter_trend_allowed


def evaluate_smc_filters(
    direction: Direction,
    current_price: float,
    session_high: float,
    session_low: float,
    swing_high: float,
    swing_low: float,
    ob_zones: list[OBZone],
    fvg_zones: list[Zone],
    h4_highs: list[float],
    h4_lows: list[float],
    adx_h4: float,
) -> SmcFilterResult:
    pd = premium_discount_state(current_price, session_high, session_low)
    if direction == "BUY" and pd != "DISCOUNT":
        return SmcFilterResult(
            allow=False, reason=f"PD_REJECT: price in {pd}, BUY needs DISCOUNT",
            premium_discount=pd, ote_active=False, ob_freshness="N/A",
            fvg_ob_confluence=False, h4_aligned=False,
        )
    if direction == "SELL" and pd != "PREMIUM":
        return SmcFilterResult(
            allow=False, reason=f"PD_REJECT: price in {pd}, SELL needs PREMIUM",
            premium_discount=pd, ote_active=False, ob_freshness="N/A",
            fvg_ob_confluence=False, h4_aligned=False,
        )

    ote_low, ote_high = compute_ote_zone(swing_low, swing_high, direction)
    ote_active = price_in_ote(current_price, ote_low, ote_high)
    if not ote_active:
        return SmcFilterResult(
            allow=False, reason="OTE_REJECT: price outside 0.62-0.79 fib zone",
            premium_discount=pd, ote_active=False, ob_freshness="N/A",
            fvg_ob_confluence=False, h4_aligned=False,
        )

    fresh_obs = [ob for ob in ob_zones if ob_is_fresh(ob.touches)]
    if not fresh_obs:
        return SmcFilterResult(
            allow=False, reason="OB_REJECT: no fresh OBs available",
            premium_discount=pd, ote_active=True, ob_freshness="CONSUMED",
            fvg_ob_confluence=False, h4_aligned=False,
        )
    freshest = min(fresh_obs, key=lambda o: o.touches)
    ob_label = ob_freshness_label(freshest.touches)

    has_confluence = fvg_ob_confluence(current_price, fresh_obs, fvg_zones)
    if not has_confluence:
        return SmcFilterResult(
            allow=False, reason="CONFLUENCE_REJECT: no OB+FVG overlap at current price",
            premium_discount=pd, ote_active=True, ob_freshness=ob_label,
            fvg_ob_confluence=False, h4_aligned=False,
        )

    aligned, bias, ct_allowed = h4_bias_aligned(direction, h4_highs, h4_lows, adx_h4)
    if not aligned and not ct_allowed:
        return SmcFilterResult(
            allow=False,
            reason=f"H4_REJECT: H4 bias {bias} contradicts {direction} (ADX={adx_h4:.1f})",
            premium_discount=pd, ote_active=True, ob_freshness=ob_label,
            fvg_ob_confluence=True, h4_aligned=False, counter_trend_allowed=False,
        )

    return SmcFilterResult(
        allow=True,
        reason="OK" if aligned else f"COUNTER_TREND_ALLOWED: ADX={adx_h4:.1f} ranging",
        premium_discount=pd, ote_active=True, ob_freshness=ob_label,
        fvg_ob_confluence=True, h4_aligned=aligned, counter_trend_allowed=not aligned,
    )
