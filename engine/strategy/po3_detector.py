from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping


Po3Direction = Literal["BUY", "SELL", "NONE", "EXPIRED"]

SWEEP_PIPS_THRESHOLD = 10.0
RETURN_BAND_PIPS = 5.0
MAX_BARS_SINCE_OPEN = 6


@dataclass(frozen=True)
class Po3Verdict:
    direction: Po3Direction
    detected: bool
    sweep_pips: float
    reason: str


def _to_pips(diff: float, ref_price: float) -> float:
    if ref_price <= 0:
        return 0.0
    return diff / ref_price * 10000.0


def detect_po3(
    bars_since_session_open: Iterable[Mapping],
    session_open_price: float,
    pips_threshold: float = SWEEP_PIPS_THRESHOLD,
) -> Po3Verdict:
    bars = list(bars_since_session_open)
    if len(bars) < 3:
        return Po3Verdict("NONE", False, 0.0, "WARMUP: <3 bars since session open")
    if len(bars) > MAX_BARS_SINCE_OPEN:
        return Po3Verdict("EXPIRED", False, 0.0,
                          f"EXPIRED: {len(bars)} bars > {MAX_BARS_SINCE_OPEN}")

    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    current = float(bars[-1]["close"])
    session_high = max(highs)
    session_low = min(lows)
    sweep_up_pips = _to_pips(session_high - session_open_price, session_open_price)
    sweep_down_pips = _to_pips(session_open_price - session_low, session_open_price)
    return_pips = _to_pips(abs(current - session_open_price), session_open_price)
    returned_to_open = return_pips < RETURN_BAND_PIPS

    if sweep_down_pips > pips_threshold and current > session_open_price and returned_to_open:
        return Po3Verdict("BUY", True, sweep_down_pips,
                          f"BUY: down-sweep {sweep_down_pips:.1f}p then reclaim")
    if sweep_up_pips > pips_threshold and current < session_open_price and returned_to_open:
        return Po3Verdict("SELL", True, sweep_up_pips,
                          f"SELL: up-sweep {sweep_up_pips:.1f}p then reclaim")
    return Po3Verdict("NONE", False, max(sweep_up_pips, sweep_down_pips),
                      "NONE: no sweep+reclaim pattern")
