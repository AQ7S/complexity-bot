from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Literal, Mapping


Vote = Literal["BUY", "SELL", "HOLD"]

OFI_VOTE_THRESHOLD = 0.30
MIN_TICKS = 20
ARRIVAL_RATE_WINDOW_S = 60
TRADE_INTENSITY_HALFLIFE_S = 30


def _ts_seconds(ts) -> float:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        v = float(ts)
        if v > 1e12:  # millisecond epoch
            return v / 1000.0
        return v
    return 0.0


def compute_ofi(ticks: Iterable[Mapping]) -> float:
    items = list(ticks)
    if len(items) < MIN_TICKS:
        return 0.0
    recent = items[-MIN_TICKS:]
    bid_vol = 0.0
    ask_vol = 0.0
    for t in recent:
        vol = float(t.get("volume", 1) or 1)
        flags = int(t.get("flags", 0) or 0)
        if flags & 4:
            bid_vol += vol
        elif flags & 2:
            ask_vol += vol
        else:
            mid = (float(t.get("bid", 0)) + float(t.get("ask", 0))) / 2.0
            last = float(t.get("last", mid) or mid)
            if last > mid:
                ask_vol += vol
            elif last < mid:
                bid_vol += vol
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return (ask_vol - bid_vol) / total


def ofi_vote(ofi_score: float, threshold: float = OFI_VOTE_THRESHOLD) -> Vote:
    if ofi_score > threshold:
        return "BUY"
    if ofi_score < -threshold:
        return "SELL"
    return "HOLD"


def tick_arrival_rate(
    ticks: Iterable[Mapping],
    *,
    window_s: int = ARRIVAL_RATE_WINDOW_S,
    now_ts: float | None = None,
) -> float:
    """Ticks per second over the trailing `window_s` seconds.

    High arrival rate = active institutional period (open / NFP / FOMC).
    Low arrival rate during expected liquid hours = warning sign.
    """
    items = list(ticks)
    if not items or window_s <= 0:
        return 0.0
    if now_ts is None:
        now_ts = _ts_seconds(items[-1].get("ts", 0)) or 0.0
    cutoff = now_ts - float(window_s)
    count = 0
    for t in items:
        ts_s = _ts_seconds(t.get("ts", 0))
        if ts_s >= cutoff:
            count += 1
    return count / float(window_s)


def trade_intensity(
    ticks: Iterable[Mapping],
    *,
    half_life_s: float = TRADE_INTENSITY_HALFLIFE_S,
    now_ts: float | None = None,
) -> float:
    """Exponential time-decay weighted count of *price-changing* ticks.

    Quiet markets have many same-price ticks (no information flow); active
    markets have many price-changers. The decay weighting emphasizes recent
    ticks so the measure tracks current intensity rather than a stale window.
    """
    items = list(ticks)
    if not items or half_life_s <= 0:
        return 0.0
    if now_ts is None:
        now_ts = _ts_seconds(items[-1].get("ts", 0)) or 0.0
    decay = math.log(2.0) / float(half_life_s)
    last_price: float | None = None
    intensity = 0.0
    for t in items:
        bid = float(t.get("bid", 0.0) or 0.0)
        ask = float(t.get("ask", 0.0) or 0.0)
        if "last" in t and t["last"] is not None:
            price = float(t["last"])
        elif bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        else:
            price = bid if bid > 0 else ask
        if price <= 0:
            continue
        if last_price is not None and price != last_price:
            ts_s = _ts_seconds(t.get("ts", 0)) or now_ts
            age = max(0.0, now_ts - ts_s)
            intensity += math.exp(-decay * age)
        last_price = price
    return intensity
