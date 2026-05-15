"""ICT kill zone helpers + EST/UTC conversions.

Kill zone definitions follow standard ICT conventions, expressed in
US Eastern time (handles DST automatically via pytz). The four windows used
by the consensus engine are:

    Asian   session   19:00 – 22:00 EST   (Tokyo / Sydney overlap)
    London  open      02:00 – 05:00 EST
    NY      open      07:00 – 10:00 EST
    London  close     10:00 – 12:00 EST

Symbols flagged in `config.symbols.is_always_on()` (XAUUSD, BTCUSD#, ETHUSD#,
AI_INDX#, Crypto_10#) bypass these checks — `kill_zone_active()` returns True
unconditionally for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

import pytz

from engine.config.symbols import is_always_on

EST = pytz.timezone("US/Eastern")


@dataclass(frozen=True)
class KillZone:
    name: str
    start: time
    end: time

    def covers(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        # Wraps midnight (not used by the four below, but supported).
        return t >= self.start or t < self.end


KILL_ZONES: tuple[KillZone, ...] = (
    KillZone("ASIAN",        time(19, 0), time(22, 0)),
    KillZone("LONDON_OPEN",  time(2, 0),  time(5, 0)),
    KillZone("NY_OPEN",      time(7, 0),  time(10, 0)),
    KillZone("LONDON_CLOSE", time(10, 0), time(12, 0)),
)


def to_est(dt: datetime) -> datetime:
    """Convert a naive (assumed UTC) or aware datetime to US/Eastern."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(EST)


def active_kill_zone(dt: datetime) -> str | None:
    est = to_est(dt)
    t = est.time()
    for kz in KILL_ZONES:
        if kz.covers(t):
            return kz.name
    return None


def kill_zone_active(symbol: str, dt: datetime) -> bool:
    """True if the symbol may be traded right now under the ICT kill-zone rule."""
    if is_always_on(symbol):
        return True
    return active_kill_zone(dt) is not None
