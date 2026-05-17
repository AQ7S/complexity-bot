"""Per-symbol × per-hour spread profile.

The existing spread monitor compares the current spread to a single global
rolling average. That misses the structural fact that spreads have a
predictable diurnal shape — XAUUSD at 03:00 UTC has fundamentally different
typical spreads than XAUUSD at 10:00 UTC, and a blanket multiplier either
blocks too much in the dead session or admits too much during the open.

This module learns a per-symbol × per-hour median spread from
spread_history (DuckDB) over the trailing N days, and exposes a fast
`is_spread_acceptable()` gate that the consensus engine can call before
admitting a signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MULTIPLIER = 1.5
MIN_SAMPLES_PER_HOUR = 30


def _median(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


@dataclass
class HourlyProfile:
    """24-slot profile of median spread + sample count per UTC hour."""

    symbol: str
    medians: list[float] = field(default_factory=lambda: [0.0] * 24)
    counts: list[int] = field(default_factory=lambda: [0] * 24)
    fallback_median: float = 0.0

    def median_at(self, utc_hour: int) -> float:
        if not (0 <= utc_hour < 24):
            return self.fallback_median
        if self.counts[utc_hour] < MIN_SAMPLES_PER_HOUR:
            return self.fallback_median if self.fallback_median > 0 else self.medians[utc_hour]
        return self.medians[utc_hour]


def build_hourly_profile(
    samples: Iterable[Mapping],
    *,
    symbol: str,
) -> HourlyProfile:
    """Build a profile from an iterable of {ts: datetime, spread: float} dicts.

    `samples` is normally the result of a SELECT on spread_history; the test
    suite passes synthetic dicts.
    """
    by_hour: list[list[float]] = [[] for _ in range(24)]
    all_spreads: list[float] = []
    for s in samples:
        ts = s.get("ts")
        if not isinstance(ts, datetime):
            continue
        spread = float(s.get("spread", 0.0) or 0.0)
        if spread <= 0:
            continue
        h = ts.astimezone(timezone.utc).hour if ts.tzinfo else ts.hour
        by_hour[h].append(spread)
        all_spreads.append(spread)
    profile = HourlyProfile(symbol=symbol)
    for h in range(24):
        bucket = by_hour[h]
        profile.medians[h] = _median(bucket)
        profile.counts[h] = len(bucket)
    profile.fallback_median = _median(all_spreads)
    return profile


def load_hourly_profile_from_duckdb(
    symbol: str,
    *,
    days_lookback: int = DEFAULT_LOOKBACK_DAYS,
    db_path: str | None = None,
) -> HourlyProfile:
    """Pull spread_history for `symbol` from DuckDB and build a profile."""
    from engine.data import duckdb_store
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    with duckdb_store.open_store(db_path, read_only=True) as con:
        rows = con.execute(
            "SELECT ts, spread FROM spread_history "
            "WHERE symbol = ? AND ts >= ? ORDER BY ts",
            [symbol, cutoff],
        ).fetchall()
    samples = [{"ts": r[0], "spread": float(r[1])} for r in rows]
    return build_hourly_profile(samples, symbol=symbol)


def is_spread_acceptable(
    profile: HourlyProfile,
    current_spread: float,
    *,
    utc_hour: int | None = None,
    multiplier: float = DEFAULT_MULTIPLIER,
    now: datetime | None = None,
) -> bool:
    """True iff `current_spread` ≤ multiplier × hour-specific median."""
    if utc_hour is None:
        ref = now or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        utc_hour = ref.astimezone(timezone.utc).hour
    median = profile.median_at(utc_hour)
    if median <= 0:
        return True
    return current_spread <= multiplier * median


def expected_spread(profile: HourlyProfile, utc_hour: int) -> float:
    return profile.median_at(utc_hour)
