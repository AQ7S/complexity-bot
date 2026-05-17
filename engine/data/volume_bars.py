"""Volume bars — bar aggregation by volume traded, not by time.

A volume bar closes when accumulated volume since the last close reaches V
(the target volume per bar). This produces bars that are statistically more
stationary than time bars: weekend gaps disappear, overnight low-volume
periods don't generate noise, and intra-day liquidity bursts get their own
dedicated bars instead of being smeared into a single M5 candle.

Used as the primary input for high-frequency strategies (scalping in Tier 6).
Per López de Prado AFML ch. 2: dollar bars / volume bars consistently
outperform time bars on ML metrics for short-horizon prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class VolumeBar:
    ts_open: pd.Timestamp
    ts_close: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int


class VolumeBarBuilder:
    """Streaming builder. Call `add_tick()` per arriving tick; new bars
    emerge via `pop_completed()`. Suitable for the live data path.
    """

    def __init__(self, target_volume: float):
        if target_volume <= 0:
            raise ValueError("target_volume must be positive")
        self.target = float(target_volume)
        self._reset()
        self._completed: list[VolumeBar] = []

    def _reset(self) -> None:
        self._ts_open: pd.Timestamp | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._volume: float = 0.0
        self._tick_count: int = 0
        self._ts_last: pd.Timestamp | None = None

    def add_tick(
        self,
        ts: pd.Timestamp,
        price: float,
        volume: float,
    ) -> VolumeBar | None:
        """Append a tick. Returns a completed bar if this tick fills one."""
        if volume <= 0:
            return None
        if self._ts_open is None:
            self._ts_open = ts
            self._open = price
            self._high = price
            self._low = price
        assert self._high is not None and self._low is not None
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._ts_last = ts
        remaining = self.target - self._volume
        completed: VolumeBar | None = None
        if volume >= remaining:
            # Fill exactly to threshold; spill the rest into a new bar.
            self._volume += remaining
            self._tick_count += 1
            completed = self._snapshot()
            spill = volume - remaining
            self._reset()
            if spill > 0:
                # Start the next bar with the same tick's data and any spill.
                self._ts_open = ts
                self._open = price
                self._high = price
                self._low = price
                self._close = price
                self._volume = spill
                self._tick_count = 1
                self._ts_last = ts
                # If the spill itself exceeds another full bar, emit again.
                while self._volume >= self.target:
                    extra = self._snapshot()
                    overflow = self._volume - self.target
                    self._completed.append(extra)
                    self._reset()
                    if overflow <= 0:
                        break
                    self._ts_open = ts
                    self._open = price
                    self._high = price
                    self._low = price
                    self._close = price
                    self._volume = overflow
                    self._tick_count = 1
                    self._ts_last = ts
        else:
            self._volume += volume
            self._tick_count += 1
        return completed

    def _snapshot(self) -> VolumeBar:
        assert self._open is not None and self._high is not None
        assert self._low is not None and self._close is not None
        assert self._ts_open is not None and self._ts_last is not None
        return VolumeBar(
            ts_open=self._ts_open,
            ts_close=self._ts_last,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            tick_count=self._tick_count,
        )

    def pop_completed(self) -> list[VolumeBar]:
        out = self._completed
        self._completed = []
        return out


def from_tick_history(
    ticks: Iterable[Mapping],
    target_volume: float,
) -> list[VolumeBar]:
    """Batch-mode helper: aggregate a finite tick stream into volume bars."""
    builder = VolumeBarBuilder(target_volume=target_volume)
    bars: list[VolumeBar] = []
    for t in ticks:
        ts_raw = t.get("ts")
        if ts_raw is None:
            continue
        ts = pd.Timestamp(ts_raw)
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
        volume = float(t.get("volume", 1.0) or 1.0)
        completed = builder.add_tick(ts, price, volume)
        if completed:
            bars.append(completed)
        bars.extend(builder.pop_completed())
    return bars


def calibrate_target_volume(
    bars_m5: pd.DataFrame,
    *,
    target_bars_per_day: int = 288,
) -> float:
    """Suggest a `target_volume` so volume bars arrive at roughly the same
    rate as M5 bars during normal liquidity hours.

    288 = 24h * (60/5) — the M5 bar count per day. Adjust for higher- or
    lower-frequency strategies by scaling `target_bars_per_day`.
    """
    if "volume" not in bars_m5.columns or bars_m5.empty:
        return 1.0
    daily = bars_m5.copy()
    daily.index = pd.to_datetime(daily.index)
    grouped = daily.groupby(daily.index.normalize())["volume"].sum()
    if grouped.empty:
        return 1.0
    avg_daily_vol = float(grouped.mean())
    target_bars_per_day = max(1, target_bars_per_day)
    return max(avg_daily_vol / target_bars_per_day, 1.0)
