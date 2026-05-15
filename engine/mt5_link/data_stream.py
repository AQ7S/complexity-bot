"""Async tick + bar streamer that pumps live MT5 data into DuckDB.

Design notes:
- The official MetaTrader5 package has no callback API, so we poll
  `copy_ticks_from(symbol, last_ts, ...)` per symbol on a short interval.
  Polling is wrapped in `asyncio.to_thread` so the event loop stays free.
- Ticks are buffered in memory and flushed to DuckDB once per second.
- Bars use `copy_rates_from_pos` per (symbol, timeframe) on each event tick;
  duplicates are absorbed by the unique index on `bars`.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from engine.data import duckdb_store

TICK_POLL_INTERVAL_S = 0.25
TICK_FLUSH_INTERVAL_S = 1.0

TIMEFRAME_MAP: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


@dataclass
class StreamerState:
    last_tick_time_ms: dict[str, int] = field(default_factory=dict)
    tick_buffer: list[pd.DataFrame] = field(default_factory=list)
    tick_count: dict[str, int] = field(default_factory=dict)


def _ticks_since(symbol: str, last_ms: int) -> pd.DataFrame:
    """Pull ticks newer than `last_ms`; if last_ms is 0, pull last 1s only."""
    if last_ms == 0:
        from_ts = datetime.now(timezone.utc).timestamp() - 1.0
    else:
        from_ts = last_ms / 1000.0
    ticks = mt5.copy_ticks_from(symbol, from_ts, 10_000, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    if "time_msc" in df.columns:
        df = df[df["time_msc"] > last_ms].copy()
        df["ts"] = pd.to_datetime(df["time_msc"], unit="ms")
    else:
        df["ts"] = pd.to_datetime(df["time"], unit="s")
    if df.empty:
        return df
    out = pd.DataFrame({
        "symbol": symbol,
        "ts": df["ts"],
        "bid": df["bid"].astype(float),
        "ask": df["ask"].astype(float),
        "volume": df.get("volume", pd.Series([None] * len(df))),
        "flags": df.get("flags", pd.Series([None] * len(df))),
        "source": "mt5",
    })
    return out


def _poll_symbols_blocking(
    symbols: Sequence[str], state: StreamerState
) -> list[pd.DataFrame]:
    """Synchronous polling loop body (runs in a worker thread)."""
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        last_ms = state.last_tick_time_ms.get(sym, 0)
        df = _ticks_since(sym, last_ms)
        if df.empty:
            continue
        # Advance watermark to last seen tick.
        ms_series = (df["ts"].astype("int64") // 1_000_000).astype("int64")
        state.last_tick_time_ms[sym] = int(ms_series.max())
        state.tick_count[sym] = state.tick_count.get(sym, 0) + len(df)
        frames.append(df)
    return frames


async def stream_ticks(
    symbols: Sequence[str],
    *,
    duration_s: float,
    db_path: str | None = None,
) -> StreamerState:
    """Stream live ticks for `symbols` for `duration_s` seconds.

    Ticks are flushed to DuckDB every TICK_FLUSH_INTERVAL_S seconds.
    Returns the StreamerState (so callers can read per-symbol counts).
    """
    state = StreamerState()
    deadline = time.monotonic() + duration_s
    last_flush = time.monotonic()

    con = duckdb_store.connect(db_path)
    try:
        while time.monotonic() < deadline:
            frames = await asyncio.to_thread(_poll_symbols_blocking, list(symbols), state)
            if frames:
                state.tick_buffer.extend(frames)
            now = time.monotonic()
            if now - last_flush >= TICK_FLUSH_INTERVAL_S and state.tick_buffer:
                batch = pd.concat(state.tick_buffer, ignore_index=True)
                state.tick_buffer.clear()
                await asyncio.to_thread(duckdb_store.insert_ticks, con, batch)
                last_flush = now
            await asyncio.sleep(TICK_POLL_INTERVAL_S)

        # Final drain.
        if state.tick_buffer:
            batch = pd.concat(state.tick_buffer, ignore_index=True)
            state.tick_buffer.clear()
            await asyncio.to_thread(duckdb_store.insert_ticks, con, batch)
    finally:
        con.close()
    logger.info("tick stream finished: counts={}", state.tick_count)
    return state


def fetch_recent_bars(symbol: str, timeframe: str, n: int = 500) -> pd.DataFrame:
    tf = TIMEFRAME_MAP[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    out = pd.DataFrame({
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": pd.to_datetime(df["time"], unit="s"),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["tick_volume"].astype(float),
        "spread": df.get("spread"),
    })
    return out


async def snapshot_bars(
    symbols: Iterable[str],
    timeframes: Iterable[str] = ("M1", "M5", "M15", "H1", "H4", "D1"),
    *,
    n_per_timeframe: int = 500,
    db_path: str | None = None,
) -> dict[tuple[str, str], int]:
    """Pull the last N bars per (symbol, timeframe) into DuckDB. One-shot."""
    written: dict[tuple[str, str], int] = {}
    con = duckdb_store.connect(db_path)
    try:
        for sym in symbols:
            for tf in timeframes:
                df = await asyncio.to_thread(fetch_recent_bars, sym, tf, n_per_timeframe)
                if df.empty:
                    written[(sym, tf)] = 0
                    continue
                n = await asyncio.to_thread(duckdb_store.insert_bars, con, df)
                written[(sym, tf)] = n
    finally:
        con.close()
    return written
