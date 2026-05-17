"""24-hour × 13-symbol performance heatmap.

The existing session-level heatmap (London / NY / Asian) is too coarse:
within a single "NY session" a symbol can be profitable in the first
hour and a guaranteed loser in the last hour. A 24×13 grid gives the
operator the resolution needed to switch off bad hours per symbol via
each strategy's `hour_blacklist`.

The aggregator reads closed `trades` from SQLite, buckets each by
(symbol, UTC-hour-of-entry), and computes per-cell:

  * `trades`     — count
  * `wins`       — wins
  * `win_rate`   — wins/trades
  * `pnl_usd`    — sum
  * `expectancy` — avg pnl per trade

The output is dict-keyed for direct UI consumption (`grid[symbol][hour]`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from engine.data.sqlite_journal import open_journal


@dataclass
class HeatmapCell:
    trades: int = 0
    wins: int = 0
    pnl_usd: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades > 0 else 0.0

    @property
    def expectancy(self) -> float:
        return (self.pnl_usd / self.trades) if self.trades > 0 else 0.0


def _parse_hour(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.hour)


def build_hour_symbol_grid(
    symbols: Iterable[str] | None = None,
    *,
    db_path: str | None = None,
    lookback_days: int | None = None,
) -> dict[str, dict[int, HeatmapCell]]:
    """Build the 13×24 grid from closed trades.

    `lookback_days=None` uses ALL history; pass an integer to restrict.
    `symbols` filter is optional — when None, every symbol seen in the
    journal is included.
    """
    sym_filter = set(symbols) if symbols else None
    grid: dict[str, dict[int, HeatmapCell]] = defaultdict(lambda: defaultdict(HeatmapCell))

    sql = "SELECT symbol, open_time, pnl FROM trades WHERE pnl IS NOT NULL"
    params: list = []
    if lookback_days is not None and lookback_days > 0:
        from datetime import timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(lookback_days))).isoformat()
        sql += " AND open_time >= ?"
        params.append(cutoff)

    with open_journal(db_path) as con:
        rows = con.execute(sql, params).fetchall()
    for r in rows:
        sym = str(r["symbol"])
        if sym_filter is not None and sym not in sym_filter:
            continue
        hour = _parse_hour(r["open_time"])
        if hour is None:
            continue
        pnl = float(r["pnl"] or 0.0)
        cell = grid[sym][hour]
        cell.trades += 1
        if pnl > 0:
            cell.wins += 1
        cell.pnl_usd += pnl
    return grid


def serialize_grid(grid: dict[str, dict[int, HeatmapCell]]) -> dict:
    """Return a JSON-friendly snapshot suitable for IPC broadcast."""
    out: dict = {"symbols": [], "hours": list(range(24)), "cells": {}}
    for sym, hours in sorted(grid.items()):
        out["symbols"].append(sym)
        out["cells"][sym] = {
            str(h): {
                "trades": int(c.trades),
                "wins": int(c.wins),
                "win_rate": float(c.win_rate),
                "pnl_usd": float(c.pnl_usd),
                "expectancy": float(c.expectancy),
            }
            for h, c in hours.items()
        }
    return out


def worst_hours(
    grid: dict[str, dict[int, HeatmapCell]],
    *,
    min_trades: int = 10,
    top_n: int = 10,
) -> list[tuple[str, int, float]]:
    """Return the worst (symbol, hour, expectancy) cells with ≥ min_trades.

    Useful for surfacing a "ban these hours from this symbol" recommendation.
    """
    candidates: list[tuple[str, int, float]] = []
    for sym, hours in grid.items():
        for h, c in hours.items():
            if c.trades >= min_trades:
                candidates.append((sym, h, c.expectancy))
    candidates.sort(key=lambda t: t[2])
    return candidates[:top_n]
