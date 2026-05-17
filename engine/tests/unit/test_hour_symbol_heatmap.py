"""Tests for the 24×13 heatmap (Tier 7.3)."""
from __future__ import annotations

import tempfile

from engine.data import sqlite_journal
from engine.learning.hour_symbol_heatmap import (
    HeatmapCell,
    build_hour_symbol_grid,
    serialize_grid,
    worst_hours,
)


def _journal_with_trades(rows):
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    path = tmp.name
    with sqlite_journal.open_journal(path) as con:
        for r in rows:
            con.execute(
                "INSERT INTO trades(symbol, direction, entry_price, lot_size, sl, tp, "
                "pnl, open_time, close_time, close_reason) "
                "VALUES (?, 'BUY', 1.0, 0.1, 0.99, 1.01, ?, ?, ?, 'TP')",
                (r["symbol"], r["pnl"], r["open_time"], r["close_time"]),
            )
        con.commit()
    return path


def test_empty_journal_returns_empty_grid():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    with sqlite_journal.open_journal(tmp.name):
        pass
    grid = build_hour_symbol_grid(db_path=tmp.name)
    assert dict(grid) == {}


def test_grid_aggregates_per_hour_per_symbol():
    db = _journal_with_trades([
        {"symbol": "EURUSD#", "pnl": 10.0,
         "open_time": "2024-01-01T08:30:00+00:00", "close_time": "2024-01-01T09:30:00+00:00"},
        {"symbol": "EURUSD#", "pnl": -5.0,
         "open_time": "2024-01-01T08:45:00+00:00", "close_time": "2024-01-01T09:45:00+00:00"},
        {"symbol": "EURUSD#", "pnl": 20.0,
         "open_time": "2024-01-01T14:00:00+00:00", "close_time": "2024-01-01T15:00:00+00:00"},
    ])
    grid = build_hour_symbol_grid(db_path=db)
    assert grid["EURUSD#"][8].trades == 2
    assert grid["EURUSD#"][8].wins == 1
    assert grid["EURUSD#"][14].trades == 1


def test_serialize_grid_shape():
    cell = HeatmapCell(trades=3, wins=2, pnl_usd=15.0)
    grid = {"EURUSD#": {10: cell}}
    out = serialize_grid(grid)
    assert "EURUSD#" in out["symbols"]
    assert out["cells"]["EURUSD#"]["10"]["trades"] == 3
    assert out["cells"]["EURUSD#"]["10"]["expectancy"] == 5.0


def test_worst_hours_filters_by_min_trades():
    grid = {
        "EURUSD#": {
            9: HeatmapCell(trades=20, wins=2, pnl_usd=-300.0),     # bad
            10: HeatmapCell(trades=5, wins=0, pnl_usd=-100.0),     # bad but few trades
            11: HeatmapCell(trades=20, wins=15, pnl_usd=+200.0),   # good
        },
    }
    worst = worst_hours(grid, min_trades=10, top_n=2)
    assert worst[0][1] == 9
    assert len(worst) <= 2


def test_symbol_filter_excludes_others():
    db = _journal_with_trades([
        {"symbol": "EURUSD#", "pnl": 10.0,
         "open_time": "2024-01-01T08:30:00+00:00", "close_time": "2024-01-01T09:30:00+00:00"},
        {"symbol": "GOLD#", "pnl": -10.0,
         "open_time": "2024-01-01T08:30:00+00:00", "close_time": "2024-01-01T09:30:00+00:00"},
    ])
    grid = build_hour_symbol_grid(symbols=["EURUSD#"], db_path=db)
    assert "GOLD#" not in grid
    assert "EURUSD#" in grid
