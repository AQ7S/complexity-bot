"""Tests for the extended daily report (Tier 7.4)."""
from __future__ import annotations

import tempfile
from datetime import datetime

from engine.data import sqlite_journal
from engine.notifications.daily_report import (
    build_report,
    to_discord_embed,
)


def _journal_with_trades(rows):
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    path = tmp.name
    with sqlite_journal.open_journal(path) as con:
        for r in rows:
            con.execute(
                "INSERT INTO trades(symbol, direction, entry_price, lot_size, sl, tp, "
                "pnl, open_time, close_time, close_reason, claude_reasoning) "
                "VALUES (?, ?, 1.0, 0.1, 0.99, 1.01, ?, ?, ?, ?, ?)",
                (r.get("symbol", "EURUSD#"),
                 r.get("direction", "BUY"),
                 r["pnl"],
                 r.get("open_time", "2024-01-01T08:00:00+00:00"),
                 r["close_time"],
                 r.get("close_reason", "TP"),
                 r.get("claude_reasoning", "")),
            )
        con.commit()
    return path


def test_empty_journal_yields_zero_report():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    with sqlite_journal.open_journal(tmp.name):
        pass
    r = build_report(date_str="2099-01-01", db_path=tmp.name)
    assert r["total_trades"] == 0
    assert r["per_strategy"] == []
    assert r["featured_loser"] is None


def test_report_aggregates_today_only():
    db = _journal_with_trades([
        {"pnl": +20.0, "close_time": "2024-05-17T10:00:00+00:00"},
        {"pnl": -10.0, "close_time": "2024-05-17T11:00:00+00:00"},
        {"pnl": +50.0, "close_time": "2024-05-16T11:00:00+00:00"},  # yesterday
    ])
    r = build_report(date_str="2024-05-17", db_path=db)
    assert r["total_trades"] == 2
    assert r["total_wins"] == 1
    assert r["total_losses"] == 1
    assert r["total_pnl_usd"] == 10.0


def test_featured_loser_identifies_worst_trade():
    db = _journal_with_trades([
        {"symbol": "EURUSD#", "pnl": -5.0, "close_time": "2024-05-17T09:00:00+00:00",
         "claude_reasoning": "thin OB"},
        {"symbol": "GOLD#", "pnl": -50.0, "close_time": "2024-05-17T10:00:00+00:00",
         "claude_reasoning": "overlap chop"},
        {"symbol": "BTCUSD#", "pnl": +30.0, "close_time": "2024-05-17T11:00:00+00:00"},
    ])
    r = build_report(date_str="2024-05-17", db_path=db)
    loser = r["featured_loser"]
    assert loser is not None
    assert loser["symbol"] == "GOLD#"
    assert loser["pnl"] == -50.0


def test_featured_loser_none_on_winning_day():
    db = _journal_with_trades([
        {"pnl": +5.0, "close_time": "2024-05-17T08:00:00+00:00"},
        {"pnl": +10.0, "close_time": "2024-05-17T09:00:00+00:00"},
    ])
    r = build_report(date_str="2024-05-17", db_path=db)
    assert r["featured_loser"] is None


def test_to_discord_embed_basic_shape():
    report = build_report(date_str="2099-01-01")
    payload = to_discord_embed(report)
    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert "Daily Operations Report" in embed["title"]
    assert any(f["name"] == "Trades" for f in embed["fields"])


def test_embed_color_red_when_losing():
    report = {
        "date_str": "2024-05-17",
        "total_trades": 1, "total_wins": 0, "total_losses": 1,
        "total_pnl_usd": -25.0,
        "per_strategy": [], "featured_loser": None,
        "latency_p95": "n/a",
    }
    payload = to_discord_embed(report)
    assert payload["embeds"][0]["color"] == 15158332  # RED


def test_embed_color_green_when_winning():
    report = {
        "date_str": "2024-05-17",
        "total_trades": 1, "total_wins": 1, "total_losses": 0,
        "total_pnl_usd": +25.0,
        "per_strategy": [], "featured_loser": None,
        "latency_p95": "n/a",
    }
    payload = to_discord_embed(report)
    assert payload["embeds"][0]["color"] == 3066993  # GREEN
