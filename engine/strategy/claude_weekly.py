"""Monday 06:00 UTC weekly debrief job.

Pulls the last 7 days of trades + signals from SQLite, sends them to Claude
asking for the structured markdown debrief specified in §12, persists the
result to Supabase `weekly_debriefs`. The cron-style schedule is wired in
the engine main loop via the `schedule` library.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from anthropic import Anthropic
from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal

WEEKLY_SYSTEM_PROMPT = (
    "You are a senior trading-systems analyst. You receive a JSON payload of the "
    "engine's trades + signals + drawdown profile from the previous week. "
    "Reply with markdown ONLY (no JSON wrapper) using EXACTLY these sections:\n"
    "## Top 3 Mistakes\n## Top 3 Successful Patterns\n"
    "## Recommended Parameter Changes (JSON)\n```json\n{...}\n```\n"
    "## Market Conditions Summary\n"
    "Be concise and actionable; focus on patterns the engine can encode."
)


def _week_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or datetime.now(timezone.utc)
    end = now
    start = end - timedelta(days=7)
    return start, end


def collect_week_payload(con: sqlite3.Connection, *, now: datetime | None = None) -> dict[str, Any]:
    start, end = _week_window(now)
    trades = con.execute(
        "SELECT * FROM trades WHERE open_time >= ? AND open_time <= ? ORDER BY open_time",
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    signals = con.execute(
        "SELECT * FROM signals_log WHERE ts >= ? AND ts <= ? ORDER BY ts",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    trades_d = [dict(r) for r in trades]
    pnl = sum(float(t.get("pnl") or 0) for t in trades_d)
    wins = sum(1 for t in trades_d if (t.get("pnl") or 0) > 0)
    losses = sum(1 for t in trades_d if (t.get("pnl") or 0) < 0)

    return {
        "week_start": start.date().isoformat(),
        "week_end": end.date().isoformat(),
        "n_trades": len(trades_d),
        "n_wins": wins,
        "n_losses": losses,
        "net_pnl": pnl,
        "trades": trades_d,
        "signals": [dict(r) for r in signals],
    }


def run_debrief(*, now: datetime | None = None, model: str | None = None) -> dict:
    """Generate the weekly markdown debrief and persist to Supabase."""
    if not settings.have_anthropic():
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    with open_journal() as con:
        payload = collect_week_payload(con, now=now)
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model or settings.CLAUDE_MODEL,
        max_tokens=4096,
        system=WEEKLY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        timeout=60,
    )
    markdown = "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", "") == "text"
    )

    record = {
        "week_start": payload["week_start"],
        "markdown": markdown,
        "trades_count": payload["n_trades"],
        "net_pnl": payload["net_pnl"],
        "win_rate": (payload["n_wins"] / payload["n_trades"]) if payload["n_trades"] else 0.0,
    }

    if settings.have_supabase():
        try:
            from engine.supabase_sync.client import get_client
            client = get_client()
            client.table("weekly_debriefs").upsert(record, on_conflict="week_start").execute()
            logger.info("weekly debrief synced to Supabase: week_start={}", record["week_start"])
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase weekly debrief upsert failed: {}", e)

    return record
