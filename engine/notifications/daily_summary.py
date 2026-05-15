"""23:00 local daily summary job — pulls today's trades from SQLite,
posts a Discord embed + a toast notification.

Scheduled by the engine main loop (Phase 10's heartbeat extends to host
this once a day). Idempotent: safe to call multiple times — the embed is
purely a snapshot of current SQLite state.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from engine.data.sqlite_journal import open_journal
from engine.notifications import discord, windows_toast


def _today_iso_local() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def build_summary(date_str: str | None = None) -> dict:
    """Aggregate today's closed trades. Returns a dict suitable for the embed builder."""
    date_str = date_str or _today_iso_local()
    with open_journal() as con:
        rows = con.execute(
            "SELECT symbol, pnl FROM trades "
            "WHERE close_time IS NOT NULL AND substr(close_time,1,10)=? ",
            (date_str,),
        ).fetchall()
        eq_row = con.execute(
            "SELECT v FROM settings_kv WHERE k='last_equity'"
        ).fetchone()
        dd_row = con.execute(
            "SELECT v FROM settings_kv WHERE k='today_max_drawdown_pct'"
        ).fetchone()

    trades = len(rows)
    wins   = sum(1 for r in rows if (r["pnl"] or 0) > 0)
    losses = sum(1 for r in rows if (r["pnl"] or 0) < 0)
    net    = sum(float(r["pnl"] or 0) for r in rows)
    best   = max(rows, key=lambda r: r["pnl"] or 0, default=None)
    worst  = min(rows, key=lambda r: r["pnl"] or 0, default=None)
    return {
        "date_str": date_str,
        "trades": trades, "wins": wins, "losses": losses,
        "net_pnl": net,
        "equity": float(eq_row["v"]) if eq_row else 0.0,
        "best_trade":  f"{best['symbol']} ${best['pnl']:+.2f}"   if best  else "-",
        "worst_trade": f"{worst['symbol']} ${worst['pnl']:+.2f}" if worst else "-",
        "drawdown_max_pct": float(dd_row["v"]) if dd_row else 0.0,
    }


def post_daily_summary(date_str: str | None = None) -> bool:
    summary = build_summary(date_str)
    payload = discord.daily_summary(**summary)
    ok = discord.post(payload)
    windows_toast.notify(
        "TRAINING_COMPLETE",   # closest sound match (soft chime)
        title=f"Daily Summary {summary['date_str']}",
        body=f"{summary['trades']} trades, ${summary['net_pnl']:+.2f}, "
             f"{summary['wins']}W/{summary['losses']}L",
        sound="complete.wav",
    )
    logger.info("daily summary posted ok={} trades={}", ok, summary["trades"])
    return ok
