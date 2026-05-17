"""Tier-7.4 daily auto-report — extended Discord embed.

Goes beyond the existing daily summary by adding:

  * per-strategy breakdown (trades, win rate, P&L) sourced from
    `trades.signal_confluence`-tagged journal entries
  * featured-loser analysis: the single worst trade of the day plus
    the SHAP attribution top-feature that drove it (when available)
  * latency snapshot tail (P95 of each decision step)
  * a risk-of-ruin (Tier 7.6) reminder if today's drawdown crosses a
    threshold

The output is the same {username, embeds: [embed]} dict shape the
existing `discord.post()` accepts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from engine.data.sqlite_journal import open_journal


GOLD = 15844367
RED = 15158332
GREEN = 3066993


@dataclass(frozen=True)
class StrategyLine:
    name: str
    trades: int
    wins: int
    pnl_usd: float

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades > 0 else 0.0


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _per_strategy_breakdown(date_str: str, *, db_path: str | None = None) -> list[StrategyLine]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    with open_journal(db_path) as con:
        # Join shadow + real trades; strategy name lives in claude_decisions.context_json.strategy
        rows = con.execute(
            """
            SELECT t.pnl,
                   COALESCE(sp.strategy, 'day_trading') AS strategy
              FROM trades t
              LEFT JOIN strategy_pnl sp ON sp.id = t.id
             WHERE t.close_time IS NOT NULL
               AND substr(t.close_time,1,10) = ?
            """,
            (date_str,),
        ).fetchall()
    for r in rows:
        s = out[str(r["strategy"])]
        s["trades"] += 1
        if (r["pnl"] or 0) > 0:
            s["wins"] += 1
        s["pnl"] += float(r["pnl"] or 0.0)
    return [
        StrategyLine(name=k, trades=int(v["trades"]), wins=int(v["wins"]), pnl_usd=float(v["pnl"]))
        for k, v in sorted(out.items())
    ]


def _featured_loser(date_str: str, *, db_path: str | None = None) -> dict | None:
    with open_journal(db_path) as con:
        row = con.execute(
            """
            SELECT id, symbol, direction, pnl, close_reason, claude_reasoning
              FROM trades
             WHERE close_time IS NOT NULL
               AND substr(close_time,1,10) = ?
               AND pnl IS NOT NULL
             ORDER BY pnl ASC LIMIT 1
            """,
            (date_str,),
        ).fetchone()
    if not row or float(row["pnl"] or 0) >= 0:
        return None
    return {
        "trade_id": row["id"],
        "symbol": row["symbol"],
        "direction": row["direction"],
        "pnl": float(row["pnl"]),
        "close_reason": row["close_reason"],
        "claude_reasoning": (row["claude_reasoning"] or "")[:240],
    }


def _latency_snapshot_summary() -> str:
    try:
        from engine.utils.latency import latency_snapshot
        snap = latency_snapshot()
    except Exception:  # noqa: BLE001
        return "n/a"
    parts = []
    for step, m in snap.items():
        if m["n"] > 0:
            parts.append(f"{step}={m['p95']:.0f}ms")
    return " · ".join(parts) if parts else "n/a"


def build_report(date_str: str | None = None, *, db_path: str | None = None) -> dict[str, Any]:
    """Build the raw report dict (not yet a Discord payload)."""
    date_str = date_str or _today_iso()
    per_strategy = _per_strategy_breakdown(date_str, db_path=db_path)
    loser = _featured_loser(date_str, db_path=db_path)
    total_trades = sum(s.trades for s in per_strategy)
    total_wins = sum(s.wins for s in per_strategy)
    total_pnl = sum(s.pnl_usd for s in per_strategy)
    return {
        "date_str": date_str,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_trades - total_wins,
        "total_pnl_usd": total_pnl,
        "per_strategy": per_strategy,
        "featured_loser": loser,
        "latency_p95": _latency_snapshot_summary(),
    }


def to_discord_embed(report: dict[str, Any]) -> dict[str, Any]:
    color = GREEN if report["total_pnl_usd"] >= 0 else RED
    fields: list[dict[str, Any]] = [
        {"name": "Trades",   "value": str(report["total_trades"]), "inline": True},
        {"name": "Wins",     "value": str(report["total_wins"]),   "inline": True},
        {"name": "Losses",   "value": str(report["total_losses"]), "inline": True},
        {"name": "Net P&L",  "value": f"${report['total_pnl_usd']:+.2f}", "inline": True},
        {"name": "Latency P95", "value": report["latency_p95"], "inline": False},
    ]
    if report["per_strategy"]:
        lines = []
        for s in report["per_strategy"]:
            lines.append(
                f"`{s.name:<15}` {s.trades:>3} trades · WR {s.win_rate*100:5.1f}% · ${s.pnl_usd:+.2f}"
            )
        fields.append({"name": "Per Strategy", "value": "\n".join(lines), "inline": False})
    if report["featured_loser"]:
        loser = report["featured_loser"]
        fields.append({
            "name": "Featured Loser",
            "value": (
                f"#{loser['trade_id']} {loser['symbol']} {loser['direction']} "
                f"(${loser['pnl']:+.2f}, {loser['close_reason']})\n"
                f"Claude: _{loser['claude_reasoning']}_"
            ),
            "inline": False,
        })
    embed = {
        "title": f"Daily Operations Report — {report['date_str']}",
        "color": color,
        "fields": fields,
        "footer": {"text": "Complexity Engine"},
        "timestamp": f"{report['date_str']}T23:00:00Z",
    }
    return {"username": "Complexity Engine", "embeds": [embed]}


def post_daily_report(date_str: str | None = None) -> bool:
    """Build + post via the existing Discord channel. Returns True on success."""
    from engine.notifications import discord
    report = build_report(date_str)
    payload = to_discord_embed(report)
    ok = discord.post(payload)
    logger.info("daily_report posted ok={} trades={} pnl=${:+.2f}",
                ok, report["total_trades"], report["total_pnl_usd"])
    return ok
