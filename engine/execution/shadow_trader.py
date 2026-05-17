from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

from loguru import logger

from engine.config import settings
from engine.data.event_log import log_event
from engine.data.sqlite_journal import open_journal


Direction = Literal["BUY", "SELL"]
SHADOW_BARS_PER_HOUR = 12
SHADOW_TIME_EXIT_BARS = 48
MONITOR_INTERVAL_S = 60.0


@dataclass
class ShadowSignal:
    timestamp: datetime
    symbol: str
    direction: Direction
    entry_price: float
    sl_price: float
    tp_price: float
    confluence_score: int
    claude_decision: str | None
    claude_confidence: int | None
    model_version: str | None


def record_shadow_trade(signal: ShadowSignal, *, db_path: str | None = None) -> int:
    with open_journal(db_path) as con:
        cur = con.execute(
            """
            INSERT INTO shadow_trades (
                timestamp, symbol, direction, entry_price, sl_price, tp_price,
                claude_decision, claude_confidence, confluence_score, model_version,
                bars_held, hypothetical_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'OPEN')
            """,
            (
                signal.timestamp.isoformat(),
                signal.symbol, signal.direction,
                float(signal.entry_price), float(signal.sl_price), float(signal.tp_price),
                signal.claude_decision, signal.claude_confidence,
                int(signal.confluence_score), signal.model_version,
            ),
        )
        con.commit()
        new_id = int(cur.lastrowid)
    log_event("SHADOW_TRADE_OPENED", signal.symbol, {
        "id": new_id, "direction": signal.direction,
        "entry": signal.entry_price, "sl": signal.sl_price, "tp": signal.tp_price,
        "confluence": signal.confluence_score, "claude": signal.claude_decision,
    })
    return new_id


def _pip_size(symbol: str) -> float:
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if any(t in s for t in ("XAU", "GOLD", "BTC", "ETH", "INDX", "CRYPTO", "TRUMP", "HARRIS")):
        return 1.0
    return 0.0001


def _close_open_shadow(
    row: dict, current_price: float, now: datetime,
    *, db_path: str | None = None,
) -> str | None:
    direction = row["direction"]
    entry = float(row["entry_price"])
    sl = float(row["sl_price"])
    tp = float(row["tp_price"])
    opened_at = datetime.fromisoformat(row["timestamp"])
    bars_held = max(0, int((now - opened_at).total_seconds() // 300))
    outcome: str | None = None
    exit_price: float | None = None

    if direction == "BUY":
        if current_price <= sl:
            outcome, exit_price = "LOSS", sl
        elif current_price >= tp:
            outcome, exit_price = "WIN", tp
    else:
        if current_price >= sl:
            outcome, exit_price = "LOSS", sl
        elif current_price <= tp:
            outcome, exit_price = "WIN", tp

    if outcome is None and bars_held >= SHADOW_TIME_EXIT_BARS:
        outcome, exit_price = "TIME_EXIT", current_price

    if outcome is None:
        with open_journal(db_path) as con:
            con.execute(
                "UPDATE shadow_trades SET bars_held = ? WHERE id = ?",
                (bars_held, int(row["id"])),
            )
            con.commit()
        return None

    pip = _pip_size(row["symbol"])
    risk_pips = abs(entry - sl) / pip if pip > 0 else 0.0
    pnl_pips = (exit_price - entry) / pip if direction == "BUY" else (entry - exit_price) / pip
    pnl_r = pnl_pips / risk_pips if risk_pips > 0 else 0.0
    pnl_usd = pnl_r * (settings.RISK_PCT_PER_TRADE * 10_000.0)

    with open_journal(db_path) as con:
        con.execute(
            """
            UPDATE shadow_trades
               SET close_time = ?, exit_price = ?, hypothetical_outcome = ?,
                   pnl_r = ?, pnl_usd = ?, bars_held = ?
             WHERE id = ?
            """,
            (now.isoformat(), float(exit_price), outcome,
             float(pnl_r), float(pnl_usd), bars_held, int(row["id"])),
        )
        con.commit()
    log_event("SHADOW_TRADE_CLOSED", row["symbol"], {
        "id": int(row["id"]), "outcome": outcome,
        "exit": exit_price, "pnl_r": pnl_r,
    })
    try:
        from engine.models.replay_buffer import TradeExperience
        from engine.execution.order_router import _get_replay_buffer  # noqa: PLC2701
        buf = _get_replay_buffer()
        buf.add(TradeExperience(
            symbol=row["symbol"], timeframe="M5", timestamp=now.isoformat(),
            features=[entry, float(exit_price), float(row.get("confluence_score") or 0)],
            label=0 if direction == "BUY" else 1,
            pnl=float(pnl_usd), confluence=int(row.get("confluence_score") or 0),
            regime="SHADOW",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug("shadow replay append failed: {}", e)
    return outcome


def monitor_open_shadow_trades(
    price_lookup: dict[str, float],
    *, now: datetime | None = None, db_path: str | None = None,
) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    closed = {"WIN": 0, "LOSS": 0, "TIME_EXIT": 0}
    with open_journal(db_path) as con:
        rows = con.execute(
            "SELECT * FROM shadow_trades WHERE hypothetical_outcome = 'OPEN'"
        ).fetchall()
    for r in rows:
        sym = r["symbol"]
        if sym not in price_lookup:
            continue
        outcome = _close_open_shadow(dict(r), float(price_lookup[sym]), now, db_path=db_path)
        if outcome and outcome in closed:
            closed[outcome] += 1
    return closed


@dataclass
class ShadowStats:
    total: int
    open_count: int
    closed_count: int
    wins: int
    losses: int
    time_exits: int
    win_rate: float
    avg_r: float
    sharpe: float
    cumulative_pnl_r: float


def compute_shadow_stats(*, db_path: str | None = None) -> ShadowStats:
    with open_journal(db_path) as con:
        rows = con.execute(
            "SELECT hypothetical_outcome, pnl_r, pnl_usd FROM shadow_trades"
        ).fetchall()
    if not rows:
        return ShadowStats(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    total = len(rows)
    open_count = sum(1 for r in rows if r["hypothetical_outcome"] == "OPEN")
    closed = [r for r in rows if r["hypothetical_outcome"] != "OPEN"]
    closed_count = len(closed)
    wins = sum(1 for r in closed if r["hypothetical_outcome"] == "WIN")
    losses = sum(1 for r in closed if r["hypothetical_outcome"] == "LOSS")
    time_exits = sum(1 for r in closed if r["hypothetical_outcome"] == "TIME_EXIT")
    win_rate = wins / closed_count if closed_count else 0.0
    r_values = [float(r["pnl_r"] or 0.0) for r in closed]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    cumulative_r = sum(r_values)
    if len(r_values) > 1:
        mean = avg_r
        var = sum((x - mean) ** 2 for x in r_values) / (len(r_values) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0
    return ShadowStats(
        total=total, open_count=open_count, closed_count=closed_count,
        wins=wins, losses=losses, time_exits=time_exits,
        win_rate=win_rate, avg_r=avg_r, sharpe=sharpe,
        cumulative_pnl_r=cumulative_r,
    )


def is_promotion_ready(
    current_model_sharpe: float | None = None,
    *, db_path: str | None = None,
) -> tuple[bool, ShadowStats]:
    stats = compute_shadow_stats(db_path=db_path)
    if stats.closed_count < settings.SHADOW_PROMOTION_MIN_TRADES:
        return False, stats
    if stats.win_rate < settings.SHADOW_PROMOTION_WR_FLOOR:
        return False, stats
    bar = (current_model_sharpe or 0.0) * settings.SHADOW_PROMOTION_SHARPE_FACTOR
    return stats.sharpe > bar, stats


async def monitor_loop(stop_event: asyncio.Event, get_ticks):
    while not stop_event.is_set():
        try:
            prices = await asyncio.to_thread(get_ticks)
            await asyncio.to_thread(monitor_open_shadow_trades, prices)
        except Exception as e:  # noqa: BLE001
            logger.warning("shadow monitor: {}", e)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=MONITOR_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
