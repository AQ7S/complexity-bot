"""Risk manager — drawdown kills, position caps, news pause, emergency close.

The manager is mostly a set of pure decision functions over an injected
account snapshot + state. The only side-effecting helper is
`emergency_close_all()` which routes through `execution.order_router`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

from loguru import logger

from engine.config import settings
from engine.execution.order_router import close_all
from engine.mt5_link import account


@dataclass
class RiskState:
    week_start: date
    starting_balance_today: float
    starting_balance_week: float
    intraday_kill_armed: bool = False
    weekly_kill_armed: bool = False


def new_state(snapshot_balance: float, *, today: date | None = None) -> RiskState:
    today = today or datetime.now(timezone.utc).date()
    return RiskState(
        week_start=today,
        starting_balance_today=snapshot_balance,
        starting_balance_week=snapshot_balance,
    )


def reset_daily(state: RiskState, snapshot_balance: float) -> None:
    state.starting_balance_today = snapshot_balance
    state.intraday_kill_armed = False


def reset_weekly(state: RiskState, snapshot_balance: float, *, today: date | None = None) -> None:
    state.starting_balance_week = snapshot_balance
    state.starting_balance_today = snapshot_balance
    state.week_start = today or datetime.now(timezone.utc).date()
    state.weekly_kill_armed = False
    state.intraday_kill_armed = False


def evaluate_kill_triggers(
    state: RiskState, *, equity: float,
) -> tuple[bool, str, float]:
    """Return (kill_now, kind, drawdown_pct)."""
    if state.starting_balance_week > 0:
        wk_dd = max(0.0, (state.starting_balance_week - equity) / state.starting_balance_week)
        if wk_dd >= settings.WEEKLY_KILL_PCT and not state.weekly_kill_armed:
            return True, "WEEKLY", wk_dd
    if state.starting_balance_today > 0:
        day_dd = max(0.0, (state.starting_balance_today - equity) / state.starting_balance_today)
        if day_dd >= settings.INTRADAY_KILL_PCT and not state.intraday_kill_armed:
            return True, "INTRADAY", day_dd
    return False, "", 0.0


@dataclass(frozen=True)
class PreTradeChecks:
    is_paused: bool = False
    kill_active: bool = False
    open_positions: int = 0
    correlated_open: int = 0
    spread_widened: bool = False
    news_blocked: bool = False
    killzone_ok: bool = True


def can_open_new(checks: PreTradeChecks) -> tuple[bool, str | None]:
    """Pure precondition check; mirrors consensus.evaluate's order."""
    if checks.is_paused:                 return False, "PAUSED"
    if checks.kill_active:               return False, "KILL_ACTIVE"
    if checks.spread_widened:            return False, "SPREAD_WIDENED"
    if checks.open_positions >= settings.MAX_CONCURRENT_POSITIONS:
        return False, "MAX_POSITIONS"
    if checks.correlated_open >= settings.MAX_CORRELATED_POSITIONS:
        return False, "CORRELATION"
    if checks.news_blocked:              return False, "NEWS"
    if not checks.killzone_ok:           return False, "KILL_ZONE"
    return True, None


def emergency_close_all(*, reason: str = "MANUAL") -> dict[str, int]:
    """Close every open position via the order router."""
    logger.warning("EMERGENCY close_all reason={}", reason)
    return close_all(reason=reason)


def correlated_open_count(
    open_symbols: Iterable[str],
    candidate: str,
    correlation_matrix,  # pandas DataFrame keyed by symbol
    *,
    threshold: float = settings.CORRELATION_THRESHOLD,
) -> int:
    """Count open positions whose |corr| with `candidate` ≥ threshold."""
    if candidate not in correlation_matrix.columns:
        return 0
    n = 0
    for sym in open_symbols:
        if sym == candidate or sym not in correlation_matrix.columns:
            continue
        if abs(float(correlation_matrix.at[candidate, sym])) >= threshold:
            n += 1
    return n


def snapshot_account_drawdown(state: RiskState) -> dict:
    """Pull a live MT5 snapshot and return drawdown deltas vs the state's anchors."""
    snap = account.snapshot()
    return {
        "equity": snap.equity,
        "balance": snap.balance,
        "intraday_dd_pct": max(0.0, (state.starting_balance_today - snap.equity) / max(state.starting_balance_today, 1.0)),
        "weekly_dd_pct":   max(0.0, (state.starting_balance_week - snap.equity) / max(state.starting_balance_week, 1.0)),
        "open_positions": snap.open_positions,
    }
