from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Literal

from loguru import logger


Session = Literal["LONDON_OPEN", "NY_OPEN", "LONDON_NY_OVERLAP", "ASIAN", "DEAD"]

ROLLING_WINDOW = 20
LOW_WIN_RATE_THRESHOLD = 0.30
MIN_TRADES_FOR_ADJUST = 10
CONSECUTIVE_LOSS_LIMIT = 3
CIRCUIT_BREAKER_MINUTES = 120
SESSION_MIN_TRADES = 30


@dataclass
class TradeOutcome:
    symbol: str
    pnl: float
    session: Session
    closed_at: datetime


@dataclass
class CircuitBreakerState:
    active: bool = False
    expires_at: datetime | None = None
    reason: str = ""


class PerformanceTracker:
    def __init__(self) -> None:
        self._results: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self._session_results: dict[Session, deque[int]] = defaultdict(lambda: deque(maxlen=200))
        self._recent_outcomes: deque[TradeOutcome] = deque(maxlen=50)
        self._size_overrides: dict[str, float] = {}
        self._breaker = CircuitBreakerState()
        self._lock = Lock()

    def record_trade(self, outcome: TradeOutcome) -> None:
        won = outcome.pnl > 0
        with self._lock:
            self._results[outcome.symbol].append(1 if won else 0)
            self._session_results[outcome.session].append(1 if won else 0)
            self._recent_outcomes.append(outcome)
            results = list(self._results[outcome.symbol])
            if len(results) >= MIN_TRADES_FOR_ADJUST:
                win_rate = sum(results) / len(results)
                self._size_overrides[outcome.symbol] = 0.5 if win_rate < LOW_WIN_RATE_THRESHOLD else 1.0
            self._maybe_trigger_breaker(outcome.closed_at)

    def size_multiplier(self, symbol: str) -> float:
        with self._lock:
            return self._size_overrides.get(symbol, 1.0)

    def session_win_rate(self, session: Session) -> float:
        with self._lock:
            results = list(self._session_results.get(session, ()))
        if not results:
            return 0.55
        return sum(results) / len(results)

    def session_confluence_minimum(self, session: Session) -> int:
        with self._lock:
            results = list(self._session_results.get(session, ()))
        if len(results) < SESSION_MIN_TRADES:
            return 3
        rate = sum(results) / len(results)
        if rate > 0.65:
            return 3
        if rate >= 0.50:
            return 3
        return 4

    def _maybe_trigger_breaker(self, now: datetime) -> None:
        recent = list(self._recent_outcomes)[-CONSECUTIVE_LOSS_LIMIT:]
        if len(recent) < CONSECUTIVE_LOSS_LIMIT:
            return
        if all(o.pnl < 0 for o in recent):
            self._breaker.active = True
            self._breaker.expires_at = now + timedelta(minutes=CIRCUIT_BREAKER_MINUTES)
            symbols = [o.symbol for o in recent]
            self._breaker.reason = f"CIRCUIT_BREAK: 3 consecutive losses ({symbols})"
            logger.error(self._breaker.reason)

    def circuit_breaker_active(self, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if not self._breaker.active:
                return False, ""
            if self._breaker.expires_at is None or now >= self._breaker.expires_at:
                self._breaker.active = False
                self._breaker.expires_at = None
                self._breaker.reason = ""
                logger.info("CIRCUIT_BREAK_CLEARED")
                return False, ""
            return True, self._breaker.reason


_singleton: PerformanceTracker | None = None


def get_performance_tracker() -> PerformanceTracker:
    global _singleton
    if _singleton is None:
        _singleton = PerformanceTracker()
    return _singleton
