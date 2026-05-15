from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from loguru import logger


HISTORY_LEN = 20
SPREAD_REJECT_MULTIPLIER = 2.5
MIN_SAMPLES_BEFORE_GUARDING = 5


@dataclass
class SpreadVerdict:
    accepted: bool
    multiplier: float
    avg_spread: float
    reason: str


class SpreadMonitor:
    def __init__(self) -> None:
        self._spread_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=HISTORY_LEN)
        )
        self._lock = Lock()

    def update(self, symbol: str, current_spread: float) -> None:
        if current_spread < 0:
            return
        with self._lock:
            self._spread_history[symbol].append(float(current_spread))

    def average(self, symbol: str) -> float:
        with self._lock:
            history = self._spread_history.get(symbol)
            if not history:
                return 0.0
            return sum(history) / len(history)

    def multiplier(self, symbol: str, current_spread: float) -> float:
        avg = self.average(symbol)
        if avg <= 0:
            return 1.0
        return current_spread / avg

    def evaluate(self, symbol: str, current_spread: float) -> SpreadVerdict:
        with self._lock:
            history = list(self._spread_history.get(symbol, ()))
        if len(history) < MIN_SAMPLES_BEFORE_GUARDING:
            return SpreadVerdict(
                accepted=True,
                multiplier=1.0,
                avg_spread=current_spread,
                reason="WARMUP",
            )
        avg = sum(history) / len(history)
        mult = current_spread / avg if avg > 0 else 1.0
        if mult > SPREAD_REJECT_MULTIPLIER:
            logger.warning(
                f"SPREAD_BLOCK: {symbol} spread={current_spread:.5f} "
                f"avg={avg:.5f} multiplier={mult:.2f}× > {SPREAD_REJECT_MULTIPLIER}×"
            )
            return SpreadVerdict(
                accepted=False,
                multiplier=mult,
                avg_spread=avg,
                reason=f"SPREAD_BLOCK: {mult:.2f}× avg",
            )
        return SpreadVerdict(
            accepted=True, multiplier=mult, avg_spread=avg, reason="OK"
        )


_singleton: SpreadMonitor | None = None


def get_spread_monitor() -> SpreadMonitor:
    global _singleton
    if _singleton is None:
        _singleton = SpreadMonitor()
    return _singleton
