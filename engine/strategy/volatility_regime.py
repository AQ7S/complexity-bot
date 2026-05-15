from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from loguru import logger


Regime = Literal["LOW", "NORMAL", "HIGH", "EXTREME"]

ATR_HISTORY_LEN = 200
EXTREME_PCT = 0.90
HIGH_PCT = 0.70
LOW_PCT = 0.20


@dataclass
class RegimeVerdict:
    regime: Regime
    percentile: float
    atr14: float
    risk_multiplier: float
    sl_multiplier: float
    blocked: bool
    reason: str


class VolatilityRegimeTracker:
    def __init__(self) -> None:
        self._atr_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=ATR_HISTORY_LEN)
        )
        self._lock = Lock()

    def update(self, symbol: str, atr14: float) -> None:
        if atr14 <= 0:
            return
        with self._lock:
            self._atr_history[symbol].append(float(atr14))

    def classify(self, symbol: str, atr14_current: float) -> RegimeVerdict:
        with self._lock:
            history = list(self._atr_history.get(symbol, ()))
        if len(history) < 20:
            return RegimeVerdict(
                regime="NORMAL",
                percentile=0.5,
                atr14=atr14_current,
                risk_multiplier=1.0,
                sl_multiplier=1.0,
                blocked=False,
                reason="WARMUP",
            )
        below = sum(1 for a in history if a < atr14_current)
        pct = below / len(history)
        if pct > EXTREME_PCT:
            regime: Regime = "EXTREME"
            risk_mult = 0.0
            sl_mult = 2.0
            blocked = True
            reason = f"VOL_EXTREME_BLOCK: pct={pct:.2f}"
            logger.warning(f"{symbol} {reason} atr={atr14_current:.5f}")
        elif pct > HIGH_PCT:
            regime = "HIGH"
            risk_mult = 0.5
            sl_mult = 1.5
            blocked = False
            reason = f"HIGH_VOL: pct={pct:.2f}, reduced size"
        elif pct < LOW_PCT:
            regime = "LOW"
            risk_mult = 0.75
            sl_mult = 0.8
            blocked = False
            reason = f"LOW_VOL: pct={pct:.2f}, tight SL"
        else:
            regime = "NORMAL"
            risk_mult = 1.0
            sl_mult = 1.0
            blocked = False
            reason = "NORMAL"
        return RegimeVerdict(
            regime=regime,
            percentile=pct,
            atr14=atr14_current,
            risk_multiplier=risk_mult,
            sl_multiplier=sl_mult,
            blocked=blocked,
            reason=reason,
        )


_singleton: VolatilityRegimeTracker | None = None


def get_volatility_tracker() -> VolatilityRegimeTracker:
    global _singleton
    if _singleton is None:
        _singleton = VolatilityRegimeTracker()
    return _singleton
