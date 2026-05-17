"""Correlation breakdown alarm.

Normally-correlated symbol pairs (EURUSD ↔ GBPUSD: ~+0.80; XAUUSD ↔
USDJPY: ~−0.30) tell you the market is in its usual regime. When the
correlation suddenly inverts or collapses, it almost always precedes a
risk event — central bank intervention, geopolitical shock, or a
liquidity flight.

We maintain a per-pair 30-day rolling Pearson correlation. The alarm
fires when:
    z = (current_window − long_baseline) / std_of_correlation
    |z| > 3.0
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


WINDOW_SHORT = 30      # bars in the "current" correlation window
WINDOW_LONG = 250      # baseline lookback for z-score
Z_ALARM = 3.0


@dataclass(frozen=True)
class CorrelationAlarm:
    pair: tuple[str, str]
    z_score: float
    current_correlation: float
    baseline_correlation: float
    baseline_std: float


class CorrelationMonitor:
    """Rolling correlation tracker for pre-declared symbol pairs.

    Returns alarms only when |z| > Z_ALARM AND the baseline std is non-trivial
    (so a stable-but-noisy pair doesn't trigger).
    """

    def __init__(self, pairs: list[tuple[str, str]] | None = None) -> None:
        self.pairs: list[tuple[str, str]] = list(pairs or [])
        self._returns: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW_LONG + WINDOW_SHORT))
        self._raised: set[tuple[str, str]] = set()

    def add_pair(self, a: str, b: str) -> None:
        if (a, b) not in self.pairs and (b, a) not in self.pairs:
            self.pairs.append((a, b))

    def add_returns(self, returns_by_symbol: dict[str, float]) -> None:
        for sym, r in returns_by_symbol.items():
            self._returns[sym].append(float(r))

    def _series(self, symbol: str) -> np.ndarray | None:
        if symbol not in self._returns:
            return None
        arr = np.array(self._returns[symbol], dtype=np.float64)
        return arr if arr.size else None

    def _rolling_corr(self, a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        n = min(len(a), len(b))
        if n < window + 1:
            return np.array([])
        a = a[-n:]; b = b[-n:]
        out = []
        for end in range(window, n + 1):
            sa = a[end - window:end]
            sb = b[end - window:end]
            if sa.std() < 1e-12 or sb.std() < 1e-12:
                out.append(0.0)
                continue
            out.append(float(np.corrcoef(sa, sb)[0, 1]))
        return np.array(out)

    def breakdown_alarms(self) -> list[CorrelationAlarm]:
        alarms: list[CorrelationAlarm] = []
        for a, b in self.pairs:
            sa = self._series(a)
            sb = self._series(b)
            if sa is None or sb is None:
                continue
            corrs = self._rolling_corr(sa, sb, WINDOW_SHORT)
            if corrs.size < 20:
                continue
            current = float(corrs[-1])
            baseline_window = corrs[:-1]
            mu = float(baseline_window.mean())
            sd = float(baseline_window.std(ddof=1))
            if sd < 0.02:
                continue
            z = (current - mu) / sd
            if abs(z) > Z_ALARM:
                alarms.append(CorrelationAlarm(
                    pair=(a, b),
                    z_score=z,
                    current_correlation=current,
                    baseline_correlation=mu,
                    baseline_std=sd,
                ))
        return alarms

    def acknowledge(self, pair: tuple[str, str]) -> None:
        self._raised.add(pair)
