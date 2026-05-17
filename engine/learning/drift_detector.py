"""Concept-drift detectors.

Drift = the statistical distribution feeding the model has changed enough
that yesterday's optimal parameters are today's losers. Retraining every
100 closed trades is a coarse heuristic; drift-triggered retraining is the
mature alternative.

Two complementary detectors:

  * Page-Hinkley (Page 1954). Sensitive to mean shifts in an online stream.
    Standard λ ≈ 50 with α ≈ 0.005 — alarm fires when the cumulative mean
    deviation from a tracked reference exceeds the threshold.

  * ADWIN — Adaptive Windowing (Bifet & Gavaldà 2007). Maintains a sliding
    window of variable size; shrinks when a statistically significant
    change is detected between any partition of the window. δ = 0.002 by
    default.

Both expose `add(value)` and `drift_detected() -> bool`. The engine's
retrain loop checks `drift_detected()` once per minute on the rolling
Sharpe of recent closed trades.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class PageHinkleyDetector:
    """Two-sided Page-Hinkley mean-shift detector.

    Accumulates the positive and negative deviation of each observation
    from the running sample mean. When either accumulator exceeds the
    threshold, a drift alarm fires.

    λ ("threshold") controls sensitivity — smaller = earlier alarm but
    more false positives. α ("min_delta") is a small slack that prevents
    drift accumulating from random noise alone.
    """

    threshold: float = 50.0
    min_delta: float = 0.005

    def __post_init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._cum_up: float = 0.0
        self._cum_dn: float = 0.0
        self._drift: bool = False

    def add(self, value: float) -> None:
        v = float(value)
        old_mean = self._mean
        self._n += 1
        # Incremental sample mean (Welford-style).
        self._mean = old_mean + (v - old_mean) / self._n
        # Two-sided cumulative deviation from the *prior* mean.
        self._cum_up = max(0.0, self._cum_up + (v - old_mean - self.min_delta))
        self._cum_dn = max(0.0, self._cum_dn + (old_mean - v - self.min_delta))
        if max(self._cum_up, self._cum_dn) > self.threshold:
            self._drift = True

    def drift_detected(self) -> bool:
        return self._drift

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._cum_up = 0.0
        self._cum_dn = 0.0
        self._drift = False

    @property
    def statistic(self) -> float:
        return max(self._cum_up, self._cum_dn)


@dataclass
class _ADWINBucket:
    total: float
    count: int
    variance: float


class ADWINDetector:
    """Adaptive Windowing change detector (Bifet & Gavaldà 2007).

    Maintains a deque of (mean, count) buckets compressed exponentially.
    Whenever a cut between two sub-windows shows a statistically
    significant mean difference (Hoeffding bound parametrised by δ), the
    left part is dropped and `drift_detected()` returns True.

    This is a simplified (single-window-shrink, no exponential bucket
    compression) implementation suitable for the engine's once-per-minute
    Sharpe-stream use. Memory is bounded by the natural window length.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 1024) -> None:
        self.delta = float(delta)
        self.max_buckets = int(max_buckets)
        self._buckets: deque[_ADWINBucket] = deque()
        self._total: float = 0.0
        self._count: int = 0
        self._drift: bool = False
        self._last_change_at: int = 0

    def add(self, value: float) -> None:
        v = float(value)
        self._buckets.append(_ADWINBucket(total=v, count=1, variance=0.0))
        self._total += v
        self._count += 1
        while len(self._buckets) > self.max_buckets:
            old = self._buckets.popleft()
            self._total -= old.total
            self._count -= old.count
        self._try_shrink()

    def _try_shrink(self) -> None:
        if self._count < 8:
            return
        # Walk a candidate cut point left→right; shrink window when first
        # significant change found.
        left_total = 0.0
        left_count = 0
        for i in range(len(self._buckets) - 1):
            left_total += self._buckets[i].total
            left_count += self._buckets[i].count
            right_total = self._total - left_total
            right_count = self._count - left_count
            if left_count == 0 or right_count == 0:
                continue
            mu_l = left_total / left_count
            mu_r = right_total / right_count
            # Hoeffding bound: ε = sqrt( (1/(2m)) * ln(4/δ_prime) ) with
            # δ_prime = δ / (cuts considered). Common simplification:
            m = 1.0 / (1.0 / left_count + 1.0 / right_count)
            delta_prime = max(self.delta / max(self._count, 1), 1e-12)
            eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))
            if abs(mu_l - mu_r) > eps:
                # Cut: drop the left buckets, signal drift.
                for _ in range(i + 1):
                    old = self._buckets.popleft()
                    self._total -= old.total
                    self._count -= old.count
                self._drift = True
                self._last_change_at += 1
                return

    def drift_detected(self) -> bool:
        return self._drift

    def acknowledge(self) -> None:
        """Reset the drift flag after the caller has handled the alarm."""
        self._drift = False

    def reset(self) -> None:
        self._buckets.clear()
        self._total = 0.0
        self._count = 0
        self._drift = False
        self._last_change_at = 0

    @property
    def window_size(self) -> int:
        return self._count
