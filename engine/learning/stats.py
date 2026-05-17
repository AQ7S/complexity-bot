"""Statistical utilities for honest reporting.

Every performance metric (win rate, Sharpe, profit factor, mean PnL) is
reported with a 95% bootstrap confidence interval. A single point estimate
on 50 trades is statistically indistinguishable from noise; the CI puts a
believability range on the number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.estimate, self.lower, self.upper


def bootstrap_ci(
    values: np.ndarray | list[float],
    statistic_fn: Callable[[np.ndarray], float] | None = None,
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int | None = 1337,
) -> BootstrapCI:
    """Percentile-method bootstrap CI for `statistic_fn` on `values`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return BootstrapCI(0.0, 0.0, 0.0, 0)
    if statistic_fn is None:
        statistic_fn = lambda x: float(np.mean(x))  # noqa: E731
    rng = np.random.default_rng(seed)
    n = arr.size
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic_fn(arr[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapCI(
        estimate=float(statistic_fn(arr)),
        lower=lo, upper=hi, n=n,
    )


def win_rate_ci(outcomes: np.ndarray | list[int], **kwargs) -> BootstrapCI:
    return bootstrap_ci(outcomes, statistic_fn=lambda x: float(np.mean(x)), **kwargs)


def sharpe_ci(returns: np.ndarray | list[float], *, periods_per_year: int = 252, **kwargs) -> BootstrapCI:
    sqrt_n = float(np.sqrt(periods_per_year))

    def _sharpe(x: np.ndarray) -> float:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        if sd < 1e-12:
            return 0.0
        return float(np.mean(x)) / sd * sqrt_n

    return bootstrap_ci(returns, statistic_fn=_sharpe, **kwargs)


def profit_factor_ci(pnls: np.ndarray | list[float], **kwargs) -> BootstrapCI:
    def _pf(x: np.ndarray) -> float:
        gains = float(x[x > 0].sum())
        losses = float(-x[x < 0].sum())
        if losses < 1e-12:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    return bootstrap_ci(pnls, statistic_fn=_pf, **kwargs)


def format_ci(ci: BootstrapCI, *, percent: bool = False, decimals: int = 2) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{ci.estimate * scale:.{decimals}f}{suffix} "
        f"[{ci.lower * scale:.{decimals}f}, {ci.upper * scale:.{decimals}f}]"
    )
