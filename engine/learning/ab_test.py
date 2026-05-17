"""A/B test framework — sample-size calculator + two-arm comparison.

Before launching a new strategy or model variant, the operator should
know: "to detect a Sharpe uplift of X at Y% power with Z% significance,
how many trades do I need to observe?" The standard answer for a
one-sided test on the difference of means is:

    n_per_arm = 2 × (σ × (z_α + z_β) / δ)²

where σ is the per-trade pnl/R standard deviation, δ is the minimum
detectable effect (in the same units), z_α = inverse-normal at the
significance threshold, z_β = inverse-normal at the desired power.

When δ is expressed as a Sharpe-ratio uplift, σ = 1 (Sharpe is already
in units of standard deviations), simplifying the calculator.

Also provides `compare_arms()` — a two-sample test on observed PnL/R
streams returning (effect_size, p_value, decision) — for actually
running the comparison once the sample is collected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampleSizeEstimate:
    n_per_arm: int
    z_alpha: float
    z_beta: float
    sigma: float
    min_effect: float
    significance: float
    power: float


def _normal_ppf(p: float) -> float:
    """Beasley-Springer-Moro inverse normal CDF approximation."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -_normal_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    return float(t - num / den)


def required_sample_size(
    *,
    min_effect: float,
    sigma: float = 1.0,
    significance: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
) -> SampleSizeEstimate:
    """Compute n_per_arm to detect `min_effect` (units of σ if not given)."""
    if min_effect <= 0:
        raise ValueError("min_effect must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < significance < 1 or not 0 < power < 1:
        raise ValueError("significance and power must lie in (0, 1)")
    z_a = _normal_ppf(1.0 - significance) if one_sided else _normal_ppf(1.0 - significance / 2.0)
    z_b = _normal_ppf(power)
    n = 2.0 * ((sigma * (z_a + z_b)) / min_effect) ** 2
    return SampleSizeEstimate(
        n_per_arm=int(math.ceil(n)),
        z_alpha=z_a, z_beta=z_b,
        sigma=sigma, min_effect=min_effect,
        significance=significance, power=power,
    )


def required_for_sharpe(
    *,
    sharpe_uplift: float,
    sigma_returns: float = 1.0,
    periods_per_year: int = 252,
    significance: float = 0.05,
    power: float = 0.80,
) -> SampleSizeEstimate:
    """Sample-size needed to detect a Sharpe ratio uplift.

    Sharpe annualises by sqrt(periods_per_year); the per-period effect
    we need to detect is `sharpe_uplift / sqrt(periods_per_year) * σ_returns`.
    """
    per_period_mean = sharpe_uplift / math.sqrt(periods_per_year) * sigma_returns
    return required_sample_size(
        min_effect=per_period_mean, sigma=sigma_returns,
        significance=significance, power=power,
    )


@dataclass(frozen=True)
class ComparisonResult:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    effect: float
    t_statistic: float
    p_value: float
    significant: bool


def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-statistic + approximate two-sided p-value via normal."""
    mean_diff = float(a.mean() - b.mean())
    var_a = float(a.var(ddof=1)) if a.size > 1 else 0.0
    var_b = float(b.var(ddof=1)) if b.size > 1 else 0.0
    se = math.sqrt(var_a / max(a.size, 1) + var_b / max(b.size, 1))
    if se == 0:
        return 0.0, 1.0
    t = mean_diff / se
    # Two-sided normal approximation (df → ∞ in Welch).
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, max(0.0, min(1.0, p))


def compare_arms(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    *,
    significance: float = 0.05,
) -> ComparisonResult:
    """Welch two-sample test on observed pnl/R streams."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return ComparisonResult(
            n_a=int(arr_a.size), n_b=int(arr_b.size),
            mean_a=0.0, mean_b=0.0, effect=0.0,
            t_statistic=0.0, p_value=1.0, significant=False,
        )
    t, p = _welch_t(arr_a, arr_b)
    return ComparisonResult(
        n_a=int(arr_a.size), n_b=int(arr_b.size),
        mean_a=float(arr_a.mean()), mean_b=float(arr_b.mean()),
        effect=float(arr_a.mean() - arr_b.mean()),
        t_statistic=t, p_value=p,
        significant=(p < significance),
    )
