"""Tests for bootstrap CI utilities (Tier 1.5)."""
from __future__ import annotations

import numpy as np

from engine.learning.stats import (
    bootstrap_ci,
    format_ci,
    profit_factor_ci,
    sharpe_ci,
    win_rate_ci,
)


def test_bootstrap_basic_mean_contains_true_value():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=1.0, scale=0.5, size=200)
    ci = bootstrap_ci(values, n_resamples=500, seed=1)
    assert ci.lower < ci.estimate < ci.upper
    assert ci.n == 200
    # True mean 1.0 should fall in the CI most of the time.
    assert ci.lower < 1.0 < ci.upper


def test_bootstrap_empty_returns_zeros():
    ci = bootstrap_ci([])
    assert ci.n == 0
    assert ci.estimate == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_win_rate_ci_bounds():
    outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5
    ci = win_rate_ci(outcomes, n_resamples=300)
    assert 0.0 <= ci.lower <= ci.estimate <= ci.upper <= 1.0


def test_sharpe_ci_zero_for_zero_var():
    returns = [0.01] * 50
    ci = sharpe_ci(returns, n_resamples=200)
    assert ci.estimate == 0.0


def test_profit_factor_ci_handles_no_losses():
    pnls = [1.0, 2.0, 0.5, 3.0]
    ci = profit_factor_ci(pnls, n_resamples=100)
    assert ci.estimate == float("inf") or ci.estimate > 0


def test_format_ci_percent():
    ci = bootstrap_ci([0.5, 0.6, 0.55, 0.45, 0.5], n_resamples=100)
    s = format_ci(ci, percent=True, decimals=1)
    assert "%" in s
    assert "[" in s and "]" in s
