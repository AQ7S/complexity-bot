"""Tests for the risk-of-ruin calculator (Tier 7.6)."""
from __future__ import annotations

import pytest

from engine.learning.risk_of_ruin import (
    analytic_ruin,
    monte_carlo_ruin,
    update_with_trades,
)


def test_zero_expectancy_ruin_one():
    e = analytic_ruin(win_rate=0.5, rr_ratio=1.0, risk_per_trade_pct=0.02, max_drawdown_pct=0.20)
    assert e.p_ruin == pytest.approx(1.0)


def test_negative_expectancy_ruin_one():
    e = analytic_ruin(win_rate=0.30, rr_ratio=1.0, risk_per_trade_pct=0.02, max_drawdown_pct=0.20)
    assert e.p_ruin == pytest.approx(1.0)


def test_strong_positive_edge_low_ruin():
    e = analytic_ruin(win_rate=0.60, rr_ratio=2.0, risk_per_trade_pct=0.02, max_drawdown_pct=0.20)
    assert e.p_ruin < 0.10


def test_more_units_lower_ruin():
    high_risk = analytic_ruin(win_rate=0.55, rr_ratio=1.5, risk_per_trade_pct=0.05, max_drawdown_pct=0.20)
    low_risk  = analytic_ruin(win_rate=0.55, rr_ratio=1.5, risk_per_trade_pct=0.01, max_drawdown_pct=0.20)
    assert low_risk.p_ruin <= high_risk.p_ruin


def test_monte_carlo_runs_and_bounds_probability():
    e = monte_carlo_ruin(
        win_rate=0.55, rr_ratio=1.5,
        risk_per_trade_pct=0.02, max_drawdown_pct=0.20,
        horizon_trades=300, n_paths=500,
    )
    assert 0.0 <= e.p_ruin <= 1.0
    assert e.n_paths == 500


def test_monte_carlo_zero_winrate_high_ruin():
    e = monte_carlo_ruin(
        win_rate=0.20, rr_ratio=1.0,
        risk_per_trade_pct=0.05, max_drawdown_pct=0.10,
        horizon_trades=50, n_paths=200,
    )
    assert e.p_ruin > 0.5


def test_bayesian_update_uses_observed_trades():
    e = update_with_trades(
        wins=80, losses=20, rr_ratio=1.5,
        risk_per_trade_pct=0.02, max_drawdown_pct=0.20,
    )
    assert e.p_ruin < 0.05  # 80% empirical WR + 1.5R should be safe


def test_unsupported_method_falls_back_to_analytic():
    e = update_with_trades(
        wins=50, losses=50, rr_ratio=1.5,
        risk_per_trade_pct=0.02, max_drawdown_pct=0.20,
        method="unknown_method",
    )
    assert e.method == "analytic"
