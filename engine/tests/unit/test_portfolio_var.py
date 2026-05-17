"""Tests for portfolio VaR/CVaR (Tier 4.2)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.risk.portfolio_var import (
    Position,
    VAR_CAP_DEFAULT,
    historical_var,
    parametric_var,
    var_breach_predictor,
)


def _make_normal_returns(n: int = 500, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, n)


def test_historical_var_zero_on_empty_positions():
    r = historical_var({"EURUSD#": _make_normal_returns()}, [], 10_000)
    assert r.var_pct == 0.0


def test_historical_var_positive_with_normal_returns():
    returns = {"EURUSD#": _make_normal_returns()}
    positions = [Position("EURUSD#", "BUY", 10_000)]
    r = historical_var(returns, positions, 10_000, confidence=0.95)
    assert r.var_pct > 0.0
    assert r.var_usd == pytest.approx(r.var_pct * 10_000, abs=1e-6)
    assert r.cvar_pct >= r.var_pct  # CVaR ≥ VaR by definition


def test_parametric_var_matches_normal_approx():
    rng = np.random.default_rng(1)
    returns = {"EURUSD#": rng.normal(0.0, 0.01, 1000)}
    positions = [Position("EURUSD#", "BUY", 10_000)]
    hist = historical_var(returns, positions, 10_000, confidence=0.95)
    para = parametric_var(returns, positions, 10_000, confidence=0.95)
    # For a clean Gaussian sample, the two methods should agree within ~30%.
    if hist.var_pct > 0:
        assert para.var_pct / hist.var_pct < 1.5
        assert hist.var_pct / para.var_pct < 1.5


def test_short_position_inverts_weight():
    rng = np.random.default_rng(2)
    returns = {"EURUSD#": rng.normal(0.0, 0.01, 500)}
    long_pos = [Position("EURUSD#", "BUY", 10_000)]
    short_pos = [Position("EURUSD#", "SELL", 10_000)]
    r_long = historical_var(returns, long_pos, 10_000, confidence=0.95)
    r_short = historical_var(returns, short_pos, 10_000, confidence=0.95)
    # Both should have similar magnitudes on a symmetric distribution.
    assert abs(r_long.var_pct - r_short.var_pct) < 0.01


def test_var_breach_predictor_flags_oversized_addition():
    returns = {"EURUSD#": _make_normal_returns(scale=0.05)}  # very volatile
    existing = []
    new_pos = Position("EURUSD#", "BUY", 50_000)
    breach, _r = var_breach_predictor(new_pos, existing, returns, 10_000, var_cap=0.01)
    assert breach


def test_var_breach_predictor_passes_small_addition():
    returns = {"EURUSD#": _make_normal_returns(scale=0.001)}
    existing = []
    new_pos = Position("EURUSD#", "BUY", 1_000)
    breach, _r = var_breach_predictor(new_pos, existing, returns, 10_000, var_cap=VAR_CAP_DEFAULT)
    assert not breach


def test_var_zero_when_no_overlap_symbols():
    r = historical_var(
        {"GBPUSD#": _make_normal_returns()},
        [Position("XAUUSD#", "BUY", 5_000)],
        equity=10_000,
    )
    assert r.var_pct == 0.0
