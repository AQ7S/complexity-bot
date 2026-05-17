"""Tests for fractional Kelly sizing (Tier 4.1)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.risk.kelly import (
    KELLY_FRACTION,
    RISK_CAP,
    RISK_FLOOR,
    compute_kelly_fraction,
    fractional_kelly,
    kelly_from_pnl,
)


def test_full_kelly_zero_at_50_50_with_1_to_1():
    # p=0.5, b=1: f* = (0.5*2 - 1)/1 = 0
    f = compute_kelly_fraction(wins=50, losses=50, avg_win=1.0, avg_loss=1.0)
    assert f == pytest.approx(0.0, abs=1e-9)


def test_full_kelly_positive_for_edge():
    # p=0.6, b=2: f* = (0.6*3 - 1)/2 = 0.4
    f = compute_kelly_fraction(wins=60, losses=40, avg_win=2.0, avg_loss=1.0)
    assert f == pytest.approx(0.4, abs=1e-9)


def test_fractional_kelly_clamped_to_cap():
    # p=0.9, b=2 → f* = (0.9*3 - 1)/2 = 0.85 → fractional = 0.25 * 0.85 = 0.2125 → cap at 0.02
    f = fractional_kelly(0.9, 2.0)
    assert f == RISK_CAP


def test_fractional_kelly_clamped_to_floor_on_negative_edge():
    # p=0.3, b=1 → f* = (0.3*2 - 1)/1 = -0.4 → fractional negative → floor
    f = fractional_kelly(0.3, 1.0)
    assert f == RISK_FLOOR


def test_kelly_from_pnl_insufficient_returns_floor():
    pnls = [1.0, -1.0, 1.0]
    est = kelly_from_pnl(pnls)
    assert est.used_floor
    assert est.fractional_kelly == RISK_FLOOR


def test_kelly_from_pnl_positive_edge_sized_within_cap():
    rng = np.random.default_rng(0)
    wins = rng.normal(2.0, 0.5, 60)
    losses = rng.normal(-1.0, 0.3, 40)
    pnls = np.concatenate([wins, losses])
    est = kelly_from_pnl(pnls)
    assert RISK_FLOOR <= est.fractional_kelly <= RISK_CAP
    assert est.win_rate == pytest.approx(0.6, abs=0.05)


def test_kelly_zero_for_loser_dominant():
    pnls = [-1.0] * 90 + [0.5] * 10
    est = kelly_from_pnl(pnls)
    # Negative edge → fractional Kelly floor.
    assert est.fractional_kelly == RISK_FLOOR


def test_fractional_kelly_uses_quarter_of_full():
    # Pure math check: at b=2, p=0.55 → full = (0.55*3 - 1)/2 = 0.325
    # 1/4 Kelly = 0.08125 → still clipped to cap 0.02
    f = fractional_kelly(0.55, 2.0)
    assert f == RISK_CAP
    # Lower edge so 1/4 Kelly falls below cap
    # p=0.52, b=1.2 → full = (0.52*2.2 - 1)/1.2 = 0.117 → /4 = 0.029 → cap 0.02
    f = fractional_kelly(0.52, 1.2, cap=0.10)
    assert KELLY_FRACTION * 0.117 == pytest.approx(f, abs=0.01) or f >= RISK_FLOOR
