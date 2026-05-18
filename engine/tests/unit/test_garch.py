"""Tests for GARCH(1,1) (Tier 8.5)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.features.garch import (
    fit_garch_11,
    forecast_volatility,
    vol_target_lot_multiplier,
)


def _simulate_garch(n: int, omega: float, alpha: float, beta: float,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    var = omega / max(1.0 - alpha - beta, 1e-9)
    for t in range(n):
        sigma = np.sqrt(var)
        r[t] = rng.normal(0, sigma)
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


def test_fit_returns_valid_parameters():
    r = _simulate_garch(500, 0.00001, 0.08, 0.88)
    p = fit_garch_11(r)
    assert p.omega > 0
    assert 0 <= p.alpha < 1
    assert 0 <= p.beta < 1
    assert p.persistence < 1.0


def test_fit_short_input_returns_safe_defaults():
    p = fit_garch_11(np.array([0.001, -0.002, 0.001]))
    assert p.alpha + p.beta < 1


def test_forecast_horizon_one():
    r = _simulate_garch(400, 0.00001, 0.08, 0.88)
    p = fit_garch_11(r)
    f = forecast_volatility(r, p, horizon=1)
    assert f.sigma_next > 0
    assert f.horizon == 1


def test_forecast_long_horizon_reverts_to_unconditional():
    r = _simulate_garch(400, 0.00001, 0.08, 0.88)
    p = fit_garch_11(r)
    f_far = forecast_volatility(r, p, horizon=200)
    # With α+β < 1 the conditional variance reverts to ω/(1-α-β).
    assert f_far.sigma_horizon == pytest.approx(np.sqrt(p.unconditional_var),
                                                 rel=0.5)


def test_vol_target_multiplier_clamped():
    from engine.features.garch import GarchForecast
    high_vol = GarchForecast(sigma_next=10.0, sigma_horizon=10.0,
                              horizon=1, last_sigma=10.0, last_return=0.0)
    low_vol = GarchForecast(sigma_next=0.001, sigma_horizon=0.001,
                             horizon=1, last_sigma=0.001, last_return=0.0)
    assert vol_target_lot_multiplier(high_vol, target_vol=0.01) == pytest.approx(0.25)
    assert vol_target_lot_multiplier(low_vol, target_vol=0.01) == pytest.approx(4.0)


def test_vol_target_zero_sigma_returns_one():
    from engine.features.garch import GarchForecast
    f = GarchForecast(sigma_next=0.0, sigma_horizon=0.0, horizon=1,
                      last_sigma=0.0, last_return=0.0)
    assert vol_target_lot_multiplier(f, target_vol=0.01) == 1.0
