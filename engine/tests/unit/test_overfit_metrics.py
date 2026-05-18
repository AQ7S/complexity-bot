"""Tests for PBO + Deflated Sharpe + CSCV (Tier 8.1)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.learning.overfit_metrics import (
    DeflatedSharpeResult,
    PBOResult,
    cscv_partitions,
    compute_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    is_overfit,
    passes_deflated_sharpe,
)


def test_cscv_partitions_split_sizes():
    parts = cscv_partitions(n_periods=16, s=8)
    for is_idx, oos_idx in parts:
        assert is_idx.size == oos_idx.size
        assert np.intersect1d(is_idx, oos_idx).size == 0


def test_cscv_partitions_count_matches_binomial():
    parts = cscv_partitions(n_periods=12, s=6)
    from math import comb
    assert len(parts) == comb(6, 3)


def test_cscv_rejects_odd_s():
    with pytest.raises(ValueError):
        cscv_partitions(n_periods=10, s=5)


def test_pbo_around_half_for_noise_only_matrix():
    rng = np.random.default_rng(7)
    R = rng.normal(0.0, 1.0, size=(64, 20))
    res = compute_pbo(R, s=8, rank_metric="sharpe")
    assert 0.20 <= res.pbo <= 0.80


def test_pbo_low_when_winner_dominates():
    rng = np.random.default_rng(11)
    R = rng.normal(0.0, 1.0, size=(64, 10))
    R[:, 0] += 1.0
    res = compute_pbo(R, s=8)
    assert res.pbo < 0.40


def test_expected_max_sharpe_grows_with_trials():
    e1 = expected_max_sharpe(1)
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    assert e1 < e10 < e100


def test_deflated_sharpe_lower_than_raw_under_selection():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.1, size=252)
    res = deflated_sharpe_ratio(r, n_trials=50, sharpe_trial_std=0.5)
    assert res.observed_sharpe > 0
    assert res.deflated_sharpe <= res.observed_sharpe + 1e-9


def test_is_overfit_threshold():
    high = PBOResult(pbo=0.6, n_trials=10, n_partitions=4,
                     logits=[], best_is_strategy_indices=[])
    low = PBOResult(pbo=0.1, n_trials=10, n_partitions=4,
                    logits=[], best_is_strategy_indices=[])
    assert is_overfit(high)
    assert not is_overfit(low)


def test_passes_deflated_sharpe_rejects_low_pvalue():
    bad = DeflatedSharpeResult(observed_sharpe=0.1, deflated_sharpe=-0.4,
                                p_value=0.9, n_trials=50, n_observations=252)
    good = DeflatedSharpeResult(observed_sharpe=2.0, deflated_sharpe=1.5,
                                 p_value=0.01, n_trials=50, n_observations=252)
    assert not passes_deflated_sharpe(bad)
    assert passes_deflated_sharpe(good)


def test_deflated_sharpe_short_returns_safe():
    r = np.array([0.01, -0.005, 0.002])
    res = deflated_sharpe_ratio(r, n_trials=10)
    assert res.p_value == 1.0
