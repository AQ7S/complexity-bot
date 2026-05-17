"""Tests for the A/B test framework (Tier 7.5)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.learning.ab_test import (
    compare_arms,
    required_for_sharpe,
    required_sample_size,
)


def test_smaller_effect_requires_more_samples():
    big = required_sample_size(min_effect=1.0, sigma=1.0)
    small = required_sample_size(min_effect=0.1, sigma=1.0)
    assert small.n_per_arm > big.n_per_arm


def test_higher_power_requires_more_samples():
    p80 = required_sample_size(min_effect=0.5, sigma=1.0, power=0.80)
    p95 = required_sample_size(min_effect=0.5, sigma=1.0, power=0.95)
    assert p95.n_per_arm >= p80.n_per_arm


def test_required_for_sharpe_basic():
    est = required_for_sharpe(sharpe_uplift=0.5, sigma_returns=0.01, periods_per_year=252)
    assert est.n_per_arm > 0


def test_invalid_effect_raises():
    with pytest.raises(ValueError):
        required_sample_size(min_effect=0)
    with pytest.raises(ValueError):
        required_sample_size(min_effect=1.0, significance=1.5)


def test_compare_arms_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.02, 0.01, 500)
    b = rng.normal(0.0,  0.01, 500)
    res = compare_arms(a, b, significance=0.05)
    assert res.significant
    assert res.p_value < 0.05
    assert res.effect > 0


def test_compare_arms_no_difference_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.01, 200)
    b = rng.normal(0.0, 0.01, 200)
    res = compare_arms(a, b, significance=0.05)
    assert not res.significant


def test_compare_arms_empty_inputs_safe():
    res = compare_arms([], [1.0, 2.0])
    assert res.n_a == 0
    assert not res.significant


def test_one_sided_yields_smaller_n_than_two_sided_via_z_alpha():
    one = required_sample_size(min_effect=0.5, sigma=1.0, one_sided=True)
    two = required_sample_size(min_effect=0.5, sigma=1.0, one_sided=False)
    assert one.z_alpha < two.z_alpha
    assert one.n_per_arm <= two.n_per_arm
