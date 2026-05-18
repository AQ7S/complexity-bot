"""Tests for OOD anomaly gate (Tier 8.7)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.learning.anomaly_gate import (
    fit_isolation_forest_gate,
    fit_mahalanobis_gate,
)


def test_mahalanobis_in_distribution_passes():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 5))
    gate = fit_mahalanobis_gate(X, quantile_threshold=0.99)
    in_dist = rng.normal(0, 1, size=5)
    assert not gate.is_anomalous(in_dist)


def test_mahalanobis_out_of_distribution_flagged():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(500, 5))
    gate = fit_mahalanobis_gate(X, quantile_threshold=0.99)
    way_out = np.array([20.0] * 5)
    assert gate.is_anomalous(way_out)


def test_mahalanobis_insufficient_data_safe():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    gate = fit_mahalanobis_gate(X)
    assert gate.method.startswith("mahalanobis")
    # threshold is inf → nothing is anomalous
    assert not gate.is_anomalous(np.array([100.0, 100.0]))


def test_isolation_forest_falls_back_when_sklearn_absent():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(500, 4))
    # Whether sklearn is installed or not, gate must be valid.
    gate = fit_isolation_forest_gate(X)
    sample = rng.normal(0, 1, size=4)
    _ = gate.score(sample)


def test_isolation_forest_flags_far_point():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(500, 4))
    gate = fit_isolation_forest_gate(X, contamination=0.01)
    way_out = np.array([50.0, -50.0, 50.0, -50.0])
    in_dist = rng.normal(0, 1, size=4)
    assert gate.score(way_out) > gate.score(in_dist)


def test_score_monotone_with_distance():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, size=(500, 3))
    gate = fit_mahalanobis_gate(X)
    near = np.array([0.1, 0.1, 0.1])
    far = np.array([10.0, 10.0, 10.0])
    assert gate.score(far) > gate.score(near)
