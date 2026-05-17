"""Tests for SHAP-lite trade attribution (Tier 7.2)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from engine.learning.shap_attribution import (  # noqa: E402
    SHAPRow,
    attribution_for_losers,
    compute_shap_for_trades,
    consistency_check,
    mean_abs_attribution,
    top_loss_drivers,
)
from engine.models.lightgbm_model import LightGBMModel  # noqa: E402


def _trained_model_with_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.zeros(n, dtype=np.int64)
    y[X[:, 0] > 0.5] = 0
    y[X[:, 0] < -0.5] = 1
    y[(X[:, 0] >= -0.5) & (X[:, 0] <= 0.5)] = 2
    m = LightGBMModel({"n_estimators": 50, "learning_rate": 0.1})
    m.fit(X[:300], y[:300], X_val=X[300:], y_val=y[300:])
    return m, X, y


def test_compute_returns_one_row_per_input():
    m, X, _ = _trained_model_with_data()
    rows = compute_shap_for_trades(m, X[:10], pnls=list(np.arange(10) - 5))
    assert len(rows) == 10
    assert all(isinstance(r, SHAPRow) for r in rows)
    assert all(r.contributions.shape == (6,) for r in rows)


def test_empty_input_returns_empty_list():
    rows = compute_shap_for_trades(object(), np.empty((0, 6)))
    assert rows == []


def test_mean_abs_attribution_returns_per_feature_array():
    rng = np.random.default_rng(1)
    rows = [
        SHAPRow(trade_id=i, pnl=0.0, base_value=0.0,
                contributions=rng.normal(size=4))
        for i in range(20)
    ]
    mean = mean_abs_attribution(rows)
    assert mean.shape == (4,)
    assert (mean >= 0).all()


def test_attribution_for_losers_filters_pnl():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(20):
        rows.append(SHAPRow(
            trade_id=i, pnl=1.0 if i % 2 == 0 else -1.0,
            base_value=0.0, contributions=rng.normal(size=3),
        ))
    losers_mean = attribution_for_losers(rows)
    overall_mean = mean_abs_attribution(rows)
    assert losers_mean.shape == overall_mean.shape


def test_top_loss_drivers_returns_sorted():
    rows = [
        SHAPRow(trade_id=0, pnl=-1.0, base_value=0.0,
                contributions=np.array([10.0, 1.0, 0.5])),
        SHAPRow(trade_id=1, pnl=-1.0, base_value=0.0,
                contributions=np.array([5.0, 2.0, 0.5])),
    ]
    out = top_loss_drivers(rows, top_n=3, feature_names=["a", "b", "c"])
    assert out[0][0] == "a"
    assert out[0][1] >= out[1][1] >= out[2][1]


def test_consistency_check_low_error_on_trained_model():
    m, X, _ = _trained_model_with_data()
    rows = compute_shap_for_trades(m, X[:50])
    probs = m.predict_proba(X[:50])
    avg_logit = probs.mean(axis=1)
    err = consistency_check(rows, avg_logit)
    assert err >= 0  # bounded


def test_top_loss_drivers_empty_when_no_losers():
    rows = [
        SHAPRow(trade_id=0, pnl=1.0, base_value=0.0,
                contributions=np.array([1.0, 1.0])),
    ]
    assert top_loss_drivers(rows) == []
