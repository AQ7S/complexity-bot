"""Tests for the LightGBM wrapper (Tier 3.4)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from engine.models.lightgbm_model import LightGBMModel  # noqa: E402


def _separable_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = np.zeros(n, dtype=np.int64)
    y[X[:, 0] > 0.4] = 0   # BUY when feature 0 strongly positive
    y[X[:, 0] < -0.4] = 1  # SELL when feature 0 strongly negative
    y[(X[:, 0] >= -0.4) & (X[:, 0] <= 0.4)] = 2  # HOLD
    return X, y


def test_fit_returns_training_result():
    X, y = _separable_dataset()
    m = LightGBMModel({"n_estimators": 30, "learning_rate": 0.1})
    r = m.fit(X[:480], y[:480], X_val=X[480:], y_val=y[480:], early_stopping=10)
    assert r.best_iteration >= 1
    assert r.best_val_logloss >= 0.0


def test_predict_proba_sums_to_one():
    X, y = _separable_dataset()
    m = LightGBMModel({"n_estimators": 30})
    m.fit(X[:400], y[:400])
    probs = m.predict_proba(X[:5])
    assert probs.shape == (5, 3)
    for row in probs:
        assert row.sum() == pytest.approx(1.0, abs=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    X, y = _separable_dataset()
    m = LightGBMModel({"n_estimators": 20})
    m.fit(X[:400], y[:400])
    p = tmp_path / "lgbm.txt"
    m.save(p)
    m2 = LightGBMModel.load(p)
    preds1 = m.predict(X[:10])
    preds2 = m2.predict(X[:10])
    assert (preds1 == preds2).all()


def test_separable_dataset_above_random():
    X, y = _separable_dataset(n=1000)
    cut = 800
    m = LightGBMModel({"n_estimators": 80, "learning_rate": 0.1})
    m.fit(X[:cut], y[:cut], X_val=X[cut:], y_val=y[cut:])
    acc = float(np.mean(m.predict(X[cut:]) == y[cut:]))
    assert acc > 0.6, f"acc={acc} too low"
