"""Tests for the meta-labeler (Tier 1.3)."""
from __future__ import annotations

import numpy as np

from engine.models.meta_labeler import (
    MetaLabelerModel,
    apply_meta_label,
    train_meta_labeler,
)


def _make_separable_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Outcome correlated with first feature.
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y


def test_train_returns_model():
    X, y = _make_separable_dataset()
    m = train_meta_labeler(X, y, epochs=100)
    assert isinstance(m, MetaLabelerModel)
    assert m.weights.shape == (4,)


def test_predict_proba_in_unit_interval():
    X, y = _make_separable_dataset()
    m = train_meta_labeler(X, y, epochs=50)
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert (p >= 0).all() and (p <= 1).all()


def test_separable_data_better_than_random():
    X, y = _make_separable_dataset(n=500)
    m = train_meta_labeler(X, y, epochs=300, lr=0.1)
    preds = m.predict(X)
    acc = float(np.mean(preds == y))
    assert acc > 0.65


def test_apply_meta_label_passes_through_hold():
    X, _ = _make_separable_dataset(n=50)
    m = train_meta_labeler(X, np.ones(50, dtype=np.int64), epochs=10)
    final, p = apply_meta_label(m, primary_pred=2, features=X[0])
    assert final == 2
    assert p == 0.0


def test_apply_meta_label_blocks_below_threshold():
    # Train so all probas are low; predict feature far from training mean.
    X = np.random.default_rng(0).normal(size=(80, 3))
    y = np.zeros(80, dtype=np.int64)  # always 0
    m = train_meta_labeler(X, y, epochs=200, lr=0.1, threshold=0.5)
    final, p = apply_meta_label(m, primary_pred=0, features=X[0])
    assert final == 2  # blocked
    assert p < 0.5


def test_apply_meta_label_accepts_above_threshold():
    X = np.random.default_rng(1).normal(size=(80, 3))
    y = np.ones(80, dtype=np.int64)
    m = train_meta_labeler(X, y, epochs=200, lr=0.1, threshold=0.5)
    final, p = apply_meta_label(m, primary_pred=0, features=X[0])
    assert final == 0
    assert p >= 0.5
