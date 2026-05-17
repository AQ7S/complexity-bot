"""Tests for purged + embargoed walk-forward CV (Tier 1.2)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.learning.purged_cv import (
    aggregate_folds,
    purged_kfold_indices,
    walk_forward_eval,
    FoldResult,
)


def test_basic_kfold_covers_all_samples_in_test():
    n = 100
    test_total = set()
    folds_seen = 0
    for k, _, test_idx in purged_kfold_indices(n, n_splits=5, embargo_pct=0.0):
        folds_seen += 1
        test_total.update(int(i) for i in test_idx)
    assert folds_seen == 5
    assert test_total == set(range(n))


def test_no_train_test_overlap():
    n = 200
    for k, train_idx, test_idx in purged_kfold_indices(n, n_splits=4, embargo_pct=0.0):
        assert len(set(train_idx).intersection(test_idx)) == 0


def test_purge_drops_overlapping_horizons():
    n = 100
    # Sample i has label horizon i+5 — so the last 5 samples before each test
    # fold should be purged.
    horizons = np.arange(n) + 5
    for k, train_idx, test_idx in purged_kfold_indices(
        n, n_splits=4, label_horizons=horizons, embargo_pct=0.0,
    ):
        test_start = int(test_idx[0])
        # Indices with horizon falling in test range must not be in train.
        for ti in train_idx:
            assert not (test_start <= horizons[ti] <= int(test_idx[-1]))


def test_embargo_widens_drop_region():
    n = 100
    no_embargo = []
    with_embargo = []
    for k, train_idx, test_idx in purged_kfold_indices(n, n_splits=4, embargo_pct=0.0):
        no_embargo.append(len(train_idx))
    for k, train_idx, test_idx in purged_kfold_indices(n, n_splits=4, embargo_pct=0.05):
        with_embargo.append(len(train_idx))
    # Embargo should never increase train size, and on at least some folds reduce it.
    assert all(a <= b for a, b in zip(with_embargo, no_embargo))
    assert any(a < b for a, b in zip(with_embargo, no_embargo))


def test_edge_folds_uneven_split():
    n = 23  # not evenly divisible by 5
    folds = list(purged_kfold_indices(n, n_splits=5, embargo_pct=0.0))
    assert len(folds) == 5
    total = sum(len(f[2]) for f in folds)
    assert total == n


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        list(purged_kfold_indices(3, n_splits=5))
    with pytest.raises(ValueError):
        list(purged_kfold_indices(100, n_splits=5, embargo_pct=0.6))


def test_walk_forward_eval_runs_dummy_model():
    class Const:
        def fit(self, X, y):
            self.c = int(np.bincount(y).argmax())
        def predict(self, X):
            return np.full(len(X), self.c, dtype=np.int64)

    n = 100
    X = np.random.default_rng(0).normal(size=(n, 5))
    y = np.random.default_rng(0).integers(0, 2, size=n)
    results = walk_forward_eval(X, y, model_factory=Const, n_splits=4, embargo_pct=0.01)
    assert len(results) == 4
    for r in results:
        assert isinstance(r, FoldResult)
        assert 0.0 <= r.score <= 1.0

    agg = aggregate_folds(results)
    assert agg["n_folds"] == 4
    assert agg["min"] <= agg["mean"] <= agg["max"]
