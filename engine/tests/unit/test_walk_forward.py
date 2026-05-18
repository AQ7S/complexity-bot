"""Tests for walk-forward optimization (Tier 8.6)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.learning.walk_forward import (
    anchored_walk_forward_splits,
    grid_search,
    rolling_walk_forward_splits,
    walk_forward_optimize,
)


def test_anchored_splits_grow_in_sample():
    splits = anchored_walk_forward_splits(n_periods=1000, is_min=200, oos_size=100)
    assert len(splits) == 8
    assert splits[0].in_sample_start == 0
    assert splits[0].in_sample_end == 200
    assert splits[-1].in_sample_end > splits[0].in_sample_end


def test_rolling_splits_fixed_in_sample():
    splits = rolling_walk_forward_splits(n_periods=1000, is_size=200, oos_size=100)
    for s in splits:
        assert s.in_sample_end - s.in_sample_start == 200
        assert s.out_of_sample_end - s.out_of_sample_start == 100


def test_splits_empty_when_too_few_periods():
    assert anchored_walk_forward_splits(50, is_min=200, oos_size=100) == []
    assert rolling_walk_forward_splits(50, is_size=200, oos_size=100) == []


def test_grid_search_picks_best_known_config():
    # Synthetic: best params known to be (a=2, b="x"); others are noise.
    def is_fn(params):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 100)
        if params["a"] == 2 and params["b"] == "x":
            base += 0.5
        return base
    best_params, best_score = grid_search(
        {"a": [1, 2, 3], "b": ["x", "y"]},
        is_returns_fn=is_fn,
    )
    assert best_params == {"a": 2, "b": "x"}
    assert best_score > 0


def test_walk_forward_optimize_end_to_end():
    rng = np.random.default_rng(0)
    full = rng.normal(0, 1, 600)
    def returns_for_window(params, start, end):
        seg = full[start:end].copy()
        if params["mult"] == 1.5:
            seg += 0.05
        return seg
    splits = rolling_walk_forward_splits(600, is_size=200, oos_size=100)
    res = walk_forward_optimize(
        splits, {"mult": [0.5, 1.0, 1.5]},
        returns_for_window_fn=returns_for_window,
    )
    assert len(res.out_of_sample_metrics) == len(splits)
    assert all(p == {"mult": 1.5} for p in res.best_params_per_window)


def test_walk_forward_empty_splits_safe():
    res = walk_forward_optimize([], {"x": [1]},
                                  returns_for_window_fn=lambda *a: np.zeros(1))
    assert res.mean_oos_metric == 0.0
