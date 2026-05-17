"""Tests for VPIN (Tier 2.1)."""
from __future__ import annotations

import pytest

from engine.features.vpin import (
    bulk_volume_classify,
    build_volume_buckets,
    compute_vpin,
    vpin_gate,
    vpin_regime,
)


def _tick(bid: float, ask: float, vol: float = 1.0) -> dict:
    return {"bid": bid, "ask": ask, "volume": vol}


def test_bulk_volume_split_is_total_volume():
    buy, sell = bulk_volume_classify(open_price=1.0, close_price=1.0, total_volume=100.0, sigma=0.01)
    assert buy + sell == pytest.approx(100.0)


def test_bulk_volume_strong_up_classifies_mostly_buy():
    buy, sell = bulk_volume_classify(open_price=1.0, close_price=1.05, total_volume=100.0, sigma=0.01)
    assert buy > sell
    assert buy > 99.0


def test_bulk_volume_strong_down_classifies_mostly_sell():
    buy, sell = bulk_volume_classify(open_price=1.05, close_price=1.0, total_volume=100.0, sigma=0.01)
    assert sell > buy
    assert sell > 99.0


def test_build_volume_buckets_respects_target():
    ticks = [_tick(1.0, 1.001, 5.0) for _ in range(20)]
    buckets = build_volume_buckets(ticks, bucket_volume=10.0, sigma=0.001)
    # 100 total volume, 10 per bucket → ~10 buckets.
    assert 9 <= len(buckets) <= 11
    for b in buckets:
        assert b.total_volume == pytest.approx(10.0, abs=1e-6)


def test_compute_vpin_balanced_random_walk_below_one():
    # Symmetric random walk → buckets are mixed up + down → VPIN nowhere
    # near the toxic threshold and well below the one-sided case.
    import random
    rng = random.Random(0)
    ticks = []
    p = 1.0
    for _ in range(2000):
        p += rng.choice([-0.0001, 0.0001])
        ticks.append(_tick(p, p + 0.0001, 1.0))
    score = compute_vpin(ticks, bucket_volume=50.0, smooth_window=20)
    # Bound: any score in [0,1]; for a random walk on bucket_volume=50, expect well below the one-sided ceiling.
    assert 0.0 <= score <= 1.0
    # Reference: same flow shape but monotonic up gives near-1.0.
    monotonic = [_tick(1.0 + i * 0.0001, 1.0 + i * 0.0001 + 0.0001, 1.0) for i in range(2000)]
    ref = compute_vpin(monotonic, bucket_volume=50.0, smooth_window=20)
    assert score < ref


def test_compute_vpin_one_sided_flow_high_score():
    # Monotonic up-tick stream → almost all buys → high imbalance per bucket.
    ticks = []
    for i in range(200):
        p = 1.0 + i * 0.0001
        ticks.append(_tick(p, p + 0.0001, 1.0))
    score = compute_vpin(ticks, bucket_volume=10.0, smooth_window=10)
    assert score > 0.5


def test_vpin_gate_threshold():
    assert vpin_gate(0.3) is True
    assert vpin_gate(0.5) is False
    assert vpin_gate(0.39, threshold=0.4) is True
    assert vpin_gate(0.41, threshold=0.4) is False


def test_vpin_regime_labels():
    assert vpin_regime(0.05) == "BENIGN"
    assert vpin_regime(0.30) == "ELEVATED"
    assert vpin_regime(0.60) == "TOXIC"


def test_empty_ticks_returns_zero():
    assert compute_vpin([], bucket_volume=10.0) == 0.0
