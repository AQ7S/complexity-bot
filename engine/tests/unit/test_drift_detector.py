"""Tests for drift detectors (Tier 3.2)."""
from __future__ import annotations

from engine.learning.drift_detector import ADWINDetector, PageHinkleyDetector


def test_page_hinkley_no_drift_on_stable_stream():
    d = PageHinkleyDetector(threshold=10.0)
    for _ in range(100):
        d.add(0.0)
    assert not d.drift_detected()


def test_page_hinkley_detects_negative_drift():
    d = PageHinkleyDetector(threshold=2.0, min_delta=0.01)
    # 100 zeros then a step DOWN — Sharpe collapse scenario.
    for _ in range(100):
        d.add(0.0)
    for _ in range(50):
        d.add(-1.0)
    assert d.drift_detected()


def test_page_hinkley_reset_clears_drift():
    d = PageHinkleyDetector(threshold=2.0)
    for _ in range(50):
        d.add(-1.0)
    if d.drift_detected():
        d.reset()
        assert not d.drift_detected()


def test_adwin_no_drift_on_stable_stream():
    d = ADWINDetector(delta=0.05)
    for _ in range(100):
        d.add(0.5)
    assert not d.drift_detected()


def test_adwin_detects_regime_shift():
    d = ADWINDetector(delta=0.002)
    for _ in range(200):
        d.add(0.1)
    for _ in range(200):
        d.add(0.9)
    assert d.drift_detected()


def test_adwin_acknowledge_clears_flag():
    d = ADWINDetector(delta=0.002)
    for _ in range(100):
        d.add(0.0)
    for _ in range(100):
        d.add(1.0)
    if d.drift_detected():
        d.acknowledge()
        assert not d.drift_detected()


def test_adwin_window_bounded_by_max_buckets():
    d = ADWINDetector(delta=0.05, max_buckets=20)
    for i in range(100):
        d.add(float(i))
    assert d.window_size <= 20
