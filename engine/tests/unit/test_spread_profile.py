"""Tests for per-hour spread profile (Tier 2.2)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engine.risk.spread_profile import (
    build_hourly_profile,
    expected_spread,
    is_spread_acceptable,
)


def _make_samples(hour: int, spread: float, n: int) -> list[dict]:
    base = datetime(2024, 1, 1, hour, 0, tzinfo=timezone.utc)
    return [{"ts": base + timedelta(minutes=i), "spread": spread} for i in range(n)]


def test_build_profile_basic_hour():
    samples = _make_samples(3, 1.0, 50) + _make_samples(10, 0.5, 50)
    p = build_hourly_profile(samples, symbol="EURUSD")
    assert expected_spread(p, 3) == pytest.approx(1.0)
    assert expected_spread(p, 10) == pytest.approx(0.5)


def test_acceptable_within_multiplier():
    samples = _make_samples(10, 0.5, 50)
    p = build_hourly_profile(samples, symbol="EURUSD")
    assert is_spread_acceptable(p, 0.6, utc_hour=10, multiplier=1.5)


def test_rejected_above_multiplier():
    samples = _make_samples(10, 0.5, 50)
    p = build_hourly_profile(samples, symbol="EURUSD")
    assert not is_spread_acceptable(p, 1.0, utc_hour=10, multiplier=1.5)


def test_hour_with_too_few_samples_uses_fallback():
    # Only 3 ticks at hour 3 → falls back to overall median.
    samples = _make_samples(3, 100.0, 3) + _make_samples(10, 1.0, 50)
    p = build_hourly_profile(samples, symbol="EURUSD")
    # hour-3 has insufficient data → fallback (overall median across 53 ≈ 1.0)
    assert p.fallback_median == pytest.approx(1.0)
    assert p.median_at(3) == pytest.approx(1.0)


def test_zero_median_is_permissive():
    p = build_hourly_profile([], symbol="EURUSD")
    assert is_spread_acceptable(p, 100.0, utc_hour=5)


def test_invalid_hour_returns_fallback():
    samples = _make_samples(10, 0.5, 50)
    p = build_hourly_profile(samples, symbol="EURUSD")
    # `median_at(25)` should fall back; treat as permissive.
    assert is_spread_acceptable(p, 0.5, utc_hour=25)
