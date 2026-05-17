"""Tests for the latency tracker (Tier 7.1)."""
from __future__ import annotations

import pytest

from engine.utils import latency


@pytest.fixture(autouse=True)
def _reset_latency():
    latency.reset()
    latency.configure(window=100)
    yield
    latency.reset()


def test_record_appends_sample():
    latency.record("step_a", 10.0)
    latency.record("step_a", 20.0)
    s = latency.step_summary("step_a")
    assert s["n"] == 2
    assert s["max"] == pytest.approx(20.0)


def test_percentiles_increase_with_load():
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]:
        latency.record("p", float(v))
    s = latency.step_summary("p")
    assert s["p50"] >= 5
    assert s["p95"] >= 9
    assert s["p99"] == 100


def test_unknown_step_returns_zero_for_empty():
    s = latency.step_summary("never_observed")
    assert s["n"] == 0
    assert s["p50"] == 0.0


def test_time_step_context_records_duration():
    with latency.time_step("ctx"):
        for _ in range(10000):
            _ = 1 + 1  # busywork
    s = latency.step_summary("ctx")
    assert s["n"] == 1
    assert s["p50"] >= 0  # non-negative wall-clock


def test_negative_duration_ignored():
    latency.record("neg", -5.0)
    assert latency.step_summary("neg")["n"] == 0


def test_window_caps_buffer():
    latency.configure(window=10)
    for v in range(50):
        latency.record("cap", float(v))
    assert latency.step_summary("cap")["n"] == 10


def test_snapshot_includes_all_known_steps():
    snap = latency.latency_snapshot()
    for step in latency.KNOWN_STEPS:
        assert step in snap
