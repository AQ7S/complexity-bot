"""Decision-path latency tracker.

CPU/memory telemetry covers process health. This module covers
*decision-path latency* — how long each step of the signal → consensus
→ Claude → order pipeline takes. The dashboard panel surfaces P50/P95/P99
over a rolling 100-signal window so the operator can see degradation
before it manifests as missed entries.

Steps tracked (canonical):

  * signal_generation_ms  — features pipeline + per-source vote computation
  * consensus_ms          — `consensus.evaluate()` total
  * claude_gate_ms        — Anthropic API round-trip + JSON parse
  * order_send_ms         — `mt5.order_send()` round-trip (or shadow record)

Usage:

    from engine.utils.latency import time_step, latency_snapshot

    with time_step("signal_generation_ms"):
        do_features_and_votes()

    snap = latency_snapshot()  # → {step: {p50, p95, p99, n}}

Thread-safe: each step has its own deque + lock so concurrent writes
from the position monitor + main consensus loop won't corrupt state.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator


DEFAULT_WINDOW = 100
KNOWN_STEPS = (
    "signal_generation_ms",
    "consensus_ms",
    "claude_gate_ms",
    "order_send_ms",
)


class _StepBuffer:
    __slots__ = ("buf", "lock", "max_observed")

    def __init__(self, window: int) -> None:
        self.buf: deque[float] = deque(maxlen=window)
        self.lock = threading.Lock()
        self.max_observed: float = 0.0


_buffers: dict[str, _StepBuffer] = {}
_buffers_lock = threading.Lock()
_window = DEFAULT_WINDOW


def configure(*, window: int = DEFAULT_WINDOW) -> None:
    """Reset the rolling window size (drops existing samples)."""
    global _window, _buffers
    with _buffers_lock:
        _window = max(2, int(window))
        _buffers = {}


def _get_buffer(step: str) -> _StepBuffer:
    with _buffers_lock:
        b = _buffers.get(step)
        if b is None:
            b = _StepBuffer(_window)
            _buffers[step] = b
        return b


def record(step: str, duration_ms: float) -> None:
    """Add a single observation to a step's rolling window."""
    if duration_ms < 0 or not step:
        return
    b = _get_buffer(step)
    with b.lock:
        b.buf.append(float(duration_ms))
        if duration_ms > b.max_observed:
            b.max_observed = float(duration_ms)


@contextmanager
def time_step(step: str) -> Iterator[None]:
    """Context manager that records `step` duration on exit."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(step, (time.perf_counter() - t0) * 1000.0)


def _percentile(buf: list[float], p: float) -> float:
    if not buf:
        return 0.0
    s = sorted(buf)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def step_summary(step: str) -> dict[str, float | int]:
    b = _get_buffer(step)
    with b.lock:
        samples = list(b.buf)
        max_obs = b.max_observed
    return {
        "n": len(samples),
        "p50": _percentile(samples, 50),
        "p95": _percentile(samples, 95),
        "p99": _percentile(samples, 99),
        "max": max_obs,
    }


def latency_snapshot() -> dict[str, dict[str, float | int]]:
    """All known steps + any custom step ever recorded."""
    out: dict[str, dict[str, float | int]] = {}
    with _buffers_lock:
        names = list(set(KNOWN_STEPS) | set(_buffers.keys()))
    for step in sorted(names):
        out[step] = step_summary(step)
    return out


def reset() -> None:
    """Drop all samples; useful between test cases."""
    global _buffers
    with _buffers_lock:
        _buffers = {}
