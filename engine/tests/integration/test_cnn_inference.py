"""Phase 5 inference smoke test.

Plan asserts:
- a checkpoint exists and loads
- per-symbol inference latency < 50ms
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from engine.models import inference


def test_checkpoint_present_and_loads():
    ckpt = inference.latest_checkpoint("cnn_lstm")
    if ckpt is None:
        pytest.skip("No CNN-LSTM checkpoint present; run engine/models/train_batch.py first")
    inf = inference.CNNLSTMInferencer(ckpt)
    assert inf.model is not None
    assert inf.checkpoint_path == ckpt


def test_inference_latency_under_50ms():
    ckpt = inference.latest_checkpoint("cnn_lstm")
    if ckpt is None:
        pytest.skip("No CNN-LSTM checkpoint present")
    inf = inference.CNNLSTMInferencer(ckpt)

    # Warm up: first forward pass triggers lazy CUDA/CPU kernels.
    rng = np.random.default_rng(0)
    warm = rng.standard_normal((60, 50)).astype(np.float32)
    inf.predict(warm)

    samples = []
    for _ in range(20):
        x = rng.standard_normal((60, 50)).astype(np.float32)
        t0 = time.perf_counter()
        pred = inf.predict(x)
        samples.append((time.perf_counter() - t0) * 1000.0)
        assert pred.label in ("BUY", "SELL", "HOLD")
        assert 0.0 <= pred.confidence <= 1.0

    median_ms = float(np.median(samples))
    p95_ms = float(np.percentile(samples, 95))
    assert median_ms < 50.0, f"median latency {median_ms:.1f}ms exceeds 50ms (p95={p95_ms:.1f}ms)"
