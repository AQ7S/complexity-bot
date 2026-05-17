"""Regime-aware specialist router.

A single CNN-LSTM cannot be optimal across trending, ranging, and high-
volatility regimes. The router maintains a separate checkpoint per regime
and selects the matching specialist at inference time based on the
current regime classification.

Naming convention for specialist checkpoints (looked up in CHECKPOINT_DIR):

    cnn_lstm_trending_v*_<date>.pt
    cnn_lstm_ranging_v*_<date>.pt
    cnn_lstm_volatile_v*_<date>.pt

If a specialist checkpoint is missing the router falls back to the
generalist `cnn_lstm_v*_<date>.pt` so the engine never crashes for lack
of a specialist. Promotion from generalist to specialist is a deliberate
operator action (or an output of Tier 3.3 champion-challenger).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from engine.models.inference import CHECKPOINT_DIR, CNNLSTMInferencer, Prediction


Regime = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY"]
SpecialistKind = Literal["trending", "ranging", "volatile"]


def select_specialist(regime: Regime | str | None) -> SpecialistKind:
    """Map a regime label to the specialist that should handle it."""
    if regime in ("TRENDING_UP", "TRENDING_DOWN"):
        return "trending"
    if regime == "RANGING":
        return "ranging"
    if regime == "HIGH_VOLATILITY":
        return "volatile"
    return "trending"


def _latest_specialist_checkpoint(kind: SpecialistKind) -> Path | None:
    if not CHECKPOINT_DIR.exists():
        return None
    pattern = f"cnn_lstm_{kind}_v*.pt"
    candidates = sorted(
        CHECKPOINT_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        try:
            if c.stat().st_size > 1024:
                return c
        except OSError:
            continue
    return None


@dataclass
class SpecialistRoutingResult:
    kind: SpecialistKind
    used_fallback: bool
    prediction: Prediction


class RegimeRouter:
    """Loads specialists lazily; reuses generalist when a specialist is absent."""

    def __init__(self, generalist: CNNLSTMInferencer | None = None, *, device: str | None = None) -> None:
        self._device = device
        self._generalist = generalist
        self._specialists: dict[SpecialistKind, CNNLSTMInferencer | None] = {
            "trending": None, "ranging": None, "volatile": None,
        }

    def _generalist_inferencer(self) -> CNNLSTMInferencer:
        if self._generalist is None:
            self._generalist = CNNLSTMInferencer(device=self._device)
        return self._generalist

    def _get_specialist(self, kind: SpecialistKind) -> CNNLSTMInferencer | None:
        if self._specialists.get(kind) is not None:
            return self._specialists[kind]
        ckpt = _latest_specialist_checkpoint(kind)
        if ckpt is None:
            self._specialists[kind] = None
            return None
        try:
            inf = CNNLSTMInferencer(checkpoint=ckpt, device=self._device)
        except Exception:  # noqa: BLE001
            inf = None  # type: ignore[assignment]
        self._specialists[kind] = inf
        return inf

    def predict_routed(self, window: np.ndarray, regime: Regime | str | None) -> SpecialistRoutingResult:
        kind = select_specialist(regime)
        specialist = self._get_specialist(kind)
        if specialist is not None:
            return SpecialistRoutingResult(
                kind=kind, used_fallback=False,
                prediction=specialist.predict(window),
            )
        return SpecialistRoutingResult(
            kind=kind, used_fallback=True,
            prediction=self._generalist_inferencer().predict(window),
        )
