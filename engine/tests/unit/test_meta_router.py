"""Tests for the regime-specialist meta-router (Tier 3.1)."""
from __future__ import annotations

from engine.models.meta_router import RegimeRouter, select_specialist


def test_select_specialist_routes_trending():
    assert select_specialist("TRENDING_UP") == "trending"
    assert select_specialist("TRENDING_DOWN") == "trending"


def test_select_specialist_routes_ranging():
    assert select_specialist("RANGING") == "ranging"


def test_select_specialist_routes_volatile():
    assert select_specialist("HIGH_VOLATILITY") == "volatile"


def test_select_specialist_default_trending_on_unknown():
    assert select_specialist(None) == "trending"
    assert select_specialist("UNKNOWN") == "trending"


def test_router_uses_fallback_when_no_specialist_present():
    # Don't load any specialist; the generalist mock returns fixed
    # prediction.
    class _Stub:
        def predict(self, window):
            from engine.models.inference import Prediction
            return Prediction(label="HOLD", confidence=0.5, probs={"BUY": 0.2, "SELL": 0.3, "HOLD": 0.5})

    import numpy as np
    router = RegimeRouter(generalist=_Stub())  # type: ignore[arg-type]
    result = router.predict_routed(np.zeros((60, 50), dtype=np.float32), "TRENDING_UP")
    assert result.used_fallback
    assert result.kind == "trending"
    assert result.prediction.label == "HOLD"
