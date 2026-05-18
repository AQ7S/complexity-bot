"""Tests for data-integrity guards (Tier 8.2): survivorship + look-ahead bias."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.integrity import (
    LookAheadReport,
    check_look_ahead_bias,
    summarize_survivorship,
    SurvivorshipReport,
)


def _bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(100, 1000, n),
    })


def test_look_ahead_clean_sma():
    bars = _bars()
    rep = check_look_ahead_bias(
        lambda b: b["close"].rolling(20).mean(),
        bars, feature_name="sma20",
    )
    assert not rep.has_leak
    assert rep.bars_compared > 0


def test_look_ahead_leak_detected_for_last_bar_broadcast():
    bars = _bars()
    # Leak: every value of the feature equals the *final* close — so when
    # we truncate the last bar, every value of the truncated series
    # changes too.
    rep = check_look_ahead_bias(
        lambda b: pd.Series([b["close"].iloc[-1]] * len(b), index=b.index),
        bars, feature_name="last_bar_broadcast",
    )
    assert rep.has_leak


def test_look_ahead_leak_detected_for_forward_max():
    bars = _bars()
    def leaky(b):
        closes = b["close"].to_numpy()
        return pd.Series([closes[i:].max() for i in range(len(closes))],
                          index=b.index)
    rep = check_look_ahead_bias(leaky, bars, feature_name="forward_max")
    assert rep.has_leak


def test_look_ahead_too_few_bars():
    bars = _bars(n=10)
    rep = check_look_ahead_bias(lambda b: b["close"], bars, feature_name="x")
    assert not rep.has_leak
    assert rep.bars_compared == 0
    assert "too few" in rep.notes


def test_look_ahead_handles_feature_exception():
    bars = _bars()
    rep = check_look_ahead_bias(
        lambda b: (_ for _ in ()).throw(RuntimeError("boom")),
        bars, feature_name="bad",
    )
    assert not rep.has_leak
    assert "raised" in rep.notes


def test_summarize_survivorship_all_ok():
    reps = [
        SurvivorshipReport("EURUSD#", 365, None, None, 0, True, "ok"),
        SurvivorshipReport("USDJPY#", 365, None, None, 0, True, "ok"),
    ]
    s = summarize_survivorship(reps)
    assert s["all_ok"] is True
    assert s["n_failing"] == 0


def test_summarize_survivorship_with_failures():
    reps = [
        SurvivorshipReport("EURUSD#", 365, None, None, 0, True, "ok"),
        SurvivorshipReport("DELISTED#", 365, None, None, 0, False, "no bars"),
    ]
    s = summarize_survivorship(reps)
    assert s["all_ok"] is False
    assert s["n_failing"] == 1
    assert "DELISTED#" in s["failing_symbols"]
