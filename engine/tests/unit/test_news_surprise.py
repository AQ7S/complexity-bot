"""Tests for news surprise normalization (Tier 8.4)."""
from __future__ import annotations

import pytest

from engine.news.surprise import (
    EVENT_SIGMA,
    compute_surprise,
    directional_kick_for_pair,
)


def test_nfp_beat_is_bullish_usd():
    s = compute_surprise(
        "Non-Farm Payrolls", currency="USD",
        actual=320_000, forecast=150_000,
    )
    assert s.surprise == pytest.approx(170_000)
    assert s.surprise_sigma > 0
    assert s.bias == "BULLISH_CCY"


def test_nfp_miss_is_bearish_usd():
    s = compute_surprise(
        "Non-Farm Payrolls", currency="USD",
        actual=30_000, forecast=150_000,
    )
    assert s.bias == "BEARISH_CCY"


def test_unemployment_inverted():
    # Higher-than-expected unemployment is BEARISH for the currency.
    s = compute_surprise(
        "Unemployment Rate", currency="USD",
        actual=4.5, forecast=4.0,
    )
    assert s.is_inverted
    assert s.bias == "BEARISH_CCY"


def test_neutral_when_below_threshold():
    s = compute_surprise(
        "CPI m/m", currency="USD",
        actual=0.21, forecast=0.20, sigma_threshold=0.5,
    )
    assert s.bias == "NEUTRAL"


def test_missing_inputs_returns_neutral():
    s = compute_surprise("NFP", currency="USD", actual=None, forecast=150_000)
    assert s.bias == "NEUTRAL"
    assert s.surprise == 0.0


def test_directional_kick_base_currency():
    s = compute_surprise(
        "Non-Farm Payrolls", currency="USD",
        actual=320_000, forecast=150_000,
    )
    # USD is the quote in EURUSD#, so USD strength → SELL the pair.
    assert directional_kick_for_pair(s, "EURUSD#") == "SELL"
    # USD is the base in USDJPY#, so USD strength → BUY the pair.
    assert directional_kick_for_pair(s, "USDJPY#") == "BUY"


def test_directional_kick_unrelated_currency_holds():
    s = compute_surprise(
        "Non-Farm Payrolls", currency="USD",
        actual=320_000, forecast=150_000,
    )
    assert directional_kick_for_pair(s, "EURJPY#") == "HOLD"


def test_event_sigma_table_populated():
    assert EVENT_SIGMA["Non-Farm Payrolls"] > 0
    assert "CPI m/m" in EVENT_SIGMA
