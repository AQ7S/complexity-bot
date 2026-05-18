"""News surprise extractor + sigma-scoring.

The existing news layer (`engine.news.jblanked`) tells us *something is
coming* and how long until it fires. The actual edge — and the highest-
expectancy 30-second window in FX — is **actual vs forecast** divergence
on the released number:

    NFP forecast 150k, actual 320k  →  +170k surprise  →  USD bid
    CPI forecast 3.2%, actual 2.8%  →  -0.4pp surprise →  USD sold

This module converts an event's (actual, forecast, previous) tuple into:

  * a raw `surprise` (actual − forecast)
  * a `surprise_sigma` (normalized to the event's typical surprise scale)
  * a directional `bias` ('BULLISH_CCY' / 'BEARISH_CCY' / 'NEUTRAL')

The normalization is per-event: NFP surprises live in the ±200k range,
CPI surprises in the ±0.5pp range; without per-event sigma we'd compare
apples to oranges. A small calibration table holds the historical 1σ
band for the major events; everything else falls back to a generic z
score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Per-event historical 1-sigma bands (approximate; refreshed yearly).
EVENT_SIGMA: dict[str, float] = {
    "Non-Farm Payrolls":          75_000.0,
    "NFP":                        75_000.0,
    "Unemployment Rate":              0.15,
    "CPI m/m":                        0.15,
    "CPI y/y":                        0.20,
    "Core CPI":                       0.15,
    "GDP":                            0.30,
    "Retail Sales":                   0.40,
    "Industrial Production":          0.40,
    "PMI Manufacturing":              1.50,
    "PMI Services":                   1.50,
    "ISM Manufacturing":              1.50,
    "FOMC Rate Decision":             0.10,
    "ECB Rate Decision":              0.10,
    "BoE Rate Decision":              0.10,
    "Trade Balance":                  3.0,
}


# How a positive surprise translates to currency direction. Most prints
# are good-for-the-currency on a beat; a few (e.g. unemployment) are
# inverted.
INVERTED_EVENTS: frozenset[str] = frozenset({
    "Unemployment Rate",
    "Trade Balance",
})


Bias = Literal["BULLISH_CCY", "BEARISH_CCY", "NEUTRAL"]


@dataclass(frozen=True)
class NewsSurprise:
    event_name: str
    currency: str
    actual: float | None
    forecast: float | None
    previous: float | None
    surprise: float           # actual − forecast
    surprise_sigma: float     # normalized to event's typical surprise scale
    bias: Bias
    is_inverted: bool


def _resolve_sigma(event_name: str) -> float:
    """Pick the per-event 1σ band; fall back to a wide generic value."""
    key = event_name.strip()
    if key in EVENT_SIGMA:
        return EVENT_SIGMA[key]
    for k, v in EVENT_SIGMA.items():
        if k.lower() in key.lower():
            return v
    return 1.0


def compute_surprise(
    event_name: str,
    *,
    currency: str,
    actual: float | None,
    forecast: float | None,
    previous: float | None = None,
    sigma_threshold: float = 0.5,
) -> NewsSurprise:
    """Convert a released event into a normalized surprise record.

    `sigma_threshold` is the minimum |surprise_sigma| to set a directional
    bias; below that we return NEUTRAL.
    """
    is_inverted = any(inv.lower() == event_name.strip().lower() for inv in INVERTED_EVENTS)
    if actual is None or forecast is None:
        return NewsSurprise(
            event_name=event_name, currency=currency.upper(),
            actual=actual, forecast=forecast, previous=previous,
            surprise=0.0, surprise_sigma=0.0,
            bias="NEUTRAL", is_inverted=is_inverted,
        )
    raw = float(actual) - float(forecast)
    sigma = _resolve_sigma(event_name)
    z = raw / max(sigma, 1e-9)
    if abs(z) < sigma_threshold:
        bias: Bias = "NEUTRAL"
    else:
        positive = raw > 0
        if is_inverted:
            positive = not positive
        bias = "BULLISH_CCY" if positive else "BEARISH_CCY"
    return NewsSurprise(
        event_name=event_name, currency=currency.upper(),
        actual=float(actual), forecast=float(forecast),
        previous=float(previous) if previous is not None else None,
        surprise=raw, surprise_sigma=z,
        bias=bias, is_inverted=is_inverted,
    )


def directional_kick_for_pair(surprise: NewsSurprise, symbol: str) -> str:
    """Translate a per-currency bias to a per-pair direction.

    `symbol` is the standard XM format (`EURUSD#`, `USDJPY#`, etc). When
    the surprise's currency is the base, BULLISH_CCY → BUY the pair.
    When it's the quote, BULLISH_CCY → SELL the pair.
    """
    if surprise.bias == "NEUTRAL":
        return "HOLD"
    base = symbol[:3].upper()
    quote = symbol[3:6].upper() if len(symbol) >= 6 else ""
    ccy = surprise.currency.upper()
    if ccy == base:
        return "BUY" if surprise.bias == "BULLISH_CCY" else "SELL"
    if ccy == quote:
        return "SELL" if surprise.bias == "BULLISH_CCY" else "BUY"
    return "HOLD"
