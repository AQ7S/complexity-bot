from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests
from loguru import logger


YieldCurveBias = Literal["USD_BULLISH", "USD_BEARISH", "NEUTRAL"]
FearGreedState = Literal["EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"]

FRED_TIMEOUT_S = 5
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
FEAR_GREED_TIMEOUT_S = 5


def _fred_key_set() -> bool:
    key = os.getenv("FRED_API_KEY", "").strip()
    return key not in ("", "unset")


@dataclass(frozen=True)
class MacroSnapshot:
    yield_curve_bias: YieldCurveBias
    crypto_fear_greed: FearGreedState
    fear_greed_value: int | None
    spread_us10y_us2y: float | None


def get_yield_curve_bias() -> tuple[YieldCurveBias, float | None]:
    if not _fred_key_set():
        return "NEUTRAL", None
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv("FRED_API_KEY", "").strip())
        us10y = fred.get_series("DGS10", limit=5)
        us2y = fred.get_series("DGS2", limit=5)
        spread_now = float(us10y.iloc[-1]) - float(us2y.iloc[-1])
        spread_prev = float(us10y.iloc[-2]) - float(us2y.iloc[-2])
        if spread_now > spread_prev:
            return "USD_BULLISH", spread_now
        if spread_now < spread_prev:
            return "USD_BEARISH", spread_now
        return "NEUTRAL", spread_now
    except Exception as e:
        logger.warning("FRED yield curve fetch failed: {}", e)
        return "NEUTRAL", None


def get_crypto_fear_greed() -> tuple[FearGreedState, int | None]:
    try:
        r = requests.get(FEAR_GREED_URL, timeout=FEAR_GREED_TIMEOUT_S)
        if r.status_code != 200:
            return "NEUTRAL", None
        value = int(r.json()["data"][0]["value"])
        if value <= 20:
            return "EXTREME_FEAR", value
        if value <= 40:
            return "FEAR", value
        if value >= 80:
            return "EXTREME_GREED", value
        if value >= 60:
            return "GREED", value
        return "NEUTRAL", value
    except Exception as e:
        logger.warning("crypto fear/greed fetch failed: {}", e)
        return "NEUTRAL", None


def get_macro_snapshot() -> MacroSnapshot:
    bias, spread = get_yield_curve_bias()
    fg, fg_value = get_crypto_fear_greed()
    return MacroSnapshot(
        yield_curve_bias=bias,
        crypto_fear_greed=fg,
        fear_greed_value=fg_value,
        spread_us10y_us2y=spread,
    )
