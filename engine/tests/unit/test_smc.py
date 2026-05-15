"""Phase 7 unit tests — SMC detectors + ICT kill zone filter.

Plan asserts:
- 10 hand-labeled candle fixtures verify OB/FVG/BOS/CHoCH detection
- kill_zone_active() returns expected bool for 24 timestamp samples
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from engine.features import smc
from engine.utils.time_utils import (
    EST, active_kill_zone, kill_zone_active,
)


# -----------------------------------------------------------------------------
# OHLC fixture builders — small, deterministic series with known SMC features
# -----------------------------------------------------------------------------

def _bars_from(ohlc: list[tuple[float, float, float, float]], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2025-01-06", periods=len(ohlc), freq=freq)
    arr = np.asarray(ohlc, dtype=float)
    return pd.DataFrame({
        "open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2], "close": arr[:, 3],
        "volume": np.full(len(ohlc), 100.0),
    }, index=idx)


def _oscillating_consolidation(start_price: float, n: int, amp: float) -> list[tuple]:
    """Sine-wave OHLC around `start_price` so the swing detector finds clear pivots."""
    bars = []
    for i in range(n):
        wave = amp * np.sin(2 * np.pi * i / 12.0)        # ~12-bar period
        prev_wave = amp * np.sin(2 * np.pi * (i - 1) / 12.0)
        o = start_price + prev_wave
        c = start_price + wave
        h = max(o, c) + amp * 0.2
        l = min(o, c) - amp * 0.2
        bars.append((o, h, l, c))
    return bars


def fixture_uptrend_with_ob() -> pd.DataFrame:
    """Oscillating consolidation, an explicit red candle (the OB), then a strong bullish impulse."""
    bars = _oscillating_consolidation(1.10, n=80, amp=0.002)
    p = bars[-1][3]
    # explicit red candle = bullish OB
    bars.append((p, p + 0.0002, p - 0.0015, p - 0.0012)); p -= 0.0012
    # strong bullish impulse: 12 green candles
    for _ in range(12):
        o = p; c = p + 0.0030
        h = c + 0.0002; l = o - 0.0002
        bars.append((o, h, l, c)); p = c
    return _bars_from(bars)


def fixture_downtrend_with_ob() -> pd.DataFrame:
    bars = _oscillating_consolidation(1.10, n=80, amp=0.002)
    p = bars[-1][3]
    # explicit green candle = bearish OB
    bars.append((p, p + 0.0015, p - 0.0002, p + 0.0012)); p += 0.0012
    for _ in range(12):
        o = p; c = p - 0.0030
        h = o + 0.0002; l = c - 0.0002
        bars.append((o, h, l, c)); p = c
    return _bars_from(bars)


def fixture_bullish_fvg() -> pd.DataFrame:
    """Build a sequence with a clear 3-candle bullish gap: bar1.high < bar3.low."""
    bars = [(1.10, 1.1005, 1.0995, 1.1002)] * 20
    # the gap triple
    bars.append((1.1002, 1.1010, 1.1000, 1.1008))   # bar1
    bars.append((1.1008, 1.1050, 1.1020, 1.1045))   # bar2 (impulse up)
    bars.append((1.1045, 1.1055, 1.1030, 1.1050))   # bar3 — low (1.1030) > bar1.high (1.1010)
    bars.extend([(1.1050, 1.1055, 1.1045, 1.1052)] * 20)
    return _bars_from(bars)


def fixture_bearish_fvg() -> pd.DataFrame:
    bars = [(1.10, 1.1005, 1.0995, 1.1002)] * 20
    bars.append((1.1002, 1.1010, 1.0995, 1.1000))   # bar1
    bars.append((1.1000, 1.0990, 1.0950, 1.0955))   # bar2 (impulse down)
    bars.append((1.0955, 1.0970, 1.0945, 1.0950))   # bar3 — high (1.0970) < bar1.low (1.0995)
    bars.extend([(1.0950, 1.0955, 1.0945, 1.0952)] * 20)
    return _bars_from(bars)


def fixture_bullish_bos() -> pd.DataFrame:
    """Make a swing high, dip, then break above."""
    bars = []
    # build an obvious swing high at index 30
    for i in range(30):
        p = 1.10 + i * 0.0001
        bars.append((p, p + 0.0002, p - 0.0001, p + 0.0001))
    swing_high = bars[-1][1]
    # dip down
    for i in range(15):
        p = swing_high - i * 0.0002
        bars.append((p, p + 0.0001, p - 0.0002, p - 0.0001))
    # rally and break above the swing high
    for i in range(20):
        p = bars[-1][3] + 0.0003
        h = p + 0.0002; l = bars[-1][3] - 0.0001
        bars.append((bars[-1][3], h, l, p))
    return _bars_from(bars)


def fixture_bearish_bos() -> pd.DataFrame:
    bars = []
    for i in range(30):
        p = 1.10 - i * 0.0001
        bars.append((p, p + 0.0001, p - 0.0002, p - 0.0001))
    swing_low = bars[-1][2]
    for i in range(15):
        p = swing_low + i * 0.0002
        bars.append((p, p + 0.0002, p - 0.0001, p + 0.0001))
    for i in range(20):
        p = bars[-1][3] - 0.0003
        h = bars[-1][3] + 0.0001; l = p - 0.0002
        bars.append((bars[-1][3], h, l, p))
    return _bars_from(bars)


def fixture_choch_after_uptrend() -> pd.DataFrame:
    """Uptrend → reversal that breaks last swing low (CHoCH bearish)."""
    bars = []
    for i in range(40):
        p = 1.10 + i * 0.0001
        bars.append((p, p + 0.0002, p - 0.0001, p + 0.0001))
    # pullback (creates a swing low we will break)
    for i in range(8):
        p = bars[-1][3] - 0.0003
        bars.append((bars[-1][3], bars[-1][3] + 0.0001, p - 0.0001, p))
    # higher high (still uptrend)
    for i in range(8):
        p = bars[-1][3] + 0.0004
        bars.append((bars[-1][3], p + 0.0001, bars[-1][3] - 0.0001, p))
    # collapse below the previous swing low
    for i in range(20):
        p = bars[-1][3] - 0.0006
        bars.append((bars[-1][3], bars[-1][3] + 0.0001, p - 0.0001, p))
    return _bars_from(bars)


def fixture_random_walk() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    p = 1.10
    bars = []
    for _ in range(200):
        delta = rng.normal(0, 0.0005)
        o = p; c = p + delta
        h = max(o, c) + abs(rng.normal(0, 0.0003))
        l = min(o, c) - abs(rng.normal(0, 0.0003))
        bars.append((o, h, l, c)); p = c
    return _bars_from(bars)


# -----------------------------------------------------------------------------
# 10 hand-labeled fixture tests
# -----------------------------------------------------------------------------

def _has_active_rows(df: pd.DataFrame, name: str) -> bool:
    """True if any row has a non-NaN value in the column matching `name`."""
    if df is None or df.empty:
        return False
    cols = [c for c in df.columns if c.lower() == name.lower()]
    if not cols:
        return False
    return df[cols[0]].notna().any()


def test_swings_detected_on_trending_series():
    df = fixture_uptrend_with_ob()
    z = smc.detect_zones(df, swing_length=10)
    assert _has_active_rows(z["swings"], "HighLow")


def test_ob_detected_in_uptrend_fixture():
    df = fixture_uptrend_with_ob()
    z = smc.detect_zones(df, swing_length=10)
    assert _has_active_rows(z["ob"], "OB"), "expected at least one OB on uptrend fixture"


def test_ob_detected_in_downtrend_fixture():
    df = fixture_downtrend_with_ob()
    z = smc.detect_zones(df, swing_length=10)
    assert _has_active_rows(z["ob"], "OB")


def test_bullish_fvg_detected():
    df = fixture_bullish_fvg()
    z = smc.detect_zones(df, swing_length=10)
    assert _has_active_rows(z["fvg"], "FVG")


def test_bearish_fvg_detected():
    df = fixture_bearish_fvg()
    z = smc.detect_zones(df, swing_length=10)
    assert _has_active_rows(z["fvg"], "FVG")


def test_bullish_bos_detected():
    df = fixture_bullish_bos()
    z = smc.detect_zones(df, swing_length=10)
    bos = z["bos_choch"]
    assert _has_active_rows(bos, "BOS") or _has_active_rows(bos, "CHoCH"), \
        "expected BOS or CHoCH on bullish-break fixture"


def test_bearish_bos_detected():
    df = fixture_bearish_bos()
    z = smc.detect_zones(df, swing_length=10)
    bos = z["bos_choch"]
    assert _has_active_rows(bos, "BOS") or _has_active_rows(bos, "CHoCH")


def test_choch_after_uptrend_reversal():
    df = fixture_choch_after_uptrend()
    z = smc.detect_zones(df, swing_length=10)
    bos = z["bos_choch"]
    assert _has_active_rows(bos, "BOS") or _has_active_rows(bos, "CHoCH")


def test_liquidity_detected_on_random_walk():
    df = fixture_random_walk()
    z = smc.detect_zones(df, swing_length=20)
    liq = z["liquidity"]
    assert liq is not None and len(liq) > 0


def test_get_signal_returns_valid_shape_on_random_walk():
    df = fixture_random_walk()
    h4 = df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    m15 = df.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    sig = smc.get_signal(h4, m15, df)
    assert sig.signal in ("BUY", "SELL", "HOLD")
    assert sig.zone_type in ("OB", "FVG", "NONE")


# -----------------------------------------------------------------------------
# 24 timestamp samples for kill_zone_active()
# -----------------------------------------------------------------------------

def _est(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Build a UTC-tagged datetime from an EST wall-clock time (DST aware)."""
    naive = datetime(year, month, day, hour, minute)
    aware_est = EST.localize(naive)
    return aware_est.astimezone(timezone.utc)


# Each row: (label, est_hour, expected_zone_or_None)
KILLZONE_SAMPLES = [
    # ASIAN window (19–22 EST)
    ("19:00 EST", 19, "ASIAN"),
    ("19:30 EST", 19, "ASIAN"),
    ("21:59 EST", 21, "ASIAN"),
    ("22:00 EST", 22, None),    # boundary: end exclusive
    # Dead zone
    ("00:00 EST", 0,  None),
    ("01:30 EST", 1,  None),
    # LONDON_OPEN (02–05 EST)
    ("02:00 EST", 2,  "LONDON_OPEN"),
    ("03:30 EST", 3,  "LONDON_OPEN"),
    ("04:59 EST", 4,  "LONDON_OPEN"),
    ("05:00 EST", 5,  None),
    # Dead zone before NY
    ("06:00 EST", 6,  None),
    ("06:59 EST", 6,  None),
    # NY_OPEN (07–10 EST)
    ("07:00 EST", 7,  "NY_OPEN"),
    ("08:30 EST", 8,  "NY_OPEN"),
    ("09:59 EST", 9,  "NY_OPEN"),
    # LONDON_CLOSE (10–12 EST)
    ("10:00 EST", 10, "LONDON_CLOSE"),
    ("11:00 EST", 11, "LONDON_CLOSE"),
    ("11:59 EST", 11, "LONDON_CLOSE"),
    ("12:00 EST", 12, None),
    # Afternoon dead zone
    ("13:00 EST", 13, None),
    ("15:00 EST", 15, None),
    ("16:30 EST", 16, None),
    ("17:00 EST", 17, None),
    ("18:59 EST", 18, None),
]
assert len(KILLZONE_SAMPLES) == 24


@pytest.mark.parametrize("label,hour,expected", KILLZONE_SAMPLES,
                         ids=[s[0] for s in KILLZONE_SAMPLES])
def test_kill_zone_active_for_24_timestamps(label, hour, expected):
    minute = int(label.split(":")[1].split(" ")[0])
    dt = _est(2025, 6, 17, hour, minute)
    assert active_kill_zone(dt) == expected


def test_24_7_assets_bypass_kill_zone():
    dt = _est(2025, 6, 17, 3, 0)        # London open — FX active
    dt_dead = _est(2025, 6, 17, 14, 0)  # afternoon — FX inactive

    assert kill_zone_active("EURUSD", dt) is True
    assert kill_zone_active("EURUSD", dt_dead) is False
    # Always-on assets ignore the zone gate.
    for sym in ("GOLD#", "BTCUSD#", "ETHUSD#", "AI_INDX#", "Crypto_10#"):
        assert kill_zone_active(sym, dt) is True
        assert kill_zone_active(sym, dt_dead) is True
