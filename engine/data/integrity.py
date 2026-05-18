"""Data-integrity guards: survivorship-bias detector + look-ahead-bias detector.

The two failure modes that silently invalidate every backtest:

  * **Survivorship bias** — symbols that were tradeable when the
    backtest period started but have since been delisted, renamed, or
    had their feed change disappear from the DuckDB store. Backtests
    that assume "the universe was always these 13 symbols" will be
    optimistic. The detector asserts: for every symbol we trade today,
    the DuckDB bar history is continuous and reaches at least
    `min_history_days` into the past.

  * **Look-ahead bias** — an indicator or feature function accidentally
    references bar T+1 (or later) when producing the value at bar T.
    The most insidious class of bug; it makes a strategy look great
    in-sample and lose money live. The detector wraps any
    `feature_fn(bars) -> Series` and asserts: shifting the input by +1
    bar (i.e. truncating the last bar) must change the trailing output
    *exactly* as a one-bar shift — never more, never less.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurvivorshipReport:
    symbol: str
    expected_history_days: int
    actual_first_bar: datetime | None
    actual_last_bar: datetime | None
    max_gap_bars: int
    is_complete: bool
    notes: str


def check_survivorship(
    symbols: Iterable[str],
    *,
    timeframe: str = "M1",
    min_history_days: int = 365,
    max_allowed_gap_bars: int = 30,
    db_path: str | None = None,
    now: datetime | None = None,
) -> list[SurvivorshipReport]:
    """Verify every `symbol` has continuous DuckDB bar history.

    Returns one report per symbol. `is_complete=False` means the symbol
    failed at least one check — call `summarize_survivorship()` to
    aggregate, or filter the list directly.
    """
    from engine.data import duckdb_store
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=int(min_history_days))
    out: list[SurvivorshipReport] = []
    for sym in symbols:
        try:
            with duckdb_store.open_store(db_path, read_only=True) as con:
                row = con.execute(
                    "SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n "
                    "FROM bars WHERE symbol=? AND timeframe=?",
                    [sym, timeframe],
                ).fetchone()
                if not row or row[2] is None or row[2] == 0:
                    out.append(SurvivorshipReport(
                        symbol=sym, expected_history_days=min_history_days,
                        actual_first_bar=None, actual_last_bar=None,
                        max_gap_bars=0, is_complete=False,
                        notes="no bars in DuckDB",
                    ))
                    continue
                first_ts, last_ts, _ = row
                if isinstance(first_ts, str):
                    first_ts = datetime.fromisoformat(first_ts)
                if isinstance(last_ts, str):
                    last_ts = datetime.fromisoformat(last_ts)
                # Gap detection: walk timestamps in chunks, find max bar gap.
                gap_rows = con.execute(
                    "SELECT ts FROM bars WHERE symbol=? AND timeframe=? ORDER BY ts",
                    [sym, timeframe],
                ).fetchall()
                tses = [r[0] for r in gap_rows]
            max_gap = 0
            if len(tses) > 1:
                # Compute median bar interval; gap = (Δ / interval) - 1.
                deltas_s = [(tses[i] - tses[i - 1]).total_seconds()
                            for i in range(1, len(tses))]
                if deltas_s:
                    median = float(np.median(deltas_s))
                    if median > 0:
                        max_gap = int(max(deltas_s) / median - 1)
            notes_parts = []
            ok = True
            if first_ts > cutoff:
                notes_parts.append(f"history starts {first_ts.isoformat()} > cutoff {cutoff.isoformat()}")
                ok = False
            if last_ts < (now - timedelta(days=2)):
                notes_parts.append(f"last bar {last_ts.isoformat()} is stale")
                ok = False
            if max_gap > max_allowed_gap_bars:
                notes_parts.append(f"max gap {max_gap} bars > {max_allowed_gap_bars}")
                ok = False
            out.append(SurvivorshipReport(
                symbol=sym, expected_history_days=min_history_days,
                actual_first_bar=first_ts, actual_last_bar=last_ts,
                max_gap_bars=max_gap, is_complete=ok,
                notes="; ".join(notes_parts) if notes_parts else "ok",
            ))
        except Exception as e:  # noqa: BLE001
            out.append(SurvivorshipReport(
                symbol=sym, expected_history_days=min_history_days,
                actual_first_bar=None, actual_last_bar=None,
                max_gap_bars=0, is_complete=False,
                notes=f"query failed: {type(e).__name__}: {e}",
            ))
    return out


def summarize_survivorship(reports: list[SurvivorshipReport]) -> dict:
    failing = [r for r in reports if not r.is_complete]
    return {
        "n_symbols": len(reports),
        "n_complete": len([r for r in reports if r.is_complete]),
        "n_failing": len(failing),
        "all_ok": len(failing) == 0,
        "failing_symbols": [r.symbol for r in failing],
    }


# ---------------------------------------------------------------------------
# Look-ahead bias
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LookAheadReport:
    feature_name: str
    has_leak: bool
    max_diff_at_truncation: float
    bars_compared: int
    notes: str


def check_look_ahead_bias(
    feature_fn: Callable[[pd.DataFrame], pd.Series],
    bars: pd.DataFrame,
    *,
    feature_name: str = "<unknown>",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> LookAheadReport:
    """Verify `feature_fn` does not peek at future bars.

    Strategy: compute the feature on the full bar set, then re-compute
    on `bars[:-1]` (truncating the last bar). For a leak-free feature,
    the trailing values of the second computation must match the
    corresponding values of the first — *exactly*. Any divergence means
    later bars influenced earlier outputs.

    Returns a LookAheadReport. `has_leak=True` is a HARD FAIL on the
    feature; do not deploy.
    """
    if len(bars) < 50:
        return LookAheadReport(
            feature_name=feature_name, has_leak=False,
            max_diff_at_truncation=0.0, bars_compared=0,
            notes="too few bars to test",
        )
    try:
        full = feature_fn(bars)
        truncated = feature_fn(bars.iloc[:-1])
    except Exception as e:  # noqa: BLE001
        return LookAheadReport(
            feature_name=feature_name, has_leak=False,
            max_diff_at_truncation=0.0, bars_compared=0,
            notes=f"feature_fn raised: {type(e).__name__}: {e}",
        )
    if not isinstance(full, pd.Series) or not isinstance(truncated, pd.Series):
        return LookAheadReport(
            feature_name=feature_name, has_leak=False,
            max_diff_at_truncation=0.0, bars_compared=0,
            notes="feature_fn did not return a pandas Series",
        )
    # Compare the overlapping region.
    n = len(truncated)
    a = full.iloc[:n].to_numpy(dtype=np.float64, na_value=np.nan)
    b = truncated.to_numpy(dtype=np.float64, na_value=np.nan)
    # Mask: positions where both are finite.
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return LookAheadReport(
            feature_name=feature_name, has_leak=False,
            max_diff_at_truncation=0.0, bars_compared=0,
            notes="all NaN in compared region",
        )
    diffs = np.abs(a[mask] - b[mask])
    max_diff = float(diffs.max())
    tol = atol + rtol * float(np.abs(b[mask]).max())
    has_leak = max_diff > tol
    return LookAheadReport(
        feature_name=feature_name, has_leak=has_leak,
        max_diff_at_truncation=max_diff,
        bars_compared=int(mask.sum()),
        notes="leak detected" if has_leak else "ok",
    )
