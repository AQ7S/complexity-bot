"""Position sizing — Appendix E formula, with all edge cases enumerated.

The formula is intentionally pure (no MT5 import here) so it can be unit-tested
without a live broker. Inputs are extracted from `mt5.symbol_info()` and
`mt5.account_info()` by the order router and passed in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from engine.config import settings


@dataclass(frozen=True)
class SymbolInfo:
    """Subset of `mt5.symbol_info()` fields needed for sizing."""
    name: str
    point: float
    digits: int
    tick_size: float
    tick_value: float           # USD per tick per 1.0 lot, account currency
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float = 100_000.0


@dataclass(frozen=True)
class LotResult:
    ok: bool
    lot: float = 0.0
    raw_lot: float = 0.0
    risk_usd: float = 0.0
    loss_per_lot: float = 0.0
    sl_distance: float = 0.0
    reason: Literal[
        "OK",
        "INVALID_SL_DISTANCE",
        "INSUFFICIENT_EQUITY_FOR_RISK",
        "LOT_CAPPED_AT_BROKER_MAX",
        "INVALID_INPUT",
    ] = "OK"
    warnings: tuple[str, ...] = ()


def _quantize(value: float, step: float) -> float:
    if step <= 0:
        return value
    n = math.floor(value / step + 1e-9)
    return round(n * step, 8)


def compute_lot(
    *,
    equity: float,
    entry: float,
    sl_price: float,
    symbol: SymbolInfo,
    risk_pct: float = settings.RISK_PCT_PER_TRADE,
    claude_risk_adjustment: float = 1.0,
) -> LotResult:
    """Compute a broker-quantised lot size for the given trade.

    Returns a `LotResult` whose `ok=True` means the trade may be placed.
    """
    if equity <= 0 or symbol.tick_size <= 0 or symbol.volume_step <= 0:
        return LotResult(ok=False, reason="INVALID_INPUT")

    K = max(0.5, min(1.5, claude_risk_adjustment))
    risk_usd = equity * risk_pct * K
    sl_distance = abs(entry - sl_price)
    if sl_distance < symbol.tick_size:
        return LotResult(
            ok=False, risk_usd=risk_usd, sl_distance=sl_distance,
            reason="INVALID_SL_DISTANCE",
        )

    fixed_lot = float(getattr(settings, "FIXED_LOT", 0.0) or 0.0)
    if fixed_lot > 0:
        quant = _quantize(fixed_lot, symbol.volume_step)
        lot = max(symbol.volume_min, min(quant, symbol.volume_max))
        ticks_to_sl_fixed = sl_distance / symbol.tick_size
        loss_per_lot_fixed = ticks_to_sl_fixed * symbol.tick_value
        return LotResult(
            ok=True, lot=lot, raw_lot=fixed_lot, risk_usd=lot * loss_per_lot_fixed,
            loss_per_lot=loss_per_lot_fixed, sl_distance=sl_distance,
            reason="OK", warnings=(f"FIXED_LOT={fixed_lot} override active",),
        )

    ticks_to_sl = sl_distance / symbol.tick_size
    loss_per_lot = ticks_to_sl * symbol.tick_value
    if loss_per_lot <= 0:
        return LotResult(
            ok=False, risk_usd=risk_usd, sl_distance=sl_distance,
            reason="INVALID_SL_DISTANCE",
        )

    raw_lot = risk_usd / loss_per_lot
    quantised = _quantize(raw_lot, symbol.volume_step)

    warnings: list[str] = []
    if raw_lot < symbol.volume_min:
        return LotResult(
            ok=False, raw_lot=raw_lot, risk_usd=risk_usd,
            loss_per_lot=loss_per_lot, sl_distance=sl_distance,
            reason="INSUFFICIENT_EQUITY_FOR_RISK",
        )
    if raw_lot > symbol.volume_max:
        warnings.append(f"raw_lot {raw_lot:.4f} > volume_max {symbol.volume_max}")
        lot = symbol.volume_max
    else:
        lot = max(symbol.volume_min, quantised)

    reason = "LOT_CAPPED_AT_BROKER_MAX" if "raw_lot" in (w[:7] for w in warnings) else "OK"
    return LotResult(
        ok=True, lot=lot, raw_lot=raw_lot, risk_usd=risk_usd,
        loss_per_lot=loss_per_lot, sl_distance=sl_distance,
        reason=reason, warnings=tuple(warnings),
    )
