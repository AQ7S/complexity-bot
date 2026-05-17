from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal


Direction = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]

MIN_LOT_FOR_SLICING = 0.05
LIMIT_EXPIRY_HOURS = 2


@dataclass(frozen=True)
class SliceOrder:
    order_type: OrderType
    lot: float
    price: float
    sl: float
    tp: float
    expires_at: datetime | None
    reason: str


def plan_vwap_sliced_entry(
    direction: Direction,
    total_lots: float,
    market_price: float,
    vwap: float,
    sl: float,
    tp: float,
    volume_step: float = 0.01,
    now: datetime | None = None,
) -> list[SliceOrder]:
    now = now or datetime.now(timezone.utc)
    if total_lots < MIN_LOT_FOR_SLICING:
        return [SliceOrder(
            order_type="MARKET", lot=total_lots, price=market_price,
            sl=sl, tp=tp, expires_at=None,
            reason=f"SINGLE_SHOT: lot {total_lots:.2f} < {MIN_LOT_FOR_SLICING}",
        )]

    slice_lot = max(volume_step, round(total_lots / 3.0 / volume_step) * volume_step)
    third_lot = round(total_lots - 2 * slice_lot, 2)
    if third_lot < volume_step:
        third_lot = slice_lot

    favourable_vwap = (
        (direction == "BUY" and market_price > vwap) or
        (direction == "SELL" and market_price < vwap)
    )

    market_slice = SliceOrder(
        order_type="MARKET", lot=slice_lot, price=market_price,
        sl=sl, tp=tp, expires_at=None,
        reason="SLICE_1_MARKET",
    )

    if favourable_vwap:
        limit_slice = SliceOrder(
            order_type="LIMIT", lot=slice_lot, price=vwap,
            sl=sl, tp=tp,
            expires_at=now + timedelta(hours=LIMIT_EXPIRY_HOURS),
            reason="SLICE_2_LIMIT_AT_VWAP",
        )
    else:
        limit_slice = SliceOrder(
            order_type="MARKET", lot=slice_lot, price=market_price,
            sl=sl, tp=tp, expires_at=None,
            reason="SLICE_2_MARKET_VWAP_UNFAVOURABLE",
        )

    third_slice = SliceOrder(
        order_type="MARKET", lot=third_lot, price=market_price,
        sl=sl, tp=tp, expires_at=None,
        reason="SLICE_3_MARKET",
    )

    return [market_slice, limit_slice, third_slice]
