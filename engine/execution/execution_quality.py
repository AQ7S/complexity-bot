from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from loguru import logger


Direction = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]

LIMIT_DISTANCE_ATR_MULT = 0.5
LIMIT_EXPIRY_HOURS = 3
MAX_SLIPPAGE_PIPS = 2.0


@dataclass
class EntryRecommendation:
    order_type: OrderType
    entry_price: float
    expires_at: datetime | None
    reason: str


def recommend_entry(
    current_price: float,
    zone_entry: float,
    direction: Direction,
    atr14: float,
    now: datetime | None = None,
) -> EntryRecommendation:
    now = now or datetime.now(timezone.utc)
    if atr14 <= 0:
        return EntryRecommendation(
            order_type="MARKET", entry_price=current_price,
            expires_at=None, reason="ATR_INVALID_FALLBACK_MARKET",
        )
    inside_zone = (
        (direction == "BUY"  and current_price <= zone_entry) or
        (direction == "SELL" and current_price >= zone_entry)
    )
    if inside_zone:
        return EntryRecommendation(
            order_type="MARKET", entry_price=current_price,
            expires_at=None, reason="INSIDE_ZONE",
        )
    distance = abs(current_price - zone_entry)
    if distance <= LIMIT_DISTANCE_ATR_MULT * atr14:
        return EntryRecommendation(
            order_type="LIMIT", entry_price=zone_entry,
            expires_at=now + timedelta(hours=LIMIT_EXPIRY_HOURS),
            reason=f"LIMIT_AT_ZONE: dist={distance:.5f} < 0.5×ATR={LIMIT_DISTANCE_ATR_MULT*atr14:.5f}",
        )
    return EntryRecommendation(
        order_type="MARKET", entry_price=current_price,
        expires_at=None,
        reason=f"TOO_FAR_FROM_ZONE: dist={distance:.5f} > 0.5×ATR",
    )


def pip_size_from_digits(digits: int) -> float:
    if digits >= 4:
        return 10 ** -(digits - 1)
    return 10 ** -digits if digits > 0 else 1.0


def slippage_pips(
    requested_price: float, executed_price: float, digits: int,
) -> float:
    pip = pip_size_from_digits(digits)
    return abs(executed_price - requested_price) / pip


def slippage_acceptable(
    requested_price: float, executed_price: float, digits: int,
    max_pips: float = MAX_SLIPPAGE_PIPS,
) -> tuple[bool, float]:
    pips = slippage_pips(requested_price, executed_price, digits)
    ok = pips <= max_pips
    if not ok:
        logger.warning(
            f"HIGH_SLIPPAGE: requested={requested_price} executed={executed_price} "
            f"slip={pips:.1f} pips > {max_pips}"
        )
    return ok, pips
