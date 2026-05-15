from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from loguru import logger


Direction = Literal["BUY", "SELL"]

CORRELATION_GROUPS: dict[str, tuple[str, ...]] = {
    "EURUSD_CLUSTER": ("EURUSD#", "GBPUSD#", "USDCHF#"),
    "JPY_CLUSTER":    ("USDJPY#", "EURJPY#"),
    "CRYPTO_CLUSTER": ("BTCUSD#", "ETHUSD#", "AI_INDX#", "Crypto_10#"),
    "GOLD_SOLO":      ("GOLD#",),
    "EVENT_CLUSTER":  ("TrumpWinners#", "HarrisWinners#"),
    "AUD_SOLO":       ("AUDUSD#",),
}

MAX_PER_GROUP = 2


@dataclass(frozen=True)
class OpenPositionRef:
    symbol: str
    direction: Direction


@dataclass
class CorrelationVerdict:
    allowed: bool
    reason: str = "OK"


def _groups_for(symbol: str) -> list[str]:
    return [name for name, members in CORRELATION_GROUPS.items() if symbol in members]


def position_allowed(
    new_symbol: str,
    new_direction: Direction,
    open_positions: Iterable[OpenPositionRef],
) -> CorrelationVerdict:
    open_list = list(open_positions)
    for group_name in _groups_for(new_symbol):
        members = CORRELATION_GROUPS[group_name]
        group_open = [p for p in open_list if p.symbol in members]
        for p in group_open:
            if p.direction != new_direction:
                reason = (
                    f"HEDGE_BLOCK: {new_direction} {new_symbol} conflicts with "
                    f"{p.direction} {p.symbol} in group {group_name}"
                )
                logger.warning(reason)
                return CorrelationVerdict(allowed=False, reason=reason)
        if len(group_open) >= MAX_PER_GROUP:
            reason = (
                f"CORRELATION_BLOCK: {new_symbol} — group {group_name} "
                f"already has {len(group_open)} open positions"
            )
            logger.warning(reason)
            return CorrelationVerdict(allowed=False, reason=reason)
    return CorrelationVerdict(allowed=True)
