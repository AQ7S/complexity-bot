from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from loguru import logger


Direction = Literal["BUY", "SELL"]
Action = Literal["NOOP", "PARTIAL_CLOSE", "MOVE_TO_BREAKEVEN", "TRAIL", "TIME_EXIT", "FULL_CLOSE"]

PARTIAL_CLOSE_STAGES = (
    {"r_multiple": 1.0, "close_pct": 0.33, "label": "STAGE_1_1R"},
    {"r_multiple": 2.0, "close_pct": 0.33, "label": "STAGE_2_2R"},
    {"r_multiple": 3.0, "close_pct": 0.34, "label": "STAGE_3_TRAIL"},
)

TIME_EXIT_HOURS = 4.0
TIME_EXIT_MIN_R = 0.5
ATR_TRAIL_MULTIPLIER = 1.5
BREAKEVEN_BUFFER_TICKS = 2


@dataclass
class ManagedPosition:
    ticket: int
    symbol: str
    direction: Direction
    entry: float
    sl: float
    initial_sl: float
    tp: float
    lot: float
    open_time: datetime
    atr14_at_entry: float
    tick_size: float
    digits: int
    stages_done: set[str] = field(default_factory=set)
    breakeven_set: bool = False
    trail_anchor: float | None = None


@dataclass
class PositionAction:
    action: Action
    close_pct: float = 0.0
    new_sl: float | None = None
    reason: str = ""


def r_multiple(entry: float, current: float, sl: float, direction: Direction) -> float:
    risk = abs(entry - sl)
    if risk == 0:
        return 0.0
    if direction == "BUY":
        return (current - entry) / risk
    return (entry - current) / risk


def breakeven_sl_price(entry: float, direction: Direction, tick_size: float) -> float:
    buf = BREAKEVEN_BUFFER_TICKS * tick_size
    return entry + buf if direction == "BUY" else entry - buf


def compute_initial_sl(
    entry: float,
    ob_boundary: float | None,
    atr14: float,
    direction: Direction,
    min_atr_mult: float = 1.0,
    max_atr_mult: float = 3.0,
) -> tuple[float | None, str]:
    atr_floor = entry - min_atr_mult * atr14 if direction == "BUY" else entry + min_atr_mult * atr14
    atr_ceiling = entry - max_atr_mult * atr14 if direction == "BUY" else entry + max_atr_mult * atr14
    if direction == "BUY":
        candidate = ob_boundary if ob_boundary is not None else atr_floor
        if candidate >= entry:
            candidate = atr_floor
        candidate = min(candidate, atr_floor)
        if candidate < atr_ceiling:
            return None, f"SL_TOO_WIDE: OB boundary {ob_boundary} exceeds {max_atr_mult}×ATR cap"
        return candidate, "OK"
    candidate = ob_boundary if ob_boundary is not None else atr_floor
    if candidate <= entry:
        candidate = atr_floor
    candidate = max(candidate, atr_floor)
    if candidate > atr_ceiling:
        return None, f"SL_TOO_WIDE: OB boundary {ob_boundary} exceeds {max_atr_mult}×ATR cap"
    return candidate, "OK"


def trail_sl(
    position: ManagedPosition, current_price: float, atr14: float,
) -> float:
    distance = ATR_TRAIL_MULTIPLIER * atr14
    if position.direction == "BUY":
        anchor = max(position.trail_anchor or current_price, current_price)
        return anchor - distance
    anchor = min(position.trail_anchor or current_price, current_price)
    return anchor + distance


def evaluate_position(
    position: ManagedPosition,
    current_price: float,
    atr14_current: float,
    now: datetime | None = None,
) -> PositionAction:
    now = now or datetime.now(timezone.utc)
    r = r_multiple(position.entry, current_price, position.initial_sl, position.direction)

    if not position.breakeven_set and r >= 1.0:
        new_sl = breakeven_sl_price(position.entry, position.direction, position.tick_size)
        logger.info(
            f"BREAKEVEN: ticket={position.ticket} {position.symbol} {position.direction} "
            f"r={r:.2f} new_sl={new_sl:.{position.digits}f}"
        )
        return PositionAction(
            action="MOVE_TO_BREAKEVEN",
            new_sl=new_sl,
            reason=f"R={r:.2f} >= 1.0",
        )

    for stage in PARTIAL_CLOSE_STAGES:
        label = stage["label"]
        if label in position.stages_done:
            continue
        if r + 1e-9 >= float(stage["r_multiple"]):
            if stage["r_multiple"] >= 3.0:
                trail = trail_sl(position, current_price, atr14_current)
                logger.info(
                    f"TRAIL_START: ticket={position.ticket} {position.symbol} "
                    f"r={r:.2f} new_sl={trail:.{position.digits}f}"
                )
                return PositionAction(
                    action="TRAIL",
                    close_pct=float(stage["close_pct"]),
                    new_sl=trail,
                    reason=f"{label}: R={r:.2f}",
                )
            logger.info(
                f"PARTIAL_CLOSE: ticket={position.ticket} {position.symbol} "
                f"{label} pct={stage['close_pct']:.2f} r={r:.2f}"
            )
            return PositionAction(
                action="PARTIAL_CLOSE",
                close_pct=float(stage["close_pct"]),
                reason=f"{label}: R={r:.2f}",
            )

    hours_open = (now - position.open_time).total_seconds() / 3600
    if hours_open > TIME_EXIT_HOURS and r < TIME_EXIT_MIN_R:
        logger.info(
            f"TIME_EXIT: ticket={position.ticket} {position.symbol} "
            f"hours={hours_open:.1f} r={r:.2f}"
        )
        return PositionAction(
            action="TIME_EXIT",
            close_pct=1.0,
            reason=f"TIME_EXIT: {hours_open:.1f}h stale, R={r:.2f}",
        )

    return PositionAction(action="NOOP")


def apply_action(position: ManagedPosition, action: PositionAction) -> ManagedPosition:
    if action.action == "MOVE_TO_BREAKEVEN":
        position.breakeven_set = True
        if action.new_sl is not None:
            position.sl = action.new_sl
    elif action.action == "PARTIAL_CLOSE":
        label = action.reason.split(":")[0].strip()
        position.stages_done.add(label)
    elif action.action == "TRAIL":
        label = action.reason.split(":")[0].strip()
        position.stages_done.add(label)
        if action.new_sl is not None:
            position.sl = action.new_sl
            position.trail_anchor = action.new_sl
    return position
