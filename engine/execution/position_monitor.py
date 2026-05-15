"""1-second position monitor — partial close at 1:1 RR + 0.5×ATR trail.

The monitor is intentionally a coroutine: the engine main loop runs it under
asyncio so it cooperates with the WS broadcaster + tick streamer. MT5 calls
are synchronous; we wrap them with `asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import MetaTrader5 as mt5
from loguru import logger

from engine.config import settings
from engine.execution.order_router import DEVIATION_POINTS

POLL_INTERVAL_S = 1.0


@dataclass
class PositionState:
    ticket: int
    partial_taken: bool = False
    trail_anchor: float = 0.0   # the most extreme favorable price seen
    initial_sl_distance: float = 0.0
    last_atr: float = 0.0


@dataclass
class MonitorState:
    by_ticket: dict[int, PositionState] = field(default_factory=dict)


def _atr(symbol: str, period: int = 14) -> float:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        return 0.0
    highs = [r["high"] for r in rates]
    lows = [r["low"] for r in rates]
    closes = [r["close"] for r in rates]
    trs = []
    for i in range(1, len(rates)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


def _modify_position(ticket: int, *, sl: float | None = None, tp: float | None = None,
                     volume: float | None = None) -> bool:
    """Modify SL/TP. If `volume` is given, partial-close that volume first."""
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return False
    pos = pos_list[0]

    if volume is not None and 0 < volume < pos.volume:
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = float(tick.bid if is_buy else tick.ask)
        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(pos.ticket),
            "symbol": pos.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": DEVIATION_POINTS,
            "magic": int(pos.magic),
            "comment": "partial",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(payload)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning("partial close failed: {}", getattr(res, "comment", "?"))
            return False
        # reload position so subsequent SL/TP modify uses the remaining volume
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return True
        pos = pos_list[0]

    if sl is None and tp is None:
        return True
    payload = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": int(pos.ticket),
        "symbol": pos.symbol,
        "sl": float(sl if sl is not None else pos.sl),
        "tp": float(tp if tp is not None else pos.tp),
    }
    res = mt5.order_send(payload)
    return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE


def _step(state: MonitorState) -> None:
    positions = mt5.positions_get() or ()
    seen: set[int] = set()
    for pos in positions:
        seen.add(int(pos.ticket))
        ps = state.by_ticket.get(int(pos.ticket))
        if ps is None:
            entry = float(pos.price_open)
            sl0 = float(pos.sl) if pos.sl else entry  # no-SL trades aren't managed
            ps = PositionState(
                ticket=int(pos.ticket),
                initial_sl_distance=abs(entry - sl0),
                trail_anchor=float(pos.price_current),
                last_atr=_atr(pos.symbol),
            )
            state.by_ticket[int(pos.ticket)] = ps

        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        entry = float(pos.price_open)
        current = float(pos.price_current)
        sl = float(pos.sl) if pos.sl else entry
        moved = (current - entry) if is_buy else (entry - current)

        # 1) partial close at 1:1
        if not ps.partial_taken and ps.initial_sl_distance > 0 and moved >= ps.initial_sl_distance * settings.PARTIAL_CLOSE_RR:
            half_vol = round(pos.volume * settings.PARTIAL_CLOSE_FRACTION, 2)
            if half_vol > 0 and _modify_position(int(pos.ticket), volume=half_vol, sl=entry):
                ps.partial_taken = True
                logger.info("partial close + SL→BE for {} (ticket {})", pos.symbol, pos.ticket)

        # 2) trailing stop after partial taken
        if ps.partial_taken:
            ps.last_atr = ps.last_atr or _atr(pos.symbol)
            trail_dist = settings.TRAIL_ATR_MULTIPLIER * ps.last_atr
            if is_buy:
                ps.trail_anchor = max(ps.trail_anchor, current)
                desired_sl = ps.trail_anchor - trail_dist
                if desired_sl > sl:
                    _modify_position(int(pos.ticket), sl=desired_sl)
            else:
                ps.trail_anchor = min(ps.trail_anchor, current)
                desired_sl = ps.trail_anchor + trail_dist
                if desired_sl < sl or sl == 0:
                    _modify_position(int(pos.ticket), sl=desired_sl)

    # drop tickets that closed
    for tkt in list(state.by_ticket):
        if tkt not in seen:
            del state.by_ticket[tkt]


async def run(stop_event: asyncio.Event | None = None) -> None:
    state = MonitorState()
    while not (stop_event is not None and stop_event.is_set()):
        try:
            await asyncio.to_thread(_step, state)
        except Exception as e:  # noqa: BLE001
            logger.exception("position_monitor step raised: {}", e)
        await asyncio.sleep(POLL_INTERVAL_S)
