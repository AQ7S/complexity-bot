"""MT5 order dispatch — market / pending limit / pending stop.

The router turns a `Sources` outcome into an MT5 `order_send` request and
records the resulting trade in the SQLite journal. It also exposes helpers
for closing a single position and for force-closing all positions
(used by the emergency-close path in risk/manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import MetaTrader5 as mt5
from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal, _now_iso
from engine.risk.lot_calc import LotResult, SymbolInfo, compute_lot

DirectionT = Literal["BUY", "SELL"]
OrderTypeT = Literal["MARKET", "LIMIT", "STOP"]

DEVIATION_POINTS = 20  # max slippage allowed by broker, in points


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: DirectionT
    lot: float
    sl: float
    tp: float
    order_type: OrderTypeT = "MARKET"
    price: float | None = None       # required for LIMIT / STOP
    comment: str = "complexity-engine"
    magic: int = 1_080_808
    signal_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    ticket: int | None
    retcode: int
    comment: str
    request: OrderRequest


def _symbol_info_from_mt5(name: str) -> SymbolInfo:
    info = mt5.symbol_info(name)
    if info is None:
        raise RuntimeError(f"symbol {name!r} not found on broker")
    return SymbolInfo(
        name=name,
        point=float(info.point),
        digits=int(info.digits),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        contract_size=float(getattr(info, "trade_contract_size", 100_000.0)),
    )


def size_lot_for(
    symbol: str, *, entry: float, sl_price: float,
    risk_pct: float = settings.RISK_PCT_PER_TRADE,
    K: float = 1.0,
    equity: float | None = None,
) -> LotResult:
    """Convenience wrapper over `compute_lot` that pulls live broker data."""
    sym = _symbol_info_from_mt5(symbol)
    if equity is None:
        acct = mt5.account_info()
        if acct is None:
            raise RuntimeError("mt5.account_info() returned None")
        equity = float(acct.equity)
    return compute_lot(
        equity=equity, entry=entry, sl_price=sl_price, symbol=sym,
        risk_pct=risk_pct, claude_risk_adjustment=K,
    )


def _build_request(req: OrderRequest) -> dict:
    is_buy = req.direction == "BUY"
    type_map = {
        ("MARKET", True):  mt5.ORDER_TYPE_BUY,
        ("MARKET", False): mt5.ORDER_TYPE_SELL,
        ("LIMIT", True):   mt5.ORDER_TYPE_BUY_LIMIT,
        ("LIMIT", False):  mt5.ORDER_TYPE_SELL_LIMIT,
        ("STOP", True):    mt5.ORDER_TYPE_BUY_STOP,
        ("STOP", False):   mt5.ORDER_TYPE_SELL_STOP,
    }
    order_type = type_map[(req.order_type, is_buy)]

    if req.order_type == "MARKET":
        tick = mt5.symbol_info_tick(req.symbol)
        if tick is None:
            raise RuntimeError(f"no tick for {req.symbol}")
        price = float(tick.ask if is_buy else tick.bid)
        action = mt5.TRADE_ACTION_DEAL
    else:
        if req.price is None:
            raise ValueError(f"{req.order_type} order requires `price`")
        price = float(req.price)
        action = mt5.TRADE_ACTION_PENDING

    return {
        "action": action,
        "symbol": req.symbol,
        "volume": float(req.lot),
        "type": order_type,
        "price": price,
        "sl": float(req.sl),
        "tp": float(req.tp),
        "deviation": DEVIATION_POINTS,
        "magic": req.magic,
        "comment": req.comment[:31],   # MT5 caps the comment at 31 chars
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }


def send_order(req: OrderRequest) -> OrderResult:
    """Dispatch an order via mt5.order_send. Logs the trade on success."""
    if not mt5.symbol_info(req.symbol).visible:
        mt5.symbol_select(req.symbol, True)
    payload = _build_request(req)
    res = mt5.order_send(payload)
    if res is None:
        code, desc = mt5.last_error()
        logger.error("order_send returned None: ({}) {}", code, desc)
        return OrderResult(ok=False, ticket=None, retcode=code, comment=desc, request=req)
    ok = res.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        logger.warning("order_send retcode={} comment={!r}", res.retcode, res.comment)
        return OrderResult(ok=False, ticket=None, retcode=res.retcode, comment=res.comment, request=req)

    ticket = int(res.order or res.deal or 0)
    _log_trade(req, fill_price=float(res.price), ticket=ticket)
    return OrderResult(ok=True, ticket=ticket, retcode=res.retcode, comment=res.comment, request=req)


def _log_trade(req: OrderRequest, *, fill_price: float, ticket: int) -> None:
    with open_journal() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO trades
              (mt5_ticket, symbol, direction, entry_price, lot_size, sl, tp, open_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket, req.symbol, req.direction, fill_price, req.lot, req.sl, req.tp, _now_iso()),
        )
        con.commit()


# ----------------------------------------------------------------------------
# Closing helpers
# ----------------------------------------------------------------------------

def close_position(ticket: int, *, reason: str = "MANUAL") -> bool:
    """Close a single position by ticket. Records the outcome in SQLite."""
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        logger.info("close_position: no open position with ticket {}", ticket)
        return False
    pos = pos_list[0]
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return False
    price = float(tick.bid if is_buy else tick.ask)
    payload = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": int(pos.ticket),
        "symbol": pos.symbol,
        "volume": float(pos.volume),
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": DEVIATION_POINTS,
        "magic": int(pos.magic),
        "comment": f"close:{reason}"[:31],
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    res = mt5.order_send(payload)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        pnl = float(pos.profit) + float(pos.swap or 0) + float(pos.commission or 0)
        with open_journal() as con:
            con.execute(
                """
                UPDATE trades
                SET exit_price=?, pnl=?, close_time=?, close_reason=?
                WHERE mt5_ticket=?
                """,
                (price, pnl, _now_iso(), reason, int(pos.ticket)),
            )
            con.commit()
    else:
        logger.warning("close_position {} failed: {}", ticket,
                       getattr(res, "comment", mt5.last_error()))
    return ok


def close_all(*, reason: str = "MANUAL") -> dict[str, int]:
    """Close every open position (engine magic or otherwise). Returns counts."""
    positions = mt5.positions_get() or ()
    closed = 0
    failed = 0
    for pos in positions:
        if close_position(int(pos.ticket), reason=reason):
            closed += 1
        else:
            failed += 1
    logger.info("close_all reason={}: closed={} failed={}", reason, closed, failed)
    return {"closed": closed, "failed": failed, "total": len(positions)}
