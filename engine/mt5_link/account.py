"""Account-level queries (equity, balance, margin)."""
from __future__ import annotations

from dataclasses import dataclass

import MetaTrader5 as mt5


@dataclass(frozen=True)
class AccountSnapshot:
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    open_positions: int

    @property
    def drawdown_pct(self) -> float:
        if self.balance <= 0:
            return 0.0
        return max(0.0, (self.balance - self.equity) / self.balance)


def snapshot() -> AccountSnapshot:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"mt5.account_info returned None: {mt5.last_error()}")
    positions = mt5.positions_get() or ()
    return AccountSnapshot(
        login=int(info.login),
        server=str(info.server),
        currency=str(info.currency),
        balance=float(info.balance),
        equity=float(info.equity),
        margin=float(info.margin),
        free_margin=float(info.margin_free),
        margin_level=float(info.margin_level or 0.0),
        open_positions=len(positions),
    )
