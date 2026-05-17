"""Maximum Adverse Excursion (MAE) + Maximum Favorable Excursion (MFE) tracker.

For every open position the tracker maintains:
  * the worst unrealised loss seen so far (MAE), in pips
  * the best unrealised profit seen so far (MFE), in pips
  * the seconds it took to reach each extreme since entry

Aggregating these over closed trades reveals systematic SL/TP placement
bugs: e.g. "BUY signals with PO3=BUY consistently see MAE > 1.2× SL —
your SL is too tight on PO3 setups."

The tracker persists per-trade extremes to `trade_excursions` (sqlite).
Aggregation queries (top-by-MAE, MAE vs. SL ratio per signal source)
live in the dashboard / reporting layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from engine.data.sqlite_journal import open_journal


@dataclass
class ExcursionState:
    trade_id: int
    entry_price: float
    direction: str
    entry_ts: float
    point_size: float = 0.0001
    max_mae_pips: float = 0.0     # absolute pips against the trade
    max_mfe_pips: float = 0.0     # absolute pips in favour of the trade
    time_to_mae_s: int = 0
    time_to_mfe_s: int = 0
    last_price: float = 0.0


class ExcursionTracker:
    """In-process tracker; persists final state on close."""

    def __init__(self) -> None:
        self._states: dict[int, ExcursionState] = {}

    def open(
        self,
        *,
        trade_id: int,
        entry_price: float,
        direction: str,
        point_size: float = 0.0001,
        now: Optional[float] = None,
    ) -> None:
        ts = now if now is not None else time.time()
        self._states[trade_id] = ExcursionState(
            trade_id=trade_id, entry_price=float(entry_price),
            direction=direction.upper(), entry_ts=ts,
            point_size=float(point_size) or 0.0001,
            last_price=float(entry_price),
        )

    def update(self, trade_id: int, current_price: float, *, now: float | None = None) -> None:
        s = self._states.get(trade_id)
        if s is None:
            return
        ts = now if now is not None else time.time()
        s.last_price = float(current_price)
        adv = (current_price - s.entry_price) if s.direction == "BUY" else (s.entry_price - current_price)
        pips = abs(adv) / s.point_size
        if adv < 0 and pips > s.max_mae_pips:
            s.max_mae_pips = pips
            s.time_to_mae_s = int(ts - s.entry_ts)
        elif adv > 0 and pips > s.max_mfe_pips:
            s.max_mfe_pips = pips
            s.time_to_mfe_s = int(ts - s.entry_ts)

    def close(self, trade_id: int, *, db_path: str | None = None) -> ExcursionState | None:
        s = self._states.pop(trade_id, None)
        if s is None:
            return None
        with open_journal(db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO trade_excursions "
                "(trade_id, max_mae_pips, max_mfe_pips, time_to_mae_s, time_to_mfe_s) "
                "VALUES (?, ?, ?, ?, ?)",
                (int(s.trade_id), float(s.max_mae_pips), float(s.max_mfe_pips),
                 int(s.time_to_mae_s), int(s.time_to_mfe_s)),
            )
            con.commit()
        return s

    def snapshot(self, trade_id: int) -> ExcursionState | None:
        return self._states.get(trade_id)


def load_excursion(trade_id: int, *, db_path: str | None = None) -> ExcursionState | None:
    with open_journal(db_path) as con:
        row = con.execute(
            "SELECT trade_id, max_mae_pips, max_mfe_pips, time_to_mae_s, time_to_mfe_s "
            "FROM trade_excursions WHERE trade_id=?",
            (int(trade_id),),
        ).fetchone()
    if row is None:
        return None
    return ExcursionState(
        trade_id=int(row["trade_id"]),
        entry_price=0.0, direction="?", entry_ts=0.0,
        max_mae_pips=float(row["max_mae_pips"] or 0.0),
        max_mfe_pips=float(row["max_mfe_pips"] or 0.0),
        time_to_mae_s=int(row["time_to_mae_s"] or 0),
        time_to_mfe_s=int(row["time_to_mfe_s"] or 0),
    )
