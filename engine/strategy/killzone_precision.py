from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal


Session = Literal["LONDON_OPEN", "NY_OPEN", "LONDON_NY_OVERLAP", "ASIAN", "TOKYO_OPEN", "SYDNEY_OPEN", "DEAD"]


SILVER_BULLET_WINDOWS_EST: dict[str, tuple[time, time]] = {
    "london_sb_1": (time(3, 0), time(4, 0)),
    "ny_sb_1":     (time(10, 0), time(11, 0)),
    "ny_sb_2":     (time(14, 0), time(15, 0)),
}

OVERLAP_START_EST = time(8, 0)
OVERLAP_END_EST = time(12, 0)

SESSION_WINDOWS_EST: dict[Session, tuple[time, time]] = {
    "ASIAN":        (time(19, 0), time(22, 0)),
    "LONDON_OPEN":  (time(2, 0), time(5, 0)),
    "NY_OPEN":      (time(7, 0), time(10, 0)),
    "LONDON_NY_OVERLAP": (time(8, 0), time(12, 0)),
    "TOKYO_OPEN":   (time(20, 0), time(23, 0)),
    "SYDNEY_OPEN":  (time(17, 0), time(20, 0)),
}

SYMBOL_OPTIMAL_SESSIONS: dict[str, tuple[str, ...]] = {
    "EURUSD#":       ("LONDON_OPEN", "NY_OPEN", "LONDON_NY_OVERLAP"),
    "GBPUSD#":       ("LONDON_OPEN", "NY_OPEN", "LONDON_NY_OVERLAP"),
    "USDJPY#":       ("LONDON_OPEN", "NY_OPEN", "TOKYO_OPEN"),
    "USDCHF#":       ("LONDON_OPEN", "NY_OPEN", "LONDON_NY_OVERLAP"),
    "GOLD#":         ("LONDON_OPEN", "NY_OPEN"),
    "BTCUSD#":       ("ALL",),
    "ETHUSD#":       ("ALL",),
    "AI_INDX#":      ("ALL",),
    "Crypto_10#":    ("ALL",),
    "EURJPY#":       ("LONDON_OPEN", "TOKYO_OPEN"),
    "AUDUSD#":       ("SYDNEY_OPEN", "TOKYO_OPEN", "LONDON_OPEN"),
    "TrumpWinners#": ("NY_OPEN",),
    "HarrisWinners#":("NY_OPEN",),
}

OVERLAP_LOT_BONUS = 1.2


@dataclass
class KillZoneContext:
    session: Session
    silver_bullet_active: bool
    overlap_active: bool
    optimal_session_for_symbol: bool
    lot_multiplier: float
    confluence_bonus: int
    confluence_penalty: int


def _between(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def silver_bullet_active(now_est: time) -> bool:
    return any(_between(now_est, s, e) for (s, e) in SILVER_BULLET_WINDOWS_EST.values())


def overlap_active(now_est: time) -> bool:
    return _between(now_est, OVERLAP_START_EST, OVERLAP_END_EST)


def current_session(now_est: time) -> Session:
    if _between(now_est, OVERLAP_START_EST, OVERLAP_END_EST):
        return "LONDON_NY_OVERLAP"
    for sess in ("LONDON_OPEN", "NY_OPEN", "ASIAN", "TOKYO_OPEN", "SYDNEY_OPEN"):
        s, e = SESSION_WINDOWS_EST[sess]
        if _between(now_est, s, e):
            return sess
    return "DEAD"


def is_optimal_session(symbol: str, session: Session) -> bool:
    optimal = SYMBOL_OPTIMAL_SESSIONS.get(symbol, ())
    if "ALL" in optimal:
        return True
    return session in optimal


def kill_zone_context(symbol: str, now_est: datetime) -> KillZoneContext:
    t = now_est.time()
    sess = current_session(t)
    sb = silver_bullet_active(t)
    ov = overlap_active(t)
    optimal = is_optimal_session(symbol, sess)
    lot_mult = OVERLAP_LOT_BONUS if ov else 1.0
    bonus = 1 if sb else 0
    penalty = 0 if optimal or "ALL" in SYMBOL_OPTIMAL_SESSIONS.get(symbol, ("ALL",)) else 1
    return KillZoneContext(
        session=sess,
        silver_bullet_active=sb,
        overlap_active=ov,
        optimal_session_for_symbol=optimal,
        lot_multiplier=lot_mult,
        confluence_bonus=bonus,
        confluence_penalty=penalty,
    )
