"""The 13 XM symbols traded by Complexity Engine.

Names match the IPC correlation_update example payload in the master plan
(Appendix D). XM's broker-side names are what `mt5.symbol_info(name)` expects;
some symbols use a `#` suffix on XM Global to denote alternate variants.
"""
from __future__ import annotations

from dataclasses import dataclass

# Assets that trade outside ICT kill zones (24/7 crypto + metals + indices).
_ALWAYS_ON = {"GOLD#", "BTCUSD#", "ETHUSD#", "AI_INDX#", "Crypto_10#"}


@dataclass(frozen=True)
class Symbol:
    name: str            # broker symbol id (passed to mt5.*)
    asset_class: str     # FX | METAL | CRYPTO | INDEX | EVENT
    pip_factor: int = 1  # placeholder; populated from mt5.symbol_info at runtime

    @property
    def always_on(self) -> bool:
        return self.name in _ALWAYS_ON


SYMBOLS_13: tuple[Symbol, ...] = (
    Symbol("EURUSD#",       "FX"),
    Symbol("USDJPY#",       "FX"),
    Symbol("GBPUSD#",       "FX"),
    Symbol("USDCHF#",       "FX"),
    Symbol("GOLD#",         "METAL"),
    Symbol("BTCUSD#",       "CRYPTO"),
    Symbol("ETHUSD#",       "CRYPTO"),
    Symbol("AI_INDX#",      "INDEX"),
    Symbol("Crypto_10#",    "INDEX"),
    Symbol("TrumpWinners#", "EVENT"),
    Symbol("HarrisWinners#", "EVENT"),
    Symbol("EURJPY#",       "FX"),
    Symbol("AUDUSD#",       "FX"),
)

SYMBOL_NAMES: tuple[str, ...] = tuple(s.name for s in SYMBOLS_13)


def is_always_on(symbol: str) -> bool:
    return symbol in _ALWAYS_ON
