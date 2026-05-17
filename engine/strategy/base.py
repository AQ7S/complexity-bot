"""Strategy base protocol — the contract every trading style implements.

A `Strategy` is a callable, side-effect-free signal producer. It receives
the current bar context for a single symbol and either returns a
`StrategySignal` (to be evaluated by consensus/risk) or `None`.

Strategies declare:
  * the timeframes they consume
  * which symbols they accept
  * their share of the daily risk budget
  * their minimum confluence requirement (passed through to consensus)
  * a maximum holding bar count
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Protocol


Vote = Literal["BUY", "SELL", "HOLD"]
StyleKind = Literal[
    "scalp", "day", "swing", "mean_reversion", "breakout", "carry",
]


@dataclass(frozen=True)
class StrategySignal:
    strategy_name: str
    symbol: str
    timeframe: str
    direction: Vote
    sl_price: float
    tp_price: float
    confidence: int
    reasoning: str
    rationale_tags: tuple[str, ...] = field(default_factory=tuple)
    intended_hold_bars: int = 0


@dataclass(frozen=True)
class StrategyContext:
    """Bag of inputs handed to `Strategy.detect()` per evaluation tick.

    Each field is optional — strategies pick what they need. The
    orchestrator builds the context once per tick from live state and
    fans it out to all active strategies.
    """
    symbol: str
    timeframe: str
    bars: object | None = None          # pd.DataFrame in practice
    ticks: list | None = None
    regime: str | None = None
    macro: dict | None = None
    open_positions: list | None = None
    vpin_score: float = 0.0
    spread_acceptable: bool = True
    killzone_ok: bool = True
    news_clear: bool = True
    h4_bias: str = "RANGING"


class Strategy(Protocol):
    name: str
    style: StyleKind
    timeframes: tuple[str, ...]
    symbols_whitelist: tuple[str, ...]
    risk_budget_pct: float
    min_confluence: int
    max_hold_bars: int

    def detect(self, ctx: StrategyContext) -> StrategySignal | None: ...

    def accepts_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols_whitelist or self.symbols_whitelist == ()
