"""Day-trading strategy — M5 timeframe, the existing 7-vote consensus stack.

This strategy is a thin wrapper that hands off to the existing consensus
engine. It exists so the orchestrator can budget its risk allocation
alongside the other styles.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.strategy.base import StrategyContext, StrategySignal


@dataclass
class DayTradingStrategy:
    name: str = "day_trading"
    style: str = "day"
    timeframes: tuple[str, ...] = ("M5",)
    symbols_whitelist: tuple[str, ...] = ()  # accept all
    risk_budget_pct: float = 0.02
    min_confluence: int = 3
    max_hold_bars: int = 48  # ~4h cap

    def accepts_symbol(self, symbol: str) -> bool:
        return True

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        # The full day-trading evaluation lives in `engine.strategy.consensus`
        # already wired into the main loop. This strategy only flags
        # *eligibility* — the consensus path produces the actual signal.
        if ctx.timeframe not in self.timeframes:
            return None
        if not ctx.spread_acceptable:
            return None
        if not ctx.killzone_ok or not ctx.news_clear:
            return None
        return None  # No standalone signal — consensus handles
