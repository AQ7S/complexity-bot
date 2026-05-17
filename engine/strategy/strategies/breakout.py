"""Breakout strategy — M15 squeeze-release momentum, ATR breakout band."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.strategy.base import StrategyContext, StrategySignal


@dataclass
class BreakoutStrategy:
    name: str = "breakout"
    style: str = "breakout"
    timeframes: tuple[str, ...] = ("M15",)
    symbols_whitelist: tuple[str, ...] = ()
    risk_budget_pct: float = 0.015
    min_confluence: int = 3
    max_hold_bars: int = 16
    squeeze_lookback: int = 20

    def accepts_symbol(self, symbol: str) -> bool:
        return True

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.timeframe not in self.timeframes:
            return None
        if not ctx.spread_acceptable or not ctx.news_clear:
            return None
        if ctx.bars is None:
            return None
        try:
            high = ctx.bars["high"].tail(self.squeeze_lookback + 1).to_numpy()
            low = ctx.bars["low"].tail(self.squeeze_lookback + 1).to_numpy()
            close = ctx.bars["close"].tail(self.squeeze_lookback + 1).to_numpy()
        except Exception:  # noqa: BLE001
            return None
        if len(close) < self.squeeze_lookback + 1:
            return None
        # Squeeze release: current bar's high/low takes out the prior N-bar range.
        prev_hi = float(np.max(high[:-1]))
        prev_lo = float(np.min(low[:-1]))
        current_hi = float(high[-1])
        current_lo = float(low[-1])
        current_close = float(close[-1])
        rng = max(prev_hi - prev_lo, 1e-9)
        direction: str | None = None
        if current_hi > prev_hi and current_close > prev_hi:
            direction = "BUY"
        elif current_lo < prev_lo and current_close < prev_lo:
            direction = "SELL"
        if direction is None:
            return None
        atr = float(np.mean(high[1:] - low[1:]))
        if atr <= 0:
            return None
        sl_dist = max(rng * 0.5, atr)
        if direction == "BUY":
            sl = current_close - sl_dist
            tp = current_close + sl_dist * 2.0
        else:
            sl = current_close + sl_dist
            tp = current_close - sl_dist * 2.0
        return StrategySignal(
            strategy_name=self.name,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=direction,  # type: ignore[arg-type]
            sl_price=sl,
            tp_price=tp,
            confidence=68,
            reasoning=f"Squeeze release: range {rng:.5f} broken {direction}",
            rationale_tags=("squeeze_release", "breakout"),
            intended_hold_bars=self.max_hold_bars,
        )
