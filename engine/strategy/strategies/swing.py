"""Swing strategy — H1+H4 timeframe, 1-5 day hold, SMC-driven."""
from __future__ import annotations

from dataclasses import dataclass

from engine.strategy.base import StrategyContext, StrategySignal


@dataclass
class SwingStrategy:
    name: str = "swing"
    style: str = "swing"
    timeframes: tuple[str, ...] = ("H1", "H4")
    symbols_whitelist: tuple[str, ...] = ()
    risk_budget_pct: float = 0.01
    min_confluence: int = 3
    max_hold_bars: int = 24 * 5  # 5 days on H1
    h4_bias_required: bool = True

    def accepts_symbol(self, symbol: str) -> bool:
        return True

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.timeframe not in self.timeframes:
            return None
        if not ctx.spread_acceptable:
            return None
        if ctx.bars is None:
            return None
        try:
            closes = ctx.bars["close"].tail(20).to_numpy()
        except Exception:  # noqa: BLE001
            return None
        if len(closes) < 5:
            return None
        # Simple swing entry: align with H4 bias when present.
        bias = (ctx.h4_bias or "").upper()
        direction: str | None = None
        if "UP" in bias or "BULL" in bias:
            direction = "BUY"
        elif "DOWN" in bias or "BEAR" in bias:
            direction = "SELL"
        if self.h4_bias_required and direction is None:
            return None
        if direction is None:
            return None
        price = float(closes[-1])
        rng = float(closes.max() - closes.min())
        if rng <= 0:
            return None
        sl = price - rng * 1.5 if direction == "BUY" else price + rng * 1.5
        tp = price + rng * 3.0 if direction == "BUY" else price - rng * 3.0
        return StrategySignal(
            strategy_name=self.name,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=direction,  # type: ignore[arg-type]
            sl_price=sl,
            tp_price=tp,
            confidence=60,
            reasoning=f"Aligned with H4 bias '{bias}' on {ctx.timeframe}",
            rationale_tags=("h4_bias",),
            intended_hold_bars=self.max_hold_bars,
        )
