"""Carry strategy — D1, positive-swap pairs only, hold until swap flips negative."""
from __future__ import annotations

from dataclasses import dataclass

from engine.strategy.base import StrategyContext, StrategySignal


# Approximate XM demo swap directions (pips/night). Positive → favours longs.
SWAP_LONG_BIAS: dict[str, float] = {
    "AUDUSD#":   0.10,
    "EURJPY#":   0.40,
    "GBPUSD#":   0.20,
    "GOLD#":    -0.50,
    "USDJPY#":   0.30,
}
SWAP_SHORT_BIAS: dict[str, float] = {
    "EURUSD#":   0.20,
    "USDCHF#":   0.10,
}


@dataclass
class CarryStrategy:
    name: str = "carry"
    style: str = "carry"
    timeframes: tuple[str, ...] = ("D1",)
    symbols_whitelist: tuple[str, ...] = tuple(
        set(SWAP_LONG_BIAS) | set(SWAP_SHORT_BIAS)
    )
    risk_budget_pct: float = 0.005
    min_confluence: int = 2
    max_hold_bars: int = 30
    min_swap_pips: float = 0.10

    def accepts_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols_whitelist

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.timeframe not in self.timeframes:
            return None
        if not self.accepts_symbol(ctx.symbol):
            return None
        if ctx.bars is None:
            return None
        try:
            close = ctx.bars["close"].tail(20).to_numpy()
        except Exception:  # noqa: BLE001
            return None
        if len(close) < 5:
            return None
        long_bias = SWAP_LONG_BIAS.get(ctx.symbol, 0.0)
        short_bias = SWAP_SHORT_BIAS.get(ctx.symbol, 0.0)
        if long_bias >= self.min_swap_pips and long_bias > short_bias:
            direction = "BUY"
            edge = long_bias
        elif short_bias >= self.min_swap_pips and short_bias > long_bias:
            direction = "SELL"
            edge = short_bias
        else:
            return None
        price = float(close[-1])
        trend_filter = float(close[-1] - close[0])
        if direction == "BUY" and trend_filter < 0:
            return None
        if direction == "SELL" and trend_filter > 0:
            return None
        rng = float(close.max() - close.min())
        if rng <= 0:
            return None
        sl = price - rng if direction == "BUY" else price + rng
        tp = price + rng * 4.0 if direction == "BUY" else price - rng * 4.0
        return StrategySignal(
            strategy_name=self.name,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=direction,  # type: ignore[arg-type]
            sl_price=sl,
            tp_price=tp,
            confidence=55,
            reasoning=f"Positive swap {edge:+.2f}p/night aligned with 20-day trend",
            rationale_tags=("carry", "swap_pos"),
            intended_hold_bars=self.max_hold_bars,
        )
