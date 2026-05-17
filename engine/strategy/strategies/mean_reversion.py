"""Mean-reversion strategy — M15 z-score bands, tight 0.5× ATR SL."""
from __future__ import annotations

from dataclasses import dataclass

from engine.strategy.base import StrategyContext, StrategySignal
from engine.strategy.mean_reversion import avellaneda_stoikov_signal, target_price


@dataclass
class MeanReversionStrategy:
    name: str = "mean_reversion"
    style: str = "mean_reversion"
    timeframes: tuple[str, ...] = ("M15",)
    symbols_whitelist: tuple[str, ...] = ()
    risk_budget_pct: float = 0.01
    min_confluence: int = 2
    max_hold_bars: int = 12  # 3h on M15
    z_threshold: float = 2.0
    atr_sl_mult: float = 0.5

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
            closes = ctx.bars["close"].tail(80).to_numpy()
            high = ctx.bars["high"].tail(15).to_numpy()
            low = ctx.bars["low"].tail(15).to_numpy()
        except Exception:  # noqa: BLE001
            return None
        if len(closes) < 30 or len(high) < 14:
            return None
        sig = avellaneda_stoikov_signal(closes, lookback=60, z_threshold=self.z_threshold)
        if sig.direction == "HOLD":
            return None
        atr = float((high - low).mean())
        if atr <= 0:
            return None
        price = float(closes[-1])
        sl_dist = atr * self.atr_sl_mult
        tp = target_price(sig)
        if tp is None:
            return None
        sl = price - sl_dist if sig.direction == "BUY" else price + sl_dist
        return StrategySignal(
            strategy_name=self.name,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=sig.direction,
            sl_price=sl,
            tp_price=float(tp),
            confidence=65,
            reasoning=f"M15 z-score {sig.z_score:+.2f}",
            rationale_tags=("zscore", "M15_revert"),
            intended_hold_bars=self.max_hold_bars,
        )
