"""Scalping strategy — M1 / volume-bar, London-NY overlap only, 5-30s holds.

Uses Avellaneda–Stoikov style mean-reversion entries with a tight ATR-
multiplied stop. Symbol whitelist is intentionally short — only the
deepest, tightest-spread pairs/instruments where scalping has any chance
of clearing costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.strategy.base import StrategyContext, StrategySignal
from engine.strategy.mean_reversion import (
    avellaneda_stoikov_signal,
    compute_inventory_skew,
    target_price,
)


@dataclass
class ScalpingStrategy:
    name: str = "scalping"
    style: str = "scalp"
    timeframes: tuple[str, ...] = ("M1",)
    symbols_whitelist: tuple[str, ...] = ("EURUSD#", "GOLD#", "BTCUSD#")
    risk_budget_pct: float = 0.005
    min_confluence: int = 2
    max_hold_bars: int = 6  # ~6 M1 bars = 6 min cap
    z_threshold: float = 2.5
    atr_sl_mult: float = 0.5

    def accepts_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols_whitelist

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.timeframe not in self.timeframes:
            return None
        if not self.accepts_symbol(ctx.symbol):
            return None
        if not ctx.spread_acceptable or not ctx.news_clear:
            return None
        if ctx.bars is None:
            return None
        try:
            closes = ctx.bars["close"].tail(80).to_numpy()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return None
        if len(closes) < 30:
            return None
        inventory = compute_inventory_skew(ctx.open_positions or [])
        sig = avellaneda_stoikov_signal(
            closes, lookback=60, z_threshold=self.z_threshold,
            inventory=inventory,
        )
        if sig.direction == "HOLD":
            return None
        price = float(closes[-1])
        atr = self._estimate_atr(ctx.bars)
        if atr <= 0:
            return None
        sl_dist = max(atr * self.atr_sl_mult, sig.rolling_std)
        tp = target_price(sig)
        if tp is None:
            return None
        if sig.direction == "BUY":
            sl = price - sl_dist
        else:
            sl = price + sl_dist
        return StrategySignal(
            strategy_name=self.name,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=sig.direction,
            sl_price=sl,
            tp_price=float(tp),
            confidence=70,
            reasoning=(
                f"AS mean reversion z={sig.z_score:+.2f} on {len(closes)} M1 closes; "
                f"inventory={inventory:+.2f}"
            ),
            rationale_tags=("AS", "mean_revert"),
            intended_hold_bars=self.max_hold_bars,
        )

    @staticmethod
    def _estimate_atr(bars, period: int = 14) -> float:
        try:
            high = bars["high"].tail(period + 1).to_numpy()
            low = bars["low"].tail(period + 1).to_numpy()
            close = bars["close"].tail(period + 1).to_numpy()
        except Exception:  # noqa: BLE001
            return 0.0
        if len(high) <= period:
            return 0.0
        trs = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        return float(sum(trs) / len(trs))
