"""PairsTradingStrategy — cointegration mean-reversion on FX pairs.

Pre-declared candidate pairs (the only ones that historically show
stable cointegration on H1+):
    EUR/USD ↔ GBP/USD
    USD/JPY ↔ USD/CHF
    GOLD    ↔ USD/JPY (negative beta)

The strategy refits cointegration every `refit_every_bars` bars; if the
relationship breaks (ADF can no longer reject unit root in the residuals)
the pair is dropped from the active set until the next refit confirms it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from engine.strategy.base import StrategyContext, StrategySignal
from engine.strategy.cointegration import (
    CointegrationResult,
    engle_granger_cointegration,
    pairs_trade_signal,
)


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("EURUSD#", "GBPUSD#"),
    ("USDJPY#", "USDCHF#"),
    ("GOLD#",   "USDJPY#"),
)


@dataclass
class PairsTradingStrategy:
    name: str = "pairs_trading"
    style: str = "pairs"
    timeframes: tuple[str, ...] = ("H1",)
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    risk_budget_pct: float = 0.005
    min_confluence: int = 2
    max_hold_bars: int = 48
    refit_every_bars: int = 24
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5

    # Internal cached models per pair: {(y, x): (CointegrationResult, bars_since_refit)}
    _models: dict[tuple[str, str], tuple[CointegrationResult, int]] = field(default_factory=dict)
    _price_cache: dict[str, list[float]] = field(default_factory=dict)

    @property
    def symbols_whitelist(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for y, x in self.pairs:
            for s in (y, x):
                if s not in seen:
                    seen.add(s); out.append(s)
        return tuple(out)

    def accepts_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols_whitelist

    def feed_close(self, symbol: str, close: float) -> None:
        """The orchestrator can call this with each H1 close to keep the
        per-symbol cache fresh; alternatively `detect()` accepts a
        pre-populated `ctx.bars` and pulls the closes itself."""
        buf = self._price_cache.setdefault(symbol, [])
        buf.append(float(close))
        if len(buf) > 1000:
            del buf[:200]

    def _ensure_model(self, pair: tuple[str, str]) -> CointegrationResult | None:
        cached = self._models.get(pair)
        if cached is not None and cached[1] < self.refit_every_bars:
            return cached[0]
        y_close = np.array(self._price_cache.get(pair[0], []), dtype=np.float64)
        x_close = np.array(self._price_cache.get(pair[1], []), dtype=np.float64)
        n = min(y_close.size, x_close.size)
        if n < 100:
            return None
        coint = engle_granger_cointegration(y_close[-n:], x_close[-n:])
        self._models[pair] = (coint, 0)
        return coint

    def _bump_bar_count(self) -> None:
        for k, (m, n) in list(self._models.items()):
            self._models[k] = (m, n + 1)

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if ctx.timeframe not in self.timeframes:
            return None
        if not self.accepts_symbol(ctx.symbol):
            return None
        if ctx.bars is None:
            return None
        try:
            close = float(ctx.bars["close"].iloc[-1])  # type: ignore[index]
        except Exception:  # noqa: BLE001
            return None
        self.feed_close(ctx.symbol, close)
        self._bump_bar_count()
        # Try every pair this symbol is part of.
        for y_sym, x_sym in self.pairs:
            if ctx.symbol not in (y_sym, x_sym):
                continue
            coint = self._ensure_model((y_sym, x_sym))
            if coint is None or not coint.is_cointegrated:
                continue
            y_close = self._price_cache.get(y_sym, [])
            x_close = self._price_cache.get(x_sym, [])
            if not y_close or not x_close:
                continue
            sig = pairs_trade_signal(
                np.array(y_close[-1:]), np.array(x_close[-1:]), coint,
                z_entry=self.z_entry, z_exit=self.z_exit, z_stop=self.z_stop,
            )
            if sig.side == "FLAT":
                continue
            # Only emit on the *y* leg (avoid double-firing); the order
            # router can use sig.hedge_ratio to compute the x lot.
            if ctx.symbol != y_sym:
                continue
            direction = "BUY" if sig.side == "LONG_Y_SHORT_X" else "SELL"
            price = float(y_close[-1])
            spread_sigma = max(coint.spread_std, 1e-9)
            sl = price - spread_sigma * 2.0 if direction == "BUY" else price + spread_sigma * 2.0
            tp = price + spread_sigma * 0.5 if direction == "BUY" else price - spread_sigma * 0.5
            return StrategySignal(
                strategy_name=self.name,
                symbol=ctx.symbol,
                timeframe=ctx.timeframe,
                direction=direction,  # type: ignore[arg-type]
                sl_price=sl,
                tp_price=tp,
                confidence=65,
                reasoning=(
                    f"cointegrated with {x_sym} (β={coint.hedge_ratio:.3f}, "
                    f"z={sig.z_score:+.2f}); mean-revert {sig.side.lower()}"
                ),
                rationale_tags=("cointegration", "pairs", x_sym),
                intended_hold_bars=self.max_hold_bars,
            )
        return None
