"""Tests for the strategy orchestrator (Tier 3.6)."""
from __future__ import annotations

from dataclasses import dataclass

from engine.strategy.base import StrategyContext, StrategySignal
from engine.strategy.orchestrator import (
    CIRCUIT_BREAKER_LOSS_STREAK,
    MAX_WEIGHT_CEILING,
    MIN_WEIGHT_FLOOR,
    StrategyOrchestrator,
)


@dataclass
class _FakeStrategy:
    name: str
    style: str = "x"
    timeframes: tuple[str, ...] = ("M5",)
    symbols_whitelist: tuple[str, ...] = ("EURUSD#",)
    risk_budget_pct: float = 0.01
    min_confluence: int = 3
    max_hold_bars: int = 10
    _emit: bool = True

    def accepts_symbol(self, symbol: str) -> bool:
        return symbol in self.symbols_whitelist

    def detect(self, ctx: StrategyContext) -> StrategySignal | None:
        if not self._emit:
            return None
        return StrategySignal(
            strategy_name=self.name, symbol=ctx.symbol,
            timeframe=ctx.timeframe, direction="BUY",
            sl_price=0.99, tp_price=1.01, confidence=60,
            reasoning="test", rationale_tags=("test",),
        )


def test_equal_sharpe_gives_equal_weights():
    o = StrategyOrchestrator([_FakeStrategy("a"), _FakeStrategy("b"), _FakeStrategy("c")])
    w = o.allocate_budget()
    assert all(abs(w[k] - 1/3) < 1e-6 for k in ("a", "b", "c"))


def test_dominant_sharpe_capped_by_ceiling():
    o = StrategyOrchestrator([_FakeStrategy("dom"), _FakeStrategy("b"), _FakeStrategy("c")])
    o.health["dom"].rolling_sharpe = 100.0
    o.health["b"].rolling_sharpe = 0.0
    o.health["c"].rolling_sharpe = 0.0
    w = o.allocate_budget()
    assert w["dom"] <= MAX_WEIGHT_CEILING + 1e-6


def test_zero_sharpe_gets_floor():
    o = StrategyOrchestrator([_FakeStrategy("good"), _FakeStrategy("bad")])
    o.health["good"].rolling_sharpe = 1.0
    o.health["bad"].rolling_sharpe = 0.0
    w = o.allocate_budget()
    assert w["bad"] >= MIN_WEIGHT_FLOOR - 1e-6


def test_tick_returns_signals_from_active_strategies():
    o = StrategyOrchestrator([_FakeStrategy("a"), _FakeStrategy("b")])
    contexts = [StrategyContext(symbol="EURUSD#", timeframe="M5")]
    res = o.tick(contexts)
    assert len(res.signals) == 2
    assert {s.strategy_name for s in res.signals} == {"a", "b"}


def test_circuit_breaker_pauses_after_loss_streak():
    o = StrategyOrchestrator([_FakeStrategy("a")])
    for _ in range(CIRCUIT_BREAKER_LOSS_STREAK):
        o.record_trade_close("a", pnl_usd=-10.0)
    assert o.health["a"].is_paused()
    res = o.tick([StrategyContext(symbol="EURUSD#", timeframe="M5")])
    assert "a" in res.skipped_paused
    assert len(res.signals) == 0


def test_record_winner_resets_loss_streak():
    o = StrategyOrchestrator([_FakeStrategy("a")])
    for _ in range(3):
        o.record_trade_close("a", pnl_usd=-1.0)
    o.record_trade_close("a", pnl_usd=+5.0)
    assert o.health["a"].consecutive_losses == 0


def test_weights_sum_to_one():
    o = StrategyOrchestrator([_FakeStrategy("a"), _FakeStrategy("b"), _FakeStrategy("c")])
    o.health["a"].rolling_sharpe = 1.5
    o.health["b"].rolling_sharpe = 0.4
    o.health["c"].rolling_sharpe = -0.2
    w = o.allocate_budget()
    assert abs(sum(w.values()) - 1.0) < 1e-6
