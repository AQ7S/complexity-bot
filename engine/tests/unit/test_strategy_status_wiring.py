"""Tests for the Tier 6 wiring: operator modes + snapshot frame."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from engine.ipc.messages import (
    PAYLOAD_TYPES,
    StrategyStatus,
    CmdStrategyToggle,
    envelope,
    parse,
)
from engine.strategy import orchestrator_runtime
from engine.strategy.base import StrategySignal
from engine.strategy.orchestrator import StrategyOrchestrator


@dataclass
class _S:
    name: str
    style: str = "test"
    timeframes: tuple[str, ...] = ("M5",)
    symbols_whitelist: tuple[str, ...] = ()
    risk_budget_pct: float = 0.01
    min_confluence: int = 3
    max_hold_bars: int = 10

    def accepts_symbol(self, _s: str) -> bool: return True
    def detect(self, _ctx): return None


@pytest.fixture(autouse=True)
def _reset_singleton():
    orchestrator_runtime.reset_for_tests()
    yield
    orchestrator_runtime.reset_for_tests()


def test_singleton_is_stable_across_calls():
    a = orchestrator_runtime.get_orchestrator()
    b = orchestrator_runtime.get_orchestrator()
    assert a is b


def test_set_mode_off_pauses_strategy():
    o = StrategyOrchestrator([_S("a")])
    assert o.set_mode("a", "OFF")
    assert o.health["a"].is_paused()


def test_set_mode_shadow_marks_shadow():
    o = StrategyOrchestrator([_S("a")])
    assert o.set_mode("a", "SHADOW")
    assert o.health["a"].is_shadow_only()


def test_set_mode_on_clears_breakers():
    o = StrategyOrchestrator([_S("a")])
    # Trip a breaker manually.
    o.health["a"].paused_until = time.time() + 9999
    assert o.set_mode("a", "ON")
    assert not o.health["a"].is_paused()


def test_set_mode_unknown_strategy_returns_false():
    o = StrategyOrchestrator([_S("a")])
    assert not o.set_mode("does_not_exist", "OFF")


def test_snapshot_payload_validates_against_pydantic_schema():
    o = StrategyOrchestrator([_S("a"), _S("b")])
    o.set_mode("a", "SHADOW")
    snap = o.snapshot()
    # Pydantic validation: the payload must round-trip through the typed model.
    model = StrategyStatus.model_validate(snap)
    assert len(model.strategies) == 2
    states = {f.name: f.state for f in model.strategies}
    assert states["a"] == "SHADOW"
    assert states["b"] == "ACTIVE"


def test_snapshot_includes_weights_sum_to_one():
    o = StrategyOrchestrator([_S("a"), _S("b"), _S("c")])
    snap = o.snapshot()
    weights = [f["weight"] for f in snap["strategies"]]
    assert abs(sum(weights) - 1.0) < 1e-6


def test_cmd_strategy_toggle_in_payload_registry():
    assert "cmd_strategy_toggle" in PAYLOAD_TYPES
    assert "strategy_status" in PAYLOAD_TYPES


def test_cmd_strategy_toggle_round_trip():
    env = envelope("cmd_strategy_toggle",
                   CmdStrategyToggle(name="scalping", mode="OFF"))
    t, model = parse(env)
    assert t == "cmd_strategy_toggle"
    assert isinstance(model, CmdStrategyToggle)
    assert model.name == "scalping"
    assert model.mode == "OFF"
