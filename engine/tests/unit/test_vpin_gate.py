"""Test the VPIN gate is wired into the consensus precondition chain."""
from __future__ import annotations

from engine.strategy.consensus import (
    ClaudeResponse, CnnVote, Sources, State, evaluate,
)


def _good_sources() -> Sources:
    return Sources(
        smc="BUY", cnn=CnnVote("BUY", 80), rl="BUY",
        killzone_ok=True, news_clear=True,
        ofi="BUY", candle="BUY",
    )


def _gate(_ctx):
    return ClaudeResponse(decision="BUY", confidence=80, reasoning="ok", risk_adjustment=1.0)


def test_vpin_toxic_rejects_otherwise_perfect_signal():
    state = State(vpin_toxic=True)
    res = evaluate(state, _good_sources(), claude_gate=_gate)
    assert res.outcome == "REJECTED_VPIN_TOXIC"


def test_vpin_benign_passes_through():
    state = State(vpin_toxic=False)
    res = evaluate(state, _good_sources(), claude_gate=_gate)
    assert res.outcome == "EXECUTED"


def test_vpin_default_is_benign():
    state = State()  # default vpin_toxic = False
    res = evaluate(state, _good_sources(), claude_gate=_gate)
    assert res.outcome == "EXECUTED"
