from engine.strategy.consensus import (
    ClaudeResponse, CnnVote, Sources, State, evaluate,
)


def _clean_state(**overrides) -> State:
    base = dict(
        is_paused=False, kill_active=False, spread_widened=False,
        open_positions=0, correlated_open=0,
    )
    base.update(overrides)
    return State(**base)


def _gate_ok(ctx):
    return ClaudeResponse(
        decision=ctx.get("__expect_dir", "BUY"),
        confidence=80, reasoning="ok", risk_adjustment=1.0,
    )


def test_ofi_can_supply_third_vote_when_rl_holds():
    src = Sources(
        smc="BUY", cnn=CnnVote("BUY", 80), rl="HOLD",
        killzone_ok=True, news_clear=True,
        ofi="BUY", candle="HOLD",
    )
    res = evaluate(_clean_state(), src, claude_gate=_gate_ok, claude_context={"__expect_dir": "BUY"})
    assert res.outcome == "EXECUTED", (res.outcome, res.reason)
    assert res.direction == "BUY"
    # smc+cnn+ofi agree + killzone + news_clear = 5; rl HOLD + candle HOLD
    assert res.confluence == 5, res.confluence


def test_candle_vote_as_seventh_source():
    src = Sources(
        smc="HOLD", cnn=CnnVote("BUY", 70), rl="BUY",
        killzone_ok=True, news_clear=True,
        ofi="HOLD", candle="BUY",
    )
    res = evaluate(_clean_state(), src, claude_gate=_gate_ok, claude_context={"__expect_dir": "BUY"})
    assert res.outcome == "EXECUTED"
    # cnn+rl+candle agree + killzone + news_clear = 5
    assert res.confluence == 5


def test_po3_adds_bonus_confluence_when_agreeing():
    src = Sources(
        smc="BUY", cnn=CnnVote("BUY", 75), rl="BUY",
        killzone_ok=True, news_clear=True,
        ofi="HOLD", candle="HOLD",
        po3="BUY",
    )
    res = evaluate(_clean_state(), src, claude_gate=_gate_ok, claude_context={"__expect_dir": "BUY"})
    assert res.outcome == "EXECUTED"
    # smc+cnn+rl agree (3) + killzone + news_clear (2) + po3 bonus (1) = 6
    assert res.confluence == 6


def test_po3_does_not_count_when_opposing():
    src = Sources(
        smc="BUY", cnn=CnnVote("BUY", 75), rl="HOLD",
        killzone_ok=True, news_clear=True,
        ofi="HOLD", candle="HOLD",
        po3="SELL",
    )
    res = evaluate(_clean_state(), src, claude_gate=_gate_ok, claude_context={"__expect_dir": "BUY"})
    assert res.outcome == "EXECUTED"
    # smc+cnn agree (2) + killzone + news (2) = 4. PO3 disagrees, no bonus.
    assert res.confluence == 4


def test_majority_uses_all_five_directional_votes():
    src = Sources(
        smc="SELL", cnn=CnnVote("SELL", 60), rl="BUY",
        killzone_ok=True, news_clear=True,
        ofi="BUY", candle="BUY",
    )
    res = evaluate(_clean_state(), src, claude_gate=_gate_ok, claude_context={"__expect_dir": "BUY"})
    # 3 BUY (rl+ofi+candle) vs 2 SELL (smc+cnn) → direction = BUY
    assert res.direction == "BUY", res.direction


def test_three_buy_three_sell_resolves_hold():
    # 2 BUY (smc+ofi) vs 2 SELL (cnn+candle), rl HOLD → tied → HOLD → reject
    src = Sources(
        smc="BUY", cnn=CnnVote("SELL", 65), rl="HOLD",
        killzone_ok=True, news_clear=True,
        ofi="BUY", candle="SELL",
    )
    res = evaluate(_clean_state(), src)
    assert res.outcome == "REJECTED_NO_DIRECTION", res.outcome
