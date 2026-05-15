from datetime import datetime, timedelta, timezone

from engine.execution.position_manager import (
    ManagedPosition, PARTIAL_CLOSE_STAGES, breakeven_sl_price, compute_initial_sl,
    evaluate_position, r_multiple,
)
from engine.execution.execution_quality import (
    recommend_entry, slippage_acceptable, slippage_pips,
)
from engine.features.smc_filters import (
    compute_ote_zone, evaluate_smc_filters, fvg_ob_confluence, h4_bias_aligned,
    ob_freshness_label, OBZone, premium_discount_state, price_in_ote, Zone,
)
from engine.learning.performance_tracker import PerformanceTracker, TradeOutcome
from engine.risk.correlation_guard import OpenPositionRef, position_allowed
from engine.strategy.claude_gate import build_rich_context, hard_reject_check
from engine.strategy.killzone_precision import (
    current_session, is_optimal_session, kill_zone_context, overlap_active,
    silver_bullet_active,
)


def test_r_multiple() -> None:
    assert r_multiple(1.10, 1.11, 1.09, "BUY") == 1.0
    assert r_multiple(1.10, 1.09, 1.11, "SELL") == 1.0
    assert r_multiple(1.10, 1.12, 1.09, "BUY") == 2.0


def test_breakeven_moves_sl_at_1r() -> None:
    pos = ManagedPosition(
        ticket=1, symbol="EURUSD#", direction="BUY",
        entry=1.10, sl=1.09, initial_sl=1.09, tp=1.12, lot=0.1,
        open_time=datetime.now(timezone.utc), atr14_at_entry=0.001,
        tick_size=0.00001, digits=5,
    )
    action = evaluate_position(pos, 1.11, 0.001)
    assert action.action == "MOVE_TO_BREAKEVEN"
    assert action.new_sl is not None and action.new_sl > pos.entry


def test_partial_close_stages_fire_in_order() -> None:
    pos = ManagedPosition(
        ticket=1, symbol="EURUSD#", direction="BUY",
        entry=1.10, sl=1.09, initial_sl=1.09, tp=1.13, lot=0.1,
        open_time=datetime.now(timezone.utc), atr14_at_entry=0.001,
        tick_size=0.00001, digits=5, breakeven_set=True,
    )
    a1 = evaluate_position(pos, 1.11, 0.001)
    assert a1.action == "PARTIAL_CLOSE"
    assert abs(a1.close_pct - 0.33) < 1e-9
    pos.stages_done.add("STAGE_1_1R")
    a2 = evaluate_position(pos, 1.12, 0.001)
    assert a2.action == "PARTIAL_CLOSE"
    pos.stages_done.add("STAGE_2_2R")
    a3 = evaluate_position(pos, 1.13, 0.001)
    assert a3.action == "TRAIL"
    assert a3.new_sl is not None


def test_time_exit_after_4h_low_r() -> None:
    open_t = datetime.now(timezone.utc) - timedelta(hours=5)
    pos = ManagedPosition(
        ticket=1, symbol="EURUSD#", direction="BUY",
        entry=1.10, sl=1.09, initial_sl=1.09, tp=1.13, lot=0.1,
        open_time=open_t, atr14_at_entry=0.001, tick_size=0.00001, digits=5,
        breakeven_set=True,
    )
    a = evaluate_position(pos, 1.103, 0.001)
    assert a.action == "TIME_EXIT"
    assert a.close_pct == 1.0


def test_compute_initial_sl_uses_ob_when_within_atr_band() -> None:
    sl, msg = compute_initial_sl(entry=1.10, ob_boundary=1.099, atr14=0.001, direction="BUY")
    assert msg == "OK" and sl is not None and sl < 1.10


def test_compute_initial_sl_rejects_when_ob_too_wide() -> None:
    sl, msg = compute_initial_sl(entry=1.10, ob_boundary=1.090, atr14=0.001, direction="BUY", max_atr_mult=3.0)
    assert sl is None and msg.startswith("SL_TOO_WIDE")


def test_correlation_blocks_third_in_group() -> None:
    opens = [
        OpenPositionRef("EURUSD#", "BUY"),
        OpenPositionRef("GBPUSD#", "BUY"),
    ]
    v = position_allowed("USDCHF#", "BUY", opens)
    assert v.allowed is False
    assert "CORRELATION_BLOCK" in v.reason


def test_correlation_blocks_hedge() -> None:
    opens = [OpenPositionRef("EURUSD#", "BUY")]
    v = position_allowed("GBPUSD#", "SELL", opens)
    assert v.allowed is False
    assert "HEDGE_BLOCK" in v.reason


def test_correlation_allows_independent_groups() -> None:
    opens = [
        OpenPositionRef("EURUSD#", "BUY"),
        OpenPositionRef("GBPUSD#", "BUY"),
    ]
    v = position_allowed("USDJPY#", "BUY", opens)
    assert v.allowed is True


def test_performance_tracker_low_winrate_halves_size() -> None:
    pt = PerformanceTracker()
    now = datetime.now(timezone.utc)
    for i in range(15):
        pnl = 1.0 if i < 3 else -1.0
        pt.record_trade(TradeOutcome("GOLD#", pnl, "NY_OPEN", now))
    assert pt.size_multiplier("GOLD#") == 0.5


def test_circuit_breaker_after_3_losses() -> None:
    pt = PerformanceTracker()
    now = datetime.now(timezone.utc)
    pt.record_trade(TradeOutcome("EURUSD#", -1.0, "NY_OPEN", now))
    pt.record_trade(TradeOutcome("GBPUSD#", -1.0, "NY_OPEN", now))
    pt.record_trade(TradeOutcome("GOLD#",   -1.0, "NY_OPEN", now))
    active, reason = pt.circuit_breaker_active(now)
    assert active is True
    assert "CIRCUIT_BREAK" in reason


def test_premium_discount_classifier() -> None:
    assert premium_discount_state(1.02, 1.10, 1.00) == "DISCOUNT"
    assert premium_discount_state(1.09, 1.10, 1.00) == "PREMIUM"
    assert premium_discount_state(1.050, 1.06, 1.04) == "EQUILIBRIUM"


def test_ote_zone_for_buy_and_sell() -> None:
    lo, hi = compute_ote_zone(1.00, 1.10, "BUY")
    assert 1.020 < lo < 1.05 and 1.030 < hi < 1.05
    lo2, hi2 = compute_ote_zone(1.00, 1.10, "SELL")
    assert price_in_ote((lo2 + hi2) / 2, lo2, hi2)


def test_ob_freshness_labels() -> None:
    assert ob_freshness_label(0) == "FRESH"
    assert ob_freshness_label(1) == "TAPPED_ONCE"
    assert ob_freshness_label(2) == "CONSUMED"


def test_fvg_ob_confluence_detects_overlap() -> None:
    ob = OBZone(high=1.10, low=1.099, touches=0)
    fvg = Zone(high=1.0996, low=1.0992)
    assert fvg_ob_confluence(1.0995, [ob], [fvg]) is True


def test_h4_alignment_bullish() -> None:
    highs = [1.05, 1.06, 1.07, 1.08]
    lows  = [1.04, 1.05, 1.06, 1.07]
    aligned, bias, _ = h4_bias_aligned("BUY", highs, lows, adx_h4=30.0)
    assert aligned is True and bias == "BULLISH"


def test_smc_filters_full_pipeline_accepts_clean_setup() -> None:
    ob = OBZone(high=1.0650, low=1.0640, touches=0)
    fvg = Zone(high=1.0648, low=1.0642)
    res = evaluate_smc_filters(
        direction="BUY",
        current_price=1.0645,
        session_high=1.10, session_low=1.05,
        swing_high=1.10, swing_low=1.05,
        ob_zones=[ob], fvg_zones=[fvg],
        h4_highs=[1.05, 1.06, 1.07, 1.08],
        h4_lows=[1.04, 1.05, 1.06, 1.07],
        adx_h4=30.0,
    )
    assert res.allow is True, res.reason
    assert res.ote_active is True
    assert res.fvg_ob_confluence is True


def test_smc_filters_reject_premium_buy() -> None:
    res = evaluate_smc_filters(
        direction="BUY",
        current_price=1.095,
        session_high=1.10, session_low=1.05,
        swing_high=1.10, swing_low=1.05,
        ob_zones=[OBZone(1.096, 1.094, 0)], fvg_zones=[Zone(1.0955, 1.0945)],
        h4_highs=[1.05, 1.06, 1.07, 1.08], h4_lows=[1.04, 1.05, 1.06, 1.07],
        adx_h4=30.0,
    )
    assert res.allow is False
    assert "PD_REJECT" in res.reason


def test_silver_bullet_and_overlap_detection() -> None:
    from datetime import time
    assert silver_bullet_active(time(3, 30)) is True
    assert silver_bullet_active(time(6, 0)) is False
    assert overlap_active(time(10, 0)) is True
    assert overlap_active(time(15, 0)) is False
    assert current_session(time(3, 30)) == "LONDON_OPEN"
    assert current_session(time(10, 0)) == "LONDON_NY_OVERLAP"


def test_optimal_session_routing() -> None:
    assert is_optimal_session("EURUSD#", "LONDON_OPEN") is True
    assert is_optimal_session("EURUSD#", "ASIAN") is False
    assert is_optimal_session("BTCUSD#", "DEAD") is True


def test_kill_zone_context_silver_bullet_bonus() -> None:
    now = datetime(2026, 5, 14, 10, 15, tzinfo=timezone.utc)
    ctx = kill_zone_context("EURUSD#", now)
    assert ctx.silver_bullet_active is True
    assert ctx.overlap_active is True
    assert ctx.confluence_bonus == 1
    assert ctx.lot_multiplier == 1.2


def test_hard_reject_extreme_vol() -> None:
    ctx = build_rich_context(
        symbol="EURUSD#", timeframe="M5", direction="BUY", confluence_score=4,
        volatility_regime="EXTREME",
    )
    reject, reason = hard_reject_check(ctx)
    assert reject is True
    assert "Extreme" in reason


def test_hard_reject_consumed_ob() -> None:
    ctx = build_rich_context(
        symbol="EURUSD#", timeframe="M5", direction="BUY", confluence_score=4,
        ob_freshness="CONSUMED",
    )
    reject, reason = hard_reject_check(ctx)
    assert reject is True


def test_hard_reject_counter_trend() -> None:
    ctx = build_rich_context(
        symbol="EURUSD#", timeframe="M5", direction="BUY", confluence_score=4,
        h4_bias="BEARISH", adx_h4=30.0,
    )
    reject, _ = hard_reject_check(ctx)
    assert reject is True


def test_hard_reject_pre_news_window() -> None:
    ctx = build_rich_context(
        symbol="EURUSD#", timeframe="M5", direction="BUY", confluence_score=4,
        hours_to_next_news=0.25,
    )
    reject, reason = hard_reject_check(ctx)
    assert reject is True
    assert "news" in reason.lower()


def test_hard_reject_clean_setup_passes() -> None:
    ctx = build_rich_context(
        symbol="EURUSD#", timeframe="M5", direction="BUY", confluence_score=4,
        ob_freshness="FRESH", h4_bias="BULLISH", adx_h4=30.0,
        volatility_regime="NORMAL", spread_multiplier=1.1,
        hours_to_next_news=10.0,
    )
    reject, _ = hard_reject_check(ctx)
    assert reject is False


def test_recommend_entry_inside_zone_uses_market() -> None:
    rec = recommend_entry(
        current_price=1.0800, zone_entry=1.0795, direction="BUY", atr14=0.001,
    )
    assert rec.order_type == "MARKET"


def test_recommend_entry_near_zone_uses_limit() -> None:
    rec = recommend_entry(
        current_price=1.0810, zone_entry=1.0808, direction="BUY", atr14=0.001,
    )
    assert rec.order_type == "LIMIT"
    assert rec.entry_price == 1.0808
    assert rec.expires_at is not None


def test_recommend_entry_far_falls_back_to_market() -> None:
    rec = recommend_entry(
        current_price=1.0900, zone_entry=1.0800, direction="BUY", atr14=0.001,
    )
    assert rec.order_type == "MARKET"
    assert "TOO_FAR" in rec.reason


def test_slippage_acceptable_and_pips() -> None:
    ok, pips = slippage_acceptable(1.10000, 1.10010, digits=5, max_pips=2.0)
    assert ok is True and 0.9 < pips < 1.1
    ok2, pips2 = slippage_acceptable(1.10000, 1.10050, digits=5, max_pips=2.0)
    assert ok2 is False and pips2 > 2.0
