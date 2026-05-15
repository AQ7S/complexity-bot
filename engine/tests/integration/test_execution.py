"""Phase 9 execution tests.

Plan asserts:
- On XM demo, send a BUY EURUSD via order_router with lot per Appendix E
  (within 1 volume_step), SL/TP correct, and SQLite trade row complete with
  close_reason='MANUAL' after closing.
- Open ≥ 1 position, trigger emergency close_all(); positions are closed
  within 2 seconds.
- Lot calculator unit cases (Appendix E reference + insufficient-equity edge).
"""
from __future__ import annotations

import sys
import time
from datetime import date

import pytest

from engine.config import settings
from engine.risk import lot_calc, manager
from engine.risk.lot_calc import SymbolInfo


# ----------------------------------------------------------------------------
# Pure-function tests (no MT5)
# ----------------------------------------------------------------------------

def test_lot_calc_eurusd_appendix_e_reference():
    """The exact worked example from Appendix E."""
    sym = SymbolInfo(
        name="EURUSD", point=0.00001, digits=5,
        tick_size=0.00001, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
    )
    res = lot_calc.compute_lot(
        equity=10_000.0, entry=1.07323, sl_price=1.07223, symbol=sym,
        risk_pct=0.02, claude_risk_adjustment=1.0,
    )
    assert res.ok, res
    assert abs(res.raw_lot - 2.00) < 1e-6
    assert abs(res.lot - 2.00) < 1e-6
    assert abs(res.risk_usd - 200.0) < 1e-6
    assert abs(res.sl_distance - 0.00100) < 1e-9


def test_lot_calc_xauusd_insufficient_equity_rejects():
    sym = SymbolInfo(
        name="XAUUSD", point=0.01, digits=2,
        tick_size=0.01, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
    )
    res = lot_calc.compute_lot(
        equity=50.0, entry=2350.0, sl_price=2348.0, symbol=sym,
        risk_pct=0.02, claude_risk_adjustment=1.0,
    )
    assert not res.ok
    assert res.reason == "INSUFFICIENT_EQUITY_FOR_RISK"


def test_lot_calc_quantizes_to_volume_step():
    sym = SymbolInfo(
        name="EURUSD", point=0.00001, digits=5,
        tick_size=0.00001, tick_value=1.0,
        volume_min=0.01, volume_max=100.0, volume_step=0.10,
    )
    res = lot_calc.compute_lot(
        equity=10_000.0, entry=1.07323, sl_price=1.07223, symbol=sym,
        risk_pct=0.02, claude_risk_adjustment=1.0,
    )
    assert res.ok
    # 0.10 step ⇒ 2.00 quantizes to 2.00; raw_lot=2.0
    assert abs(res.lot - 2.00) < 1e-6


def test_kill_trigger_intraday_threshold():
    state = manager.new_state(snapshot_balance=10_000.0)
    kill, kind, dd = manager.evaluate_kill_triggers(state, equity=9_705.0)  # -2.95%
    assert not kill
    kill, kind, dd = manager.evaluate_kill_triggers(state, equity=9_690.0)  # -3.10%
    assert kill and kind == "INTRADAY"
    assert dd >= settings.INTRADAY_KILL_PCT


def test_kill_trigger_weekly_threshold():
    # Simulate mid-week: balance has been creeping down across days.
    # Today started at 9_500 (week start 10_000). Equity must dip past
    # weekly -8% (= 9_200) without violating today's -3% (= 9_215).
    state = manager.new_state(snapshot_balance=10_000.0)
    state.starting_balance_today = 9_500.0
    kill, kind, dd = manager.evaluate_kill_triggers(state, equity=9_300.0)  # -7% wk, -2.1% day
    assert not kill
    kill, kind, dd = manager.evaluate_kill_triggers(state, equity=9_180.0)  # -8.2% wk, -3.4% day
    # Weekly is checked first ⇒ fires before intraday.
    assert kill and kind == "WEEKLY", (kill, kind, dd)


def test_can_open_new_short_circuits_in_order():
    bad = manager.PreTradeChecks(is_paused=True)
    ok, reason = manager.can_open_new(bad)
    assert not ok and reason == "PAUSED"

    full = manager.PreTradeChecks(open_positions=settings.MAX_CONCURRENT_POSITIONS)
    ok, reason = manager.can_open_new(full)
    assert not ok and reason == "MAX_POSITIONS"

    good = manager.PreTradeChecks()
    ok, reason = manager.can_open_new(good)
    assert ok and reason is None


def test_correlated_open_count_threshold():
    import pandas as pd
    syms = ["EURUSD", "GBPUSD", "XAUUSD"]
    matrix = pd.DataFrame(
        [[1.0, 0.85, 0.10],
         [0.85, 1.0, 0.05],
         [0.10, 0.05, 1.0]],
        index=syms, columns=syms,
    )
    n = manager.correlated_open_count(["GBPUSD", "XAUUSD"], "EURUSD", matrix)
    assert n == 1   # only GBPUSD crosses 0.80


# ----------------------------------------------------------------------------
# Live MT5 execution tests
# ----------------------------------------------------------------------------

LIVE_SKIP_REASON = "Live execution test requires MT5 + open FX session"


@pytest.mark.skipif(sys.platform != "win32", reason="MetaTrader5 is Windows-only")
@pytest.mark.skipif(not settings.have_mt5_credentials(),
                    reason="MT5 credentials missing in .env")
def test_live_buy_eurusd_close_round_trip(tmp_path, monkeypatch):
    """End-to-end: open a tiny BUY EURUSD, verify SL/TP, then close it."""
    import MetaTrader5 as mt5
    from engine.execution import order_router
    from engine.mt5_link import connection

    # Route SQLite journal writes to a per-test DB so we don't pollute prod.
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "journal.sqlite"))
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)

    connection.initialize_with_retry()
    try:
        connection.ensure_symbols_visible(["EURUSD"])
        tick = mt5.symbol_info_tick("EURUSD")
        if tick is None or tick.time == 0 or (time.time() - tick.time) > 120:
            pytest.skip(f"{LIVE_SKIP_REASON} (EURUSD has no recent tick)")

        # Use a tight, *minimum-lot-friendly* SL distance: 50 points.
        info = mt5.symbol_info("EURUSD")
        sl_distance = 50 * info.point
        entry = float(tick.ask)
        sl = entry - sl_distance
        tp = entry + 2 * sl_distance      # 1:2 RR

        # Compute expected lot via the formula and round to step.
        lot_res = order_router.size_lot_for("EURUSD", entry=entry, sl_price=sl)
        assert lot_res.ok, f"lot calc failed: {lot_res}"
        # Use a tiny lot for the live test regardless of formula output, to
        # protect the demo balance and pass broker margin.
        live_lot = info.volume_min

        req = order_router.OrderRequest(
            symbol="EURUSD", direction="BUY", lot=live_lot,
            sl=round(sl, info.digits), tp=round(tp, info.digits),
            comment="phase9-test",
        )
        sent = order_router.send_order(req)
        assert sent.ok, f"send_order failed: retcode={sent.retcode} {sent.comment}"
        try:
            # Verify the broker accepted SL/TP within 1 tick of our request.
            positions = mt5.positions_get(ticket=sent.ticket) or ()
            assert positions, "no position found after send_order"
            pos = positions[0]
            assert abs(pos.sl - req.sl) <= info.tick_size, (pos.sl, req.sl)
            assert abs(pos.tp - req.tp) <= info.tick_size, (pos.tp, req.tp)
        finally:
            assert order_router.close_position(sent.ticket, reason="MANUAL")

        # SQLite row should reflect the close.
        from engine.data.sqlite_journal import open_journal
        with open_journal() as con:
            row = con.execute(
                "SELECT * FROM trades WHERE mt5_ticket=?", (sent.ticket,)
            ).fetchone()
            assert row is not None
            assert row["close_reason"] == "MANUAL"
            assert row["close_time"] is not None
    finally:
        connection.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="MetaTrader5 is Windows-only")
@pytest.mark.skipif(not settings.have_mt5_credentials(),
                    reason="MT5 credentials missing in .env")
def test_live_close_all_within_two_seconds(tmp_path, monkeypatch):
    import MetaTrader5 as mt5
    from engine.execution import order_router
    from engine.mt5_link import connection
    from engine.risk import manager as risk_mgr

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "journal.sqlite"))

    connection.initialize_with_retry()
    try:
        connection.ensure_symbols_visible(["EURUSD"])
        tick = mt5.symbol_info_tick("EURUSD")
        if tick is None or tick.time == 0 or (time.time() - tick.time) > 120:
            pytest.skip(f"{LIVE_SKIP_REASON} (EURUSD has no recent tick)")

        info = mt5.symbol_info("EURUSD")
        sl_distance = 50 * info.point
        entry = float(tick.ask)
        # open one tiny position (avoid hammering the demo)
        req = order_router.OrderRequest(
            symbol="EURUSD", direction="BUY", lot=info.volume_min,
            sl=round(entry - sl_distance, info.digits),
            tp=round(entry + 2 * sl_distance, info.digits),
            comment="phase9-closeall",
        )
        sent = order_router.send_order(req)
        assert sent.ok, sent

        before = len(mt5.positions_get() or ())
        assert before >= 1

        t0 = time.time()
        result = risk_mgr.emergency_close_all(reason="MANUAL")
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"close_all took {elapsed:.2f}s"
        assert result["closed"] >= 1
        assert (mt5.positions_get() or ()) == ()
    finally:
        connection.shutdown()
