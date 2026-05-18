"""Complexity Engine — async main loop entrypoint.

Phase 10 responsibilities:
  * stand up the IPC WebSocket server on 127.0.0.1:8765
  * if MT5 credentials are present + terminal reachable, connect and run
    the background loops (account snapshot publish, tick stream broadcast,
    1-second position monitor)
  * route inbound `cmd_*` frames to the appropriate engine module
  * survive UI close (the WS server keeps listening); shut down cleanly
    on SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import io
import signal
import sys
import time
from typing import Any

from loguru import logger

from engine.config import settings


# Force UTF-8 stdout/stderr on Windows so loguru can render Unicode safely.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class EngineState:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.paused = False
        self.mt5_connected = False
        self.stop_event = asyncio.Event()

    @property
    def uptime_s(self) -> int:
        return int(time.time() - self.started_at)

    @property
    def status(self) -> str:
        if not self.mt5_connected:
            return "STARTING"
        if self.paused:
            return "PAUSED"
        return "LIVE"


async def _make_command_handler(state: EngineState):
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import (
        Ack, CmdEmergencyClose, CmdGetSettings, CmdGetTrades, CmdManualRetrain,
        CmdPause, CmdRunBacktest, CmdSetAlert, CmdSettingsUpdate,
        CmdStrategyToggle, SettingsSnapshot, TradesSnapshot,
    )
    from engine.strategy import orchestrator_runtime

    async def handle(type_: str, model: Any) -> Ack:
        if type_ == "cmd_pause":
            assert isinstance(model, CmdPause)
            state.paused = bool(model.paused)
            logger.warning("engine pause set to {}", state.paused)
            return Ack(ref_type=type_, ok=True)

        if type_ == "cmd_emergency_close":
            assert isinstance(model, CmdEmergencyClose)
            try:
                from engine.risk.manager import emergency_close_all
                result = await asyncio.to_thread(emergency_close_all, reason="MANUAL")
                return Ack(ref_type=type_, ok=True, error=str(result))
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_manual_retrain":
            assert isinstance(model, CmdManualRetrain)
            try:
                from engine.ipc.messages import ModelUpdate, Notification
                from engine.models.online_lgbm_trainer import retrain_now
                target_model = str(model.model)
                BUS.publish("model_update", ModelUpdate(
                    model_name="lightgbm", version="retrain_starting",
                    accuracy=None, loss=None,
                ))
                BUS.publish("notification", Notification(
                    event="TRAINING_COMPLETE",
                    title=f"Manual LightGBM retrain requested ({target_model})",
                    body="Running on a worker thread — should finish in <60s.",
                    sound="signal.wav",
                ))
                async def _supervise():
                    try:
                        outcome = await asyncio.to_thread(retrain_now)
                        if outcome.skipped:
                            BUS.publish("notification", Notification(
                                event="ENGINE_ERROR",
                                title="Retrain skipped",
                                body=outcome.reason,
                                sound="error.wav",
                            ))
                            logger.warning("manual retrain skipped: {}", outcome.reason)
                            return
                        BUS.publish("model_update", ModelUpdate(
                            model_name="lightgbm",
                            version=outcome.checkpoint or "unknown",
                            accuracy=None, loss=float(outcome.best_val_logloss),
                        ))
                        BUS.publish("notification", Notification(
                            event="TRAINING_COMPLETE",
                            title="LightGBM retrain complete",
                            body=(f"n_train={outcome.n_train} n_val={outcome.n_val} "
                                  f"logloss={outcome.best_val_logloss:.4f} "
                                  f"({outcome.elapsed_s:.1f}s)"),
                            sound="complete.wav",
                        ))
                        logger.info("manual LGBM retrain complete: {}", outcome.checkpoint)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("manual retrain supervisor failed")
                        BUS.publish("notification", Notification(
                            event="ENGINE_ERROR",
                            title="Retrain failed",
                            body=f"{type(e).__name__}: {e}",
                            sound="error.wav",
                        ))
                asyncio.create_task(_supervise())
                return Ack(ref_type=type_, ok=True, error="spawned LightGBM retrain")
            except Exception as e:  # noqa: BLE001
                logger.exception("manual retrain command failed")
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_settings_update":
            assert isinstance(model, CmdSettingsUpdate)
            try:
                from engine.data.sqlite_journal import open_journal
                with open_journal() as con:
                    for k, v in model.partial.items():
                        con.execute(
                            "INSERT INTO settings_kv(k,v) VALUES(?,?) "
                            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                            (str(k), str(v)),
                        )
                    con.commit()
                return Ack(ref_type=type_, ok=True)
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_run_backtest":
            assert isinstance(model, CmdRunBacktest)
            try:
                from datetime import datetime as _dt
                from engine.strategy.backtest import (
                    BacktestConfig, run_backtest as _run_bt, format_report,
                )
                from engine.ipc.messages import BacktestResult as _BR
                cfg_extra = model.strategy_config or {}
                cfg = BacktestConfig(
                    symbol=model.symbol,
                    from_date=_dt.fromisoformat(model.from_),
                    to_date=_dt.fromisoformat(model.to),
                    timeframe=str(cfg_extra.get("timeframe", "M5")),
                    starting_equity=float(cfg_extra.get("starting_equity", 10_000.0)),
                    risk_pct=float(cfg_extra.get("risk_pct", 0.02)),
                    min_confluence=int(cfg_extra.get("min_confluence", 3)),
                )

                async def _run():
                    try:
                        report = await asyncio.to_thread(_run_bt, cfg)
                        BUS.publish("backtest_result", _BR(
                            symbol=cfg.symbol, timeframe=cfg.timeframe,
                            from_date=cfg.from_date.isoformat(),
                            to_date=cfg.to_date.isoformat(),
                            total_trades=report.total_trades, wins=report.wins, losses=report.losses,
                            win_rate=report.win_rate, net_pnl_usd=report.net_pnl_usd,
                            avg_r_multiple=report.avg_r_multiple, sharpe=report.sharpe,
                            profit_factor=report.profit_factor,
                            max_drawdown_pct=report.max_drawdown_pct,
                            spread_pips_used=report.spread_pips_used,
                            slippage_pips_used=report.slippage_pips_used,
                            swap_long_pips_used=report.swap_long_pips_used,
                            swap_short_pips_used=report.swap_short_pips_used,
                            starting_equity=report.starting_equity,
                            ending_equity=report.ending_equity,
                        ))
                        logger.info("backtest complete:\n{}", format_report(report))
                    except Exception as e:  # noqa: BLE001
                        logger.exception("backtest failed")
                        BUS.publish("backtest_result", _BR(
                            symbol=cfg.symbol, timeframe=cfg.timeframe,
                            from_date=cfg.from_date.isoformat(),
                            to_date=cfg.to_date.isoformat(),
                            total_trades=0, wins=0, losses=0, win_rate=0.0,
                            net_pnl_usd=0.0, avg_r_multiple=0.0, sharpe=0.0,
                            profit_factor=0.0, max_drawdown_pct=0.0,
                            spread_pips_used=0.0, slippage_pips_used=0.0,
                            swap_long_pips_used=0.0, swap_short_pips_used=0.0,
                            starting_equity=cfg.starting_equity,
                            ending_equity=cfg.starting_equity,
                            error=f"{type(e).__name__}: {e}",
                        ))

                asyncio.create_task(_run())
                return Ack(ref_type=type_, ok=True, error=f"backtest queued: {cfg.symbol} {cfg.from_date.date()}→{cfg.to_date.date()}")
            except Exception as e:  # noqa: BLE001
                logger.exception("cmd_run_backtest dispatch failed")
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_get_trades":
            assert isinstance(model, CmdGetTrades)
            try:
                from engine.data.sqlite_journal import open_journal
                with open_journal() as con:
                    rows = con.execute(
                        "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                        (int(model.limit),),
                    ).fetchall()
                trades = [dict(r) for r in rows]
                BUS.publish("trades_snapshot", TradesSnapshot(trades=trades))
                return Ack(ref_type=type_, ok=True, error=f"{len(trades)} rows")
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_get_settings":
            assert isinstance(model, CmdGetSettings)
            try:
                from engine.data.sqlite_journal import open_journal
                with open_journal() as con:
                    rows = con.execute(
                        "SELECT k,v FROM settings_kv WHERE k NOT LIKE 'sec:%'"
                    ).fetchall()
                values = {r["k"]: r["v"] for r in rows}
                BUS.publish("settings_snapshot", SettingsSnapshot(values=values))
                return Ack(ref_type=type_, ok=True)
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_strategy_toggle":
            assert isinstance(model, CmdStrategyToggle)
            try:
                orch = orchestrator_runtime.get_orchestrator()
                ok = orch.set_mode(model.name, model.mode)
                if not ok:
                    return Ack(ref_type=type_, ok=False, error=f"unknown strategy {model.name!r}")
                BUS.publish("strategy_status", orch.snapshot())
                logger.info("strategy {} → mode {}", model.name, model.mode)
                return Ack(ref_type=type_, ok=True)
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        if type_ == "cmd_set_alert":
            assert isinstance(model, CmdSetAlert)
            try:
                from engine.data.sqlite_journal import open_journal
                with open_journal() as con:
                    con.execute(
                        "INSERT INTO price_alerts(symbol, direction, threshold, enabled) "
                        "VALUES(?,?,?,1)",
                        (model.symbol, model.direction, float(model.threshold)),
                    )
                    con.commit()
                return Ack(ref_type=type_, ok=True)
            except Exception as e:  # noqa: BLE001
                return Ack(ref_type=type_, ok=False, error=f"{type(e).__name__}: {e}")

        return Ack(ref_type=type_, ok=False, error="unhandled command")

    return handle


async def _mt5_reconnect_loop(state: EngineState, interval_s: float = 30.0) -> None:
    """Keep retrying MT5 connection until success, then re-attempt if it drops."""
    from engine.mt5_link import connection  # noqa: PLC0415
    from engine.config.symbols import SYMBOLS_13  # noqa: PLC0415
    try:
        import MetaTrader5 as mt5  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.warning("mt5_reconnect: MetaTrader5 module unavailable — loop disabled")
        return
    consecutive_failures = 0
    while not state.stop_event.is_set():
        try:
            # Verify the existing connection is still alive
            if state.mt5_connected:
                acct = await asyncio.to_thread(mt5.account_info)
                if acct is None:
                    logger.warning("mt5_reconnect: connection dropped (account_info returned None)")
                    state.mt5_connected = False
            if not state.mt5_connected:
                logger.info("mt5_reconnect: attempting connection (failure #{})", consecutive_failures + 1)
                try:
                    await asyncio.to_thread(
                        connection.initialize_with_retry, attempts=1, delay_s=1.0,
                    )
                    state.mt5_connected = True
                    consecutive_failures = 0
                    await asyncio.to_thread(
                        connection.ensure_symbols_visible, [s.name for s in SYMBOLS_13],
                    )
                    logger.info("MT5 CONNECTED — scanner will resume on next cycle")
                except Exception as e:  # noqa: BLE001
                    consecutive_failures += 1
                    if consecutive_failures % 4 == 1:  # log every ~2min
                        logger.warning(
                            "mt5_reconnect: still failing ({}x): {}",
                            consecutive_failures, e,
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning("mt5_reconnect loop raised: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _signal_scanner_loop(state: EngineState, interval_s: float = 30.0) -> None:
    """Elite live signal generation — Tier A + B + C gates wired in.

    Per `interval_s`:
      * pull bars for each watchlist symbol × strategy timeframe
      * compute regime + H4 bias + live spread vs profile + news gate
      * run orchestrator.tick → strategy signals
      * for each signal: risk preconditions → Claude gate → lot size → fire
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from datetime import datetime as _datetime, timezone as _tz  # noqa: PLC0415
    from engine.config.symbols import SYMBOLS_13
    from engine.execution.order_router import OrderRequest, send_order, size_lot_for
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import Notification, SignalDetected
    from engine.strategy import orchestrator_runtime
    from engine.strategy.base import StrategyContext
    from engine.strategy.claude_gate import gate_factory
    from engine.utils.time_utils import kill_zone_active

    try:
        import MetaTrader5 as mt5  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.warning("signal_scanner: MetaTrader5 unavailable — loop disabled")
        return

    claude_call = gate_factory() if settings.ENABLE_CLAUDE_GATE else None
    TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
    TF_TO_MT5 = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
    }

    def _pull_bars(symbol: str, tf_key: str, n: int = 300) -> "pd.DataFrame | None":
        rates = mt5.copy_rates_from_pos(symbol, TF_TO_MT5[tf_key], 0, n)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df.set_index("ts")[["open", "high", "low", "close", "volume"]]

    # ------- Tier A helpers (cached, lazy-loaded) ----------------------------
    def _classify_regime(df: "pd.DataFrame") -> str | None:
        try:
            from engine.features.regime import classify  # noqa: PLC0415
            return classify(df).regime
        except Exception:
            return None

    def _h4_bias_for(symbol: str) -> str:
        try:
            from engine.features.smc import _h4_bias  # noqa: PLC0415
            h4 = _pull_bars(symbol, "H4", n=200)
            return _h4_bias(h4) if h4 is not None else "RANGING"
        except Exception:
            return "RANGING"

    _spread_profiles: dict[str, object] = {}
    def _spread_ok(symbol: str, current_spread: float) -> bool:
        try:
            from engine.risk.spread_profile import (  # noqa: PLC0415
                load_hourly_profile_from_duckdb, is_spread_acceptable,
            )
            prof = _spread_profiles.get(symbol)
            if prof is None:
                prof = load_hourly_profile_from_duckdb(symbol=symbol)
                _spread_profiles[symbol] = prof
            return is_spread_acceptable(prof, current_spread)
        except Exception:
            return True

    def _news_clear_for(symbol: str) -> bool:
        try:
            from engine.news.jblanked import hours_until_high_impact  # noqa: PLC0415
            ccy_base = symbol[:3].upper()
            ccy_quote = symbol[3:6].upper() if len(symbol) >= 6 else ""
            for c in (ccy_base, ccy_quote):
                if not c or not c.isalpha():
                    continue
                h = hours_until_high_impact(currency=c)
                if h is not None and h * 60 <= settings.NEWS_PAUSE_MINUTES_BEFORE:
                    return False
            return True
        except Exception:
            return True

    def _vpin_for(symbol: str) -> float:
        try:
            from engine.features.vpin import compute_vpin  # noqa: PLC0415
            ticks = mt5.copy_ticks_from_pos(symbol, 0, 1000)
            if ticks is None or len(ticks) == 0:
                return 0.0
            tick_dicts = [{
                "price": float((t["bid"] + t["ask"]) / 2),
                "volume": float(t.get("volume_real", 1.0) or 1.0),
            } for t in ticks]
            return float(compute_vpin(tick_dicts))
        except Exception:
            return 0.0

    # ------- Tier B helpers --------------------------------------------------
    def _open_positions_snapshot() -> list:
        try:
            return list(mt5.positions_get() or ())
        except Exception:
            return []

    def _open_correlated(symbol: str, open_syms: list[str]) -> int:
        if not open_syms:
            return 0
        # Cheap heuristic: count opens on the same base currency
        base = symbol[:3].upper()
        return sum(1 for s in open_syms if s != symbol and s[:3].upper() == base)

    def _intraday_dd_ok() -> bool:
        try:
            from engine.mt5_link import account  # noqa: PLC0415
            snap = account.snapshot()
            if snap.balance <= 0:
                return True
            dd = (snap.balance - snap.equity) / snap.balance
            return dd < settings.INTRADAY_KILL_PCT
        except Exception:
            return True

    # ------- Main loop -------------------------------------------------------
    iter_count = 0
    while not state.stop_event.is_set():
        try:
            iter_count += 1
            if state.paused:
                if iter_count % 10 == 1:
                    logger.info("signal_scanner: engine PAUSED — sleeping")
                await asyncio.sleep(interval_s)
                continue
            if not state.mt5_connected:
                if iter_count % 10 == 1:
                    logger.info("signal_scanner: waiting for MT5 connection (still STARTING)")
                await asyncio.sleep(interval_s)
                continue
            if not _intraday_dd_ok():
                logger.warning("signal_scanner: intraday DD kill active — skipping scan")
                await asyncio.sleep(interval_s)
                continue
            if iter_count % 4 == 1:  # every ~2 minutes
                logger.info("signal_scanner heartbeat — iter={} scanning {} symbols",
                            iter_count, 13)
            orch = orchestrator_runtime.get_orchestrator()
            timeframes_needed = sorted({tf for s in orch.strategies for tf in s.timeframes},
                                        key=lambda t: TF_MINUTES.get(t, 999))
            open_positions = _open_positions_snapshot()
            if len(open_positions) >= settings.MAX_CONCURRENT_POSITIONS:
                logger.debug("signal_scanner: max concurrent positions reached — skipping")
                await asyncio.sleep(interval_s)
                continue
            open_symbols = [p.symbol for p in open_positions]

            contexts: list[StrategyContext] = []
            h4_bias_cache: dict[str, str] = {}
            regime_cache: dict[str, str | None] = {}
            for sym in SYMBOLS_13:
                tick = await asyncio.to_thread(mt5.symbol_info_tick, sym.name)
                if tick is None:
                    continue
                current_spread = float(tick.ask - tick.bid)
                spread_ok = await asyncio.to_thread(_spread_ok, sym.name, current_spread)
                news_ok = await asyncio.to_thread(_news_clear_for, sym.name)
                vpin = await asyncio.to_thread(_vpin_for, sym.name)
                h4_bias_cache[sym.name] = h4_bias_cache.get(sym.name) \
                    or await asyncio.to_thread(_h4_bias_for, sym.name)
                for tf in timeframes_needed:
                    bars = await asyncio.to_thread(_pull_bars, sym.name, tf)
                    if bars is None or len(bars) < 50:
                        continue
                    if tf not in regime_cache:
                        regime_cache[tf] = None
                    reg = await asyncio.to_thread(_classify_regime, bars)
                    kz_ok = kill_zone_active(sym.name, _datetime.now(_tz.utc))
                    contexts.append(StrategyContext(
                        symbol=sym.name, timeframe=tf, bars=bars,
                        killzone_ok=kz_ok, news_clear=news_ok,
                        spread_acceptable=spread_ok, vpin_score=vpin,
                        regime=reg, h4_bias=h4_bias_cache[sym.name],
                    ))
            if not contexts:
                await asyncio.sleep(interval_s)
                continue
            result = await asyncio.to_thread(orch.tick, contexts)
            if iter_count % 4 == 1:
                logger.info(
                    "signal_scanner: {} contexts → {} signals (paused={} shadow={})",
                    len(contexts), len(result.signals),
                    len(result.skipped_paused), len(result.skipped_shadow),
                )
            for sig in result.signals:
                try:
                    BUS.publish("signal_detected", SignalDetected(
                        signal_id=f"{sig.strategy_name}-{sig.symbol}-{int(time.time())}",
                        symbol=sig.symbol, timeframe=sig.timeframe,
                        direction=sig.direction,
                        confluence=int(sig.confidence // 20),
                        sources={"strategy": sig.strategy_name},
                        claude=None,
                    ))
                except Exception as e:  # noqa: BLE001
                    logger.debug("signal broadcast failed: {}", e)
                if sig.direction == "HOLD":
                    continue

                # Tier B: per-symbol risk preconditions
                if _open_correlated(sig.symbol, open_symbols) >= settings.MAX_CORRELATED_POSITIONS:
                    logger.info("scanner reject {} {}: correlation cap", sig.strategy_name, sig.symbol)
                    continue
                if any(p.symbol == sig.symbol for p in open_positions):
                    logger.debug("scanner reject {}: already open", sig.symbol)
                    continue

                # Tier C: Claude gate (cost-tracked + budget-guarded)
                claude_K = 1.0
                if claude_call is not None:
                    try:
                        ctx_payload = {
                            "symbol": sig.symbol, "timeframe": sig.timeframe,
                            "direction": sig.direction, "confidence": sig.confidence,
                            "strategy": sig.strategy_name,
                            "reasoning_hint": sig.reasoning[:200],
                            "sl": sig.sl_price, "tp": sig.tp_price,
                        }
                        verdict = await asyncio.to_thread(claude_call, ctx_payload)
                        if verdict.decision == "SKIP" or verdict.decision != sig.direction:
                            logger.info("claude rejected {} {} {} ({})",
                                        sig.strategy_name, sig.symbol, sig.direction, verdict.decision)
                            continue
                        if verdict.confidence < 50:
                            continue
                        claude_K = max(0.5, min(1.5, verdict.risk_adjustment))
                    except Exception as e:  # noqa: BLE001
                        logger.warning("claude gate failed (proceeding K=1.0): {}", e)

                # Sizing + execution
                try:
                    tick = mt5.symbol_info_tick(sig.symbol)
                    entry = float(tick.ask if sig.direction == "BUY" else tick.bid)
                    lot_res = await asyncio.to_thread(
                        size_lot_for, sig.symbol,
                        entry=entry, sl_price=float(sig.sl_price), K=claude_K,
                    )
                    if not lot_res.ok:
                        logger.info("lot reject {} {} {}: {} (raw={:.4f})",
                                    sig.strategy_name, sig.symbol, sig.direction,
                                    lot_res.reason, lot_res.raw_lot)
                        continue
                    res = await asyncio.to_thread(send_order, OrderRequest(
                        symbol=sig.symbol, direction=sig.direction, lot=lot_res.lot,
                        sl=float(sig.sl_price), tp=float(sig.tp_price),
                        comment=f"{sig.strategy_name[:20]}",
                    ))
                    if res.ok:
                        logger.info("ORDER OK {} {} {} lot={:.2f} ticket={}",
                                    sig.strategy_name, sig.symbol, sig.direction,
                                    lot_res.lot, res.ticket)
                        BUS.publish("notification", Notification(
                            event="TRADE_OPENED",
                            title=f"{sig.symbol} {sig.direction}",
                            body=(f"{sig.strategy_name} lot={lot_res.lot:.2f} "
                                  f"SL={sig.sl_price:.5f} TP={sig.tp_price:.5f}"),
                            sound="trading_open.wav",
                        ))
                        open_positions = _open_positions_snapshot()
                        open_symbols = [p.symbol for p in open_positions]
                    else:
                        logger.warning("order_send rejected {} {}: retcode={} {}",
                                       sig.symbol, sig.direction, res.retcode, res.comment)
                except Exception as e:  # noqa: BLE001
                    logger.warning("execution path failed for {}: {}", sig.symbol, e)
        except Exception as e:  # noqa: BLE001
            logger.warning("signal_scanner loop raised: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _meta_policy_loop(state: EngineState, interval_s: float = 3600.0) -> None:
    """Tier C: every hour, ask Claude to analyse recent losers + propose
    parameter overrides. Whitelisted params only; persisted to claude_overrides
    table. Runtime reads active overrides via active_overrides()."""
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import Notification
    from engine.learning.claude_meta_policy import (
        apply_overrides, collect_recent_losers, propose_overrides,
    )
    last_run = 0.0
    while not state.stop_event.is_set():
        try:
            now = time.time()
            if now - last_run < interval_s:
                await asyncio.sleep(60)
                continue
            losers = await asyncio.to_thread(collect_recent_losers, n=10, lookback_hours=24)
            if len(losers) < 3:
                last_run = now
                await asyncio.sleep(interval_s / 4)
                continue
            overrides = await asyncio.to_thread(propose_overrides, losers)
            if overrides:
                n_applied = await asyncio.to_thread(apply_overrides, overrides)
                logger.info("meta_policy: applied {} parameter override(s)", n_applied)
                BUS.publish("notification", Notification(
                    event="TRAINING_COMPLETE",
                    title=f"Claude proposed {n_applied} param tweak(s)",
                    body="; ".join(f"{o.param}={o.new_value}" for o in overrides[:3]),
                    sound="signal.wav",
                ))
            last_run = now
        except Exception as e:  # noqa: BLE001
            logger.warning("meta_policy loop raised: {}", e)
            last_run = time.time()
        await asyncio.sleep(60)


async def _retrain_dispatcher_loop(state: EngineState, interval_s: float = 60.0) -> None:
    """Poll closed-trade count + drift signal; auto-retrain LightGBM when triggered."""
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import ModelUpdate, Notification
    from engine.learning.retrain_dispatcher import RetrainDispatcher
    from engine.data.sqlite_journal import open_journal

    dispatcher = RetrainDispatcher()
    last_seen_trade_id = 0
    try:
        with open_journal() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(id), 0) FROM trades WHERE pnl IS NOT NULL"
            ).fetchone()
            last_seen_trade_id = int(row[0]) if row else 0
    except Exception as e:  # noqa: BLE001
        logger.warning("retrain_dispatcher: initial trade-count read failed: {}", e)

    def _on_promotion(outcome, decision):
        try:
            BUS.publish("model_update", ModelUpdate(
                model_name="lightgbm",
                version=outcome.checkpoint or f"v{int(time.time())}",
                accuracy=None, loss=float(outcome.best_val_logloss),
            ))
            BUS.publish("notification", Notification(
                event="TRAINING_COMPLETE",
                title=f"LightGBM retrain: {decision.reason}",
                body=f"n_train={outcome.n_train} logloss={outcome.best_val_logloss:.4f}",
                sound="complete.wav",
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("retrain promotion broadcast failed: {}", e)

    while not state.stop_event.is_set():
        try:
            with open_journal() as con:
                row = con.execute(
                    "SELECT COALESCE(MAX(id), 0), COUNT(*) FROM trades "
                    "WHERE pnl IS NOT NULL AND id > ?",
                    (int(last_seen_trade_id),),
                ).fetchone()
                if row:
                    new_max = int(row[0])
                    new_count = int(row[1])
                    if new_count > 0:
                        for _ in range(new_count):
                            dispatcher.record_closed_trade()
                        last_seen_trade_id = max(last_seen_trade_id, new_max)
            cpu_pct = 0.0
            try:
                import psutil  # noqa: PLC0415
                cpu_pct = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            spawned = await asyncio.to_thread(
                dispatcher.tick,
                cpu_pct=cpu_pct,
                promotion_callback=_on_promotion,
            )
            if spawned:
                logger.info("retrain_dispatcher: retrain spawned")
                BUS.publish("notification", Notification(
                    event="TRAINING_COMPLETE",
                    title="LightGBM retrain spawned",
                    body=f"trades_since_last reset; cooldown {dispatcher.cooldown_s:.0f}s",
                    sound="signal.wav",
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("retrain_dispatcher loop raised: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _telemetry_loop(state: EngineState) -> None:
    from engine.utils.telemetry import Sampler, SAMPLE_INTERVAL_S
    sampler = Sampler()
    while not state.stop_event.is_set():
        try:
            await asyncio.to_thread(sampler.tick)
        except Exception as e:  # noqa: BLE001
            logger.warning("telemetry sampler raised: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=SAMPLE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def _heartbeat_loop(state: EngineState, interval_s: float = 2.0) -> None:
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import EngineStatus

    while not state.stop_event.is_set():
        BUS.publish("engine_status", EngineStatus(
            status=state.status, uptime_s=state.uptime_s,
            mt5_connected=state.mt5_connected,
        ))
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _account_snapshot_loop(state: EngineState, interval_s: float = 2.0) -> None:
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import AccountUpdate
    from engine.mt5_link import account

    while not state.stop_event.is_set():
        if state.mt5_connected:
            try:
                snap = await asyncio.to_thread(account.snapshot)
                BUS.publish("account_update", AccountUpdate(
                    equity=snap.equity, balance=snap.balance,
                    free_margin=snap.free_margin,
                    drawdown_pct=snap.drawdown_pct,
                    open_positions=snap.open_positions,
                ))
            except Exception as e:  # noqa: BLE001
                logger.warning("account snapshot failed: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _shadow_monitor_loop(state: EngineState, interval_s: float = 60.0) -> None:
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import (
        ShadowStatus, ModelPromotionReady, CalibrationUpdate,
    )
    from engine.execution.shadow_trader import (
        monitor_open_shadow_trades, compute_shadow_stats, is_promotion_ready,
    )
    from engine.learning.calibration import (
        recompute_and_persist, closed_shadow_trade_count, latest_calibration,
    )
    last_calib_at = 0

    def _live_prices() -> dict[str, float]:
        try:
            import MetaTrader5 as mt5
            from engine.config.symbols import SYMBOLS_13
            out: dict[str, float] = {}
            for sym in SYMBOLS_13:
                tick = mt5.symbol_info_tick(sym.name)
                if tick is None or tick.time_msc == 0:
                    continue
                out[sym.name] = float((tick.bid + tick.ask) / 2.0)
            return out
        except Exception:
            return {}

    while not state.stop_event.is_set():
        try:
            prices = await asyncio.to_thread(_live_prices)
            await asyncio.to_thread(monitor_open_shadow_trades, prices)
            stats = await asyncio.to_thread(compute_shadow_stats)
            BUS.publish("shadow_status", ShadowStatus(
                active=settings.shadow_mode_active(),
                total=stats.total, open_count=stats.open_count,
                closed_count=stats.closed_count,
                wins=stats.wins, losses=stats.losses, time_exits=stats.time_exits,
                win_rate=stats.win_rate, avg_r=stats.avg_r, sharpe=stats.sharpe,
                cumulative_pnl_r=stats.cumulative_pnl_r,
            ))

            closed = await asyncio.to_thread(closed_shadow_trade_count)
            if closed >= settings.ECE_RECOMPUTE_EVERY_N_TRADES and closed != last_calib_at and closed % settings.ECE_RECOMPUTE_EVERY_N_TRADES == 0:
                cal = await asyncio.to_thread(recompute_and_persist)
                last_calib_at = closed
                if cal.n_trades > 0:
                    BUS.publish("calibration_update", CalibrationUpdate(
                        ece_score=cal.ece_score, n_trades=cal.n_trades,
                        bins=[b.as_dict() for b in cal.bins],
                        overconfident=cal.overconfident,
                    ))

            ready, ready_stats = await asyncio.to_thread(is_promotion_ready, None)
            if ready:
                BUS.publish("model_promotion_ready", ModelPromotionReady(
                    current_model_sharpe=None,
                    shadow_sharpe=ready_stats.sharpe,
                    shadow_win_rate=ready_stats.win_rate,
                    shadow_trades=ready_stats.closed_count,
                    avg_r=ready_stats.avg_r,
                ))
        except Exception as e:  # noqa: BLE001
            logger.warning("shadow monitor loop: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _macro_snapshot_loop(state: EngineState, interval_s: float = 60.0) -> None:
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import MacroSnapshot as MacroSnapshotMsg
    from engine.news.macro_data import get_macro_snapshot

    while not state.stop_event.is_set():
        try:
            snap = await asyncio.to_thread(get_macro_snapshot)
            BUS.publish("macro_snapshot", MacroSnapshotMsg(
                yield_curve_bias=snap.yield_curve_bias,
                crypto_fear_greed=snap.crypto_fear_greed,
                fear_greed_value=snap.fear_greed_value,
                spread_us10y_us2y=snap.spread_us10y_us2y,
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("macro snapshot loop: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _strategy_status_loop(state: EngineState, interval_s: float = 5.0) -> None:
    """Publish the orchestrator snapshot every `interval_s` so the
    /strategies UI page stays current with weights + breaker state."""
    from engine.ipc.broadcaster import BUS
    from engine.strategy import orchestrator_runtime

    while not state.stop_event.is_set():
        try:
            orch = orchestrator_runtime.get_orchestrator()
            BUS.publish("strategy_status", orch.snapshot())
        except Exception as e:  # noqa: BLE001
            logger.warning("strategy status loop: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _tick_publish_loop(state: EngineState, interval_s: float = 0.5) -> None:
    """Poll the latest tick for each watchlist symbol; publish deltas only."""
    from engine.config.symbols import SYMBOLS_13
    from engine.ipc.broadcaster import BUS
    from engine.ipc.messages import TickUpdate

    last_seen: dict[str, int] = {}

    def _poll() -> list[TickUpdate]:
        import MetaTrader5 as mt5
        out: list[TickUpdate] = []
        for sym in SYMBOLS_13:
            tick = mt5.symbol_info_tick(sym.name)
            if tick is None or tick.time_msc == 0:
                continue
            if last_seen.get(sym.name) == tick.time_msc:
                continue
            last_seen[sym.name] = tick.time_msc
            out.append(TickUpdate(
                symbol=sym.name, bid=float(tick.bid), ask=float(tick.ask),
                spread=float(tick.ask - tick.bid),
                volume=float(getattr(tick, "volume_real", 0.0) or 0.0),
            ))
        return out

    while not state.stop_event.is_set():
        if state.mt5_connected:
            try:
                ticks = await asyncio.to_thread(_poll)
                from engine.data.spread_monitor import get_spread_monitor
                _sm = get_spread_monitor()
                for t in ticks:
                    _sm.update(t.symbol, t.spread)
                    BUS.publish("tick_update", t)
            except Exception as e:  # noqa: BLE001
                logger.warning("tick poll failed: {}", e)
        try:
            await asyncio.wait_for(state.stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def _setup_file_logger() -> None:
    """Attach a rotating file sink so logs persist to disk for diagnostics."""
    try:
        from pathlib import Path as _P  # noqa: PLC0415
        log_dir = _P(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "engine.log",
            rotation="20 MB",
            retention="14 days",
            level=settings.LOG_LEVEL,
            backtrace=True,
            diagnose=False,
            enqueue=True,
            encoding="utf-8",
        )
        logger.info("file logger attached: {}/engine.log", log_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not attach file logger: {}", e)


async def run() -> int:
    _setup_file_logger()
    state = EngineState()
    handler = await _make_command_handler(state)

    from engine.ipc.messages import dump_schema
    schema_path = dump_schema()
    logger.info("IPC schema written to {}", schema_path)

    from engine.ipc.ws_server import WSServer
    server = WSServer(on_command=handler)
    await server.start()

    try:
        from engine.data.event_log import log_event as _log_event
        _log_event("ENGINE_START", None, {"uptime_s": 0})
    except Exception as e:  # noqa: BLE001
        logger.warning("event_log start failed: {}", e)

    try:
        from engine.watchdog import serve_health_endpoint
        await serve_health_endpoint(state)
    except Exception as e:  # noqa: BLE001
        logger.warning("/health endpoint failed to start: {}", e)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(_heartbeat_loop(state)),
        asyncio.create_task(_telemetry_loop(state)),
        asyncio.create_task(_macro_snapshot_loop(state)),
        asyncio.create_task(_shadow_monitor_loop(state)),
        asyncio.create_task(_strategy_status_loop(state)),
        asyncio.create_task(_retrain_dispatcher_loop(state)),
        asyncio.create_task(_meta_policy_loop(state)),
    ]
    logger.info(
        "shadow_mode={} — order_send will {} real MT5 calls",
        settings.shadow_mode_active(),
        "BYPASS" if settings.shadow_mode_active() else "MAKE",
    )

    if settings.have_mt5_credentials():
        try:
            from engine.mt5_link import connection
            await asyncio.to_thread(connection.initialize_with_retry, attempts=3, delay_s=2.0)
            state.mt5_connected = True
            from engine.config.symbols import SYMBOLS_13
            await asyncio.to_thread(connection.ensure_symbols_visible, [s.name for s in SYMBOLS_13])
            logger.info("MT5 connected on startup")
        except Exception as e:  # noqa: BLE001
            logger.error("MT5 boot failed on startup (background reconnect will retry): {}", e)
            state.mt5_connected = False
        # Always register the MT5-dependent loops — they no-op until reconnect.
        tasks.append(asyncio.create_task(_account_snapshot_loop(state)))
        tasks.append(asyncio.create_task(_tick_publish_loop(state)))
        tasks.append(asyncio.create_task(_signal_scanner_loop(state)))
        tasks.append(asyncio.create_task(_mt5_reconnect_loop(state)))
        from engine.execution.position_monitor import run as monitor_run
        tasks.append(asyncio.create_task(monitor_run(state.stop_event)))
    else:
        logger.warning("MT5 creds missing — running IPC-only (UI dev mode)")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM) if sys.platform != "win32" else (signal.SIGINT,):
        try:
            loop.add_signal_handler(sig, state.stop_event.set)
        except NotImplementedError:
            pass

    print("Engine ready", flush=True)
    logger.info("engine ready — main loop blocking until stop_event")
    await state.stop_event.wait()

    logger.info("shutting down")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await server.stop()
    if state.mt5_connected:
        from engine.mt5_link import connection
        await asyncio.to_thread(connection.shutdown)
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
