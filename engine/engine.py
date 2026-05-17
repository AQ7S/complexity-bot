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
                from engine.models import train_online
                from engine.ipc.messages import ModelUpdate, Notification
                from datetime import datetime as _dt
                target_model = str(model.model)
                next_v = train_online.latest_checkpoint_version(target_model) + 1
                BUS.publish("model_update", ModelUpdate(
                    model_name=target_model, version=f"retrain_starting_v{next_v}",
                    accuracy=None, loss=None,
                ))
                BUS.publish("notification", Notification(
                    event="TRAINING_COMPLETE",
                    title=f"Retrain queued: {target_model}",
                    body=f"Spawning low-priority worker for v{next_v}…",
                    sound="signal.wav",
                ))
                async def _supervise():
                    try:
                        loop = asyncio.get_event_loop()
                        st = train_online.OnlineState()
                        proc = await loop.run_in_executor(
                            None,
                            lambda: train_online.spawn_retrain(
                                st,
                                worker=train_online._stub_retrain_worker,
                                worker_args=(next_v, target_model),
                                model_name=target_model,
                            ),
                        )
                        while proc.is_alive():
                            await asyncio.sleep(0.5)
                        proc.join(timeout=2)
                        new_path = train_online._newest_checkpoint(target_model)
                        version = new_path.stem if new_path else f"{target_model}_v{next_v}_{_dt.utcnow():%Y%m%d}"
                        BUS.publish("model_update", ModelUpdate(
                            model_name=target_model, version=version,
                            accuracy=None, loss=None,
                        ))
                        BUS.publish("notification", Notification(
                            event="TRAINING_COMPLETE",
                            title=f"Retrain complete: {target_model}",
                            body=f"New checkpoint: {version}",
                            sound="complete.wav",
                        ))
                        logger.info("manual retrain complete model={} version={}",
                                    target_model, version)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("manual retrain supervisor failed")
                        BUS.publish("notification", Notification(
                            event="ENGINE_ERROR",
                            title="Retrain failed",
                            body=f"{type(e).__name__}: {e}",
                            sound="error.wav",
                        ))
                asyncio.create_task(_supervise())
                return Ack(ref_type=type_, ok=True, error=f"spawned v{next_v}")
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


async def run() -> int:
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
            tasks.append(asyncio.create_task(_account_snapshot_loop(state)))
            tasks.append(asyncio.create_task(_tick_publish_loop(state)))
            from engine.execution.position_monitor import run as monitor_run
            tasks.append(asyncio.create_task(monitor_run(state.stop_event)))
        except Exception as e:  # noqa: BLE001
            logger.error("MT5 boot failed, running IPC-only: {}", e)
            state.mt5_connected = False
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
