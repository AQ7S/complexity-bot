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
        SettingsSnapshot, TradesSnapshot,
    )

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
            logger.info("manual retrain requested for {} (Phase 12 worker)", model.model)
            return Ack(ref_type=type_, ok=True, error="queued")

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
            return Ack(ref_type=type_, ok=True, error="not implemented in phase 10")

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

    tasks: list[asyncio.Task] = [
        asyncio.create_task(_heartbeat_loop(state)),
        asyncio.create_task(_telemetry_loop(state)),
    ]

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
