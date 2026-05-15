"""asyncio WebSocket server — UI bridge on 127.0.0.1:8765.

Each client is given its own subscription queue from the broadcaster; a
writer task drains the queue to the socket while a reader task validates
inbound command frames and dispatches them to a registered handler.

The server binds localhost-only by default. If `IPC_AUTH_TOKEN` is set in
the env, the first inbound frame from each client must be `{type:"auth",
data:{token:"..."}}` or the connection is closed.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol

from engine.config import settings
from engine.ipc.broadcaster import BUS
from engine.ipc.messages import (
    Ack, COMMAND_TYPES, EngineStatus, envelope, parse,
)

CommandHandler = Callable[[str, object], Awaitable[Ack]]


class WSServer:
    def __init__(self, on_command: CommandHandler | None = None) -> None:
        self._on_command = on_command
        self._server: websockets.Serve | None = None
        self._clients: set[WebSocketServerProtocol] = set()

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handle, settings.IPC_HOST, settings.IPC_WS_PORT,
            ping_interval=20, ping_timeout=20, max_size=2**20,
        )
        logger.info("IPC WS listening on ws://{}:{}", settings.IPC_HOST, settings.IPC_WS_PORT)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def _handle(self, ws: WebSocketServerProtocol) -> None:
        peer = ws.remote_address
        self._clients.add(ws)
        logger.info("UI connected from {}", peer)
        if not await self._authenticate(ws):
            self._clients.discard(ws)
            return

        # Greet with current engine status so the UI can render immediately.
        await ws.send(json.dumps(envelope("engine_status", EngineStatus(
            status="LIVE", uptime_s=0, mt5_connected=False,
        ))))

        queue = await BUS.subscribe()
        writer = asyncio.create_task(self._writer(ws, queue))
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            writer.cancel()
            await BUS.unsubscribe(queue)
            self._clients.discard(ws)
            logger.info("UI disconnected {}", peer)

    async def _authenticate(self, ws: WebSocketServerProtocol) -> bool:
        if not settings.IPC_AUTH_TOKEN:
            return True
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            obj = json.loads(raw)
            if obj.get("type") == "auth" and obj.get("data", {}).get("token") == settings.IPC_AUTH_TOKEN:
                return True
        except (asyncio.TimeoutError, json.JSONDecodeError, ValueError):
            pass
        await ws.close(code=4401, reason="unauthorized")
        return False

    async def _writer(self, ws: WebSocketServerProtocol, queue: asyncio.Queue[dict]) -> None:
        try:
            while True:
                frame = await queue.get()
                await ws.send(json.dumps(frame))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    async def _dispatch(self, ws: WebSocketServerProtocol, raw: str | bytes) -> None:
        try:
            t, model = parse(raw)
        except (ValueError, json.JSONDecodeError) as e:
            await ws.send(json.dumps(envelope("ack", Ack(ref_type="?", ok=False, error=str(e)))))
            return
        if t not in COMMAND_TYPES:
            await ws.send(json.dumps(envelope("ack", Ack(ref_type=t, ok=False, error="not a command"))))
            return
        if self._on_command is None:
            await ws.send(json.dumps(envelope("ack", Ack(ref_type=t, ok=False, error="no handler"))))
            return
        try:
            ack = await self._on_command(t, model)
        except Exception as e:  # noqa: BLE001
            logger.exception("command handler raised on {}", t)
            ack = Ack(ref_type=t, ok=False, error=f"{type(e).__name__}: {e}")
        await ws.send(json.dumps(envelope("ack", ack)))
