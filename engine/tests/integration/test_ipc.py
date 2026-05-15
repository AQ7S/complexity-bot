"""Phase 10 — IPC server + broadcaster + message protocol round-trip tests.

The engine is a background service that survives UI close. These tests
verify:
  - Pydantic message envelopes round-trip through `parse()`/`envelope()`.
  - The WS server listens on the configured host:port, hands subscribers
    a copy of every published frame, and responds to `cmd_pause` with an
    `ack` frame.
  - A second client can connect, miss earlier frames (queues are
    per-client), and still receive subsequent broadcasts.
  - JSON Schema dump writes a non-empty file with one entry per type.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from engine.ipc.broadcaster import BUS
from engine.ipc.messages import (
    Ack, AccountUpdate, CmdPause, EngineStatus, PAYLOAD_TYPES,
    dump_schema, envelope, parse,
)


# ---------------------------------------------------------------------------
# Pure protocol tests
# ---------------------------------------------------------------------------

def test_envelope_shape():
    e = envelope("engine_status", EngineStatus(status="LIVE", uptime_s=5, mt5_connected=True))
    assert set(e) == {"type", "ts", "data"}
    assert e["type"] == "engine_status"
    assert isinstance(e["ts"], int) and e["ts"] > 0
    assert e["data"]["status"] == "LIVE"


def test_parse_roundtrip_account_update():
    raw = json.dumps(envelope("account_update", AccountUpdate(
        equity=10_050.0, balance=10_000.0, free_margin=9_800.0,
        drawdown_pct=0.0, open_positions=1,
    )))
    t, model = parse(raw)
    assert t == "account_update"
    assert isinstance(model, AccountUpdate)
    assert model.equity == 10_050.0


def test_parse_rejects_unknown_type():
    with pytest.raises(ValueError):
        parse(json.dumps({"type": "bogus", "ts": 0, "data": {}}))


def test_dump_schema_writes_all_types(tmp_path):
    out = dump_schema(tmp_path / "ipc-schema.json")
    body = json.loads(out.read_text())
    assert "envelope" in body and "payloads" in body
    assert set(body["payloads"]) == set(PAYLOAD_TYPES)


# ---------------------------------------------------------------------------
# Live WS round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def free_port(monkeypatch):
    """Pick an unused localhost port and pin it for the test."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setenv("IPC_WS_PORT", str(port))
    monkeypatch.setenv("IPC_AUTH_TOKEN", "")  # disable auth for this test
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)
    return port


@pytest.mark.asyncio
async def test_ws_server_broadcast_and_command_roundtrip(free_port):
    import websockets
    from engine.ipc.ws_server import WSServer

    received_cmds: list[tuple[str, object]] = []

    async def handler(t, model):
        received_cmds.append((t, model))
        return Ack(ref_type=t, ok=True)

    server = WSServer(on_command=handler)
    await server.start()
    try:
        url = f"ws://127.0.0.1:{free_port}"
        async with websockets.connect(url) as ws:
            # 1. Greeting frame is the engine_status snapshot.
            greeting = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            assert greeting["type"] == "engine_status"

            # Give the writer task time to attach its subscription queue.
            await asyncio.sleep(0.05)

            # 2. A broadcast lands on the connected client.
            BUS.publish("account_update", AccountUpdate(
                equity=10_000.0, balance=10_000.0, free_margin=9_900.0,
                drawdown_pct=0.0, open_positions=0,
            ))
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert frame["type"] == "account_update"
            assert frame["data"]["equity"] == 10_000.0

            # 3. A cmd_pause is dispatched and acked.
            await ws.send(json.dumps(envelope("cmd_pause", CmdPause(paused=True))))
            ack_frame = None
            for _ in range(5):
                got = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if got["type"] == "ack":
                    ack_frame = got
                    break
            assert ack_frame is not None and ack_frame["data"]["ok"] is True
            assert received_cmds and received_cmds[0][0] == "cmd_pause"
            assert received_cmds[0][1].paused is True
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_ws_server_supports_multiple_clients(free_port):
    import websockets
    from engine.ipc.ws_server import WSServer

    server = WSServer(on_command=None)
    await server.start()
    try:
        url = f"ws://127.0.0.1:{free_port}"
        async with websockets.connect(url) as a, websockets.connect(url) as b:
            # consume greetings
            await asyncio.wait_for(a.recv(), timeout=2.0)
            await asyncio.wait_for(b.recv(), timeout=2.0)
            await asyncio.sleep(0.05)

            BUS.publish("engine_status", EngineStatus(
                status="LIVE", uptime_s=10, mt5_connected=False,
            ))
            fa = json.loads(await asyncio.wait_for(a.recv(), timeout=2.0))
            fb = json.loads(await asyncio.wait_for(b.recv(), timeout=2.0))
            assert fa["type"] == fb["type"] == "engine_status"
            assert fa["data"]["uptime_s"] == 10
            assert fb["data"]["uptime_s"] == 10
    finally:
        await server.stop()
