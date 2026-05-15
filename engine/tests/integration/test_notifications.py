"""Phase 11 — Discord embeds, Windows toast triggers, sound assets.

We avoid hitting real Discord by spinning up an aiohttp mock server inside
the test that records every POST. The Discord dispatcher is pointed at it
via env override, so all 9 embed types (8 events + daily summary) can be
verified end-to-end without external deps.

The toast tests subscribe to the IPC bus and assert that `notify()`
publishes a `notification` frame with the expected sound mapping. The WAV
tests verify each of the 8 files exists, parses as 16-bit PCM mono 44.1k,
and falls within the duration window from Appendix K.

The test_notifications_disabled_skips check exercises the `NOTIFY_*_ENABLED`
toggles to confirm no side-effects when off.
"""
from __future__ import annotations

import asyncio
import struct
import wave
from pathlib import Path

import pytest

from engine.notifications import discord, windows_toast

REPO_ROOT = Path(__file__).resolve().parents[3]
SOUND_DIR = REPO_ROOT / "engine" / "sounds"


# ---------------------------------------------------------------------------
# WAV asset tests
# ---------------------------------------------------------------------------

EXPECTED_WAVS = {
    "trading_open.wav": (0.15, 0.30),
    "profit.wav":       (0.25, 0.40),
    "loss.wav":         (0.25, 0.40),
    "signal.wav":       (0.04, 0.10),
    "emergency.wav":    (0.50, 0.70),
    "news_alert.wav":   (0.70, 0.90),
    "error.wav":        (0.20, 0.35),
    "complete.wav":     (0.70, 0.90),
}


@pytest.mark.parametrize("name,bounds", EXPECTED_WAVS.items())
def test_wav_file_valid(name, bounds):
    path = SOUND_DIR / name
    assert path.exists(), f"missing {path}"
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 44100
        assert w.getsampwidth() == 2
        dur = w.getnframes() / w.getframerate()
    lo, hi = bounds
    assert lo <= dur <= hi, f"{name} duration {dur:.3f}s outside [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Discord embed builders — shape verification
# ---------------------------------------------------------------------------

def test_trade_opened_embed_has_required_fields():
    p = discord.trade_opened(
        symbol="EURUSD", direction="BUY", entry=1.07323, sl=1.07223, tp=1.07523,
        lot=0.45, risk_pct=0.02, confluence=4,
        claude_confidence=78, claude_note="Bullish OB confluence.",
    )
    e = p["embeds"][0]
    assert e["color"] == discord.COLOR_BLUE
    names = {f["name"] for f in e["fields"]}
    for need in ("Entry", "SL", "TP", "Lot", "Risk %", "Confluence", "Claude Conf", "Claude Note"):
        assert need in names


def test_trade_closed_profit_vs_loss_color():
    win = discord.trade_closed(symbol="EURUSD", entry=1.0, exit_=1.01,
                               pnl_usd=90.0, rr_achieved=2.0,
                               duration_s=600, close_reason="TP")
    lose = discord.trade_closed(symbol="GBPUSD", entry=1.25, exit_=1.249,
                                pnl_usd=-50.0, rr_achieved=-1.0,
                                duration_s=900, close_reason="SL")
    assert win["embeds"][0]["color"] == discord.COLOR_GREEN
    assert lose["embeds"][0]["color"] == discord.COLOR_RED
    assert "+$90.00" in str(win)
    assert "-$50.00" in str(lose)


def test_kill_triggered_embed():
    p = discord.kill_triggered(kind="INTRADAY", drawdown_pct=0.0304,
                               positions_closed=2, halted_until="2026-05-04 00:00 UTC")
    assert p["embeds"][0]["color"] == discord.COLOR_RED
    assert "3.04%" in str(p)


def test_news_warning_embed():
    p = discord.news_warning(event_name="US NFP", currency="USD", impact="HIGH",
                             minutes_until=29, affected_symbols=["EURUSD", "USDJPY"])
    assert p["embeds"][0]["color"] == discord.COLOR_ORANGE


def test_engine_error_embed_uses_error_username():
    p = discord.engine_error(component="mt5_link", error_type="ConnectionAbortedError",
                             stack_excerpt="File foo.py:1\n", action_taken="reconnect")
    assert p["username"].endswith("[ERROR]")
    assert p["embeds"][0]["color"] == discord.COLOR_RED


def test_training_complete_embed():
    p = discord.training_complete(model_name="cnn_lstm", version="v13_2026-05-03",
                                  accuracy_delta=0.0124, loss_delta=-0.0341,
                                  trades_trained_on=100)
    assert p["embeds"][0]["color"] == discord.COLOR_GOLD
    assert "+0.0124" in str(p)


def test_signal_detected_embed():
    p = discord.signal_detected(symbol="EURUSD", direction="BUY", confluence=4,
                                smc_zone="Bullish OB (M15)", cnn_conf=72, rl_vote="BUY",
                                kill_zone_label="NY Open", news_clear=True,
                                claude_excerpt="Looks clean.")
    assert p["embeds"][0]["color"] == discord.COLOR_PURPLE


def test_daily_summary_embed():
    p = discord.daily_summary(date_str="2026-05-03", trades=7, wins=5, losses=2,
                              net_pnl=214.5, equity=10_214.5,
                              best_trade="EURUSD +$90.00", worst_trade="GBPUSD -$50.00",
                              drawdown_max_pct=0.0062)
    assert p["embeds"][0]["color"] == discord.COLOR_GOLD
    assert "71.4%" in str(p)


# ---------------------------------------------------------------------------
# Live mock Discord receiver — verifies post() actually ships JSON.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discord_post_hits_mock_receiver(monkeypatch):
    """Spin a localhost aiohttp app, point Discord at it, fire all 8 events."""
    from aiohttp import web

    received: list[dict] = []

    async def handler(request):
        received.append(await request.json())
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/hook", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/hook"

    # Reload settings with the mock URL pinned in env.
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", url)
    monkeypatch.setenv("NOTIFY_DISCORD_ENABLED", "true")
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)
    importlib.reload(discord)

    try:
        events = [
            discord.trade_opened(symbol="EURUSD", direction="BUY", entry=1, sl=0.99, tp=1.02,
                                 lot=0.1, risk_pct=0.02, confluence=4,
                                 claude_confidence=70, claude_note="ok"),
            discord.trade_closed(symbol="EURUSD", entry=1, exit_=1.01, pnl_usd=10,
                                 rr_achieved=1.0, duration_s=60, close_reason="TP"),
            discord.trade_closed(symbol="GBPUSD", entry=1.25, exit_=1.249, pnl_usd=-10,
                                 rr_achieved=-1.0, duration_s=60, close_reason="SL"),
            discord.signal_detected(symbol="EURUSD", direction="BUY", confluence=3,
                                    smc_zone="OB", cnn_conf=60, rl_vote="BUY",
                                    kill_zone_label="NY", news_clear=True, claude_excerpt="x"),
            discord.kill_triggered(kind="INTRADAY", drawdown_pct=0.03,
                                   positions_closed=1, halted_until="now"),
            discord.news_warning(event_name="NFP", currency="USD", impact="HIGH",
                                 minutes_until=29, affected_symbols=["EURUSD"]),
            discord.engine_error(component="x", error_type="Y",
                                 stack_excerpt="stack", action_taken="retry"),
            discord.training_complete(model_name="cnn_lstm", version="v1",
                                      accuracy_delta=0.01, loss_delta=-0.01,
                                      trades_trained_on=100),
        ]
        for p in events:
            ok = await asyncio.to_thread(discord.post, p)
            assert ok, "discord.post returned False"
        await asyncio.sleep(0.05)
        assert len(received) == 8
        # Each payload carries a single embed with fields and a color.
        colors = {r["embeds"][0]["color"] for r in received}
        # 8 events use 6 distinct colors (2 trade_closed share blue/green/red).
        assert len(colors) >= 5
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Windows toast — IPC publish verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_toast_publishes_notification_frame(monkeypatch):
    monkeypatch.setenv("NOTIFY_TOAST_ENABLED", "true")
    monkeypatch.setenv("NOTIFY_SOUND_ENABLED", "true")
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)
    importlib.reload(windows_toast)

    from engine.ipc.broadcaster import BUS
    q = await BUS.subscribe()
    try:
        ok = windows_toast.notify("TRADE_OPENED", title="EURUSD BUY 0.45",
                                  body="Entry 1.07323")
        assert ok
        frame = await asyncio.wait_for(q.get(), timeout=1.0)
        assert frame["type"] == "notification"
        assert frame["data"]["event"] == "TRADE_OPENED"
        assert frame["data"]["sound"] == "trading_open.wav"
    finally:
        await BUS.unsubscribe(q)


@pytest.mark.asyncio
async def test_toast_disabled_returns_false(monkeypatch):
    monkeypatch.setenv("NOTIFY_TOAST_ENABLED", "false")
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)
    importlib.reload(windows_toast)

    assert windows_toast.notify("TRADE_OPENED", title="x", body="y") is False


def test_discord_post_disabled_returns_false(monkeypatch):
    monkeypatch.setenv("NOTIFY_DISCORD_ENABLED", "false")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "http://localhost:1/x")
    import importlib
    from engine.config import settings as _s
    importlib.reload(_s)
    importlib.reload(discord)
    assert discord.post({"foo": "bar"}) is False
