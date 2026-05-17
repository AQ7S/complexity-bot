from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from loguru import logger

from engine.config import settings
from engine.data.event_log import log_event


CHECK_INTERVAL_S = 30
MAX_FAILURES = 3
START_GRACE_PERIOD_S = 15
HEALTH_TIMEOUT_S = 5


def _health_url() -> str:
    host = getattr(settings, "IPC_HOST", "127.0.0.1")
    port = getattr(settings, "IPC_HEALTH_PORT", 8766)
    return f"http://{host}:{port}/health"


def _discord_alert(message: str) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if webhook in ("", "unset"):
        return
    try:
        requests.post(webhook, json={
            "username": "Complexity Engine Watchdog",
            "embeds": [{
                "title": "Watchdog Alert",
                "description": message,
                "color": 15105570,
            }]
        }, timeout=5)
    except Exception as e:
        logger.warning("watchdog discord post failed: {}", e)


def start_engine_process() -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parent.parent
    engine_path = repo_root / "engine" / "engine.py"
    python_exe = sys.executable
    logger.info("watchdog starting engine: {} {}", python_exe, engine_path)
    return subprocess.Popen(
        [python_exe, str(engine_path)],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ping_engine() -> bool:
    try:
        r = requests.get(_health_url(), timeout=HEALTH_TIMEOUT_S)
        if r.status_code != 200:
            return False
        body = r.json()
        return bool(body.get("ok", body.get("status") == "ok"))
    except Exception:
        return False


def run_supervisor(max_iters: int | None = None) -> None:
    proc = start_engine_process()
    log_event("WATCHDOG_RESTART", None, {"pid": proc.pid, "phase": "initial_start"})
    failures = 0
    iters = 0
    time.sleep(START_GRACE_PERIOD_S)
    while True:
        if max_iters is not None and iters >= max_iters:
            break
        iters += 1
        time.sleep(CHECK_INTERVAL_S)
        if proc.poll() is not None:
            logger.error("watchdog: engine process exited code={}", proc.returncode)
            _discord_alert(f"Engine process exited (code {proc.returncode}). Restarting…")
            log_event("WATCHDOG_RESTART", None, {
                "phase": "process_exit",
                "exit_code": proc.returncode,
            })
            proc = start_engine_process()
            failures = 0
            time.sleep(START_GRACE_PERIOD_S)
            continue
        if ping_engine():
            failures = 0
            continue
        failures += 1
        logger.warning("watchdog: health ping failed ({}/{})", failures, MAX_FAILURES)
        if failures < MAX_FAILURES:
            continue
        _discord_alert("Engine unresponsive after 3 health pings — restarting")
        log_event("WATCHDOG_RESTART", None, {
            "phase": "ping_timeout",
            "failures": failures,
        })
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception as e:
            logger.warning("watchdog kill failed: {}", e)
        time.sleep(2)
        proc = start_engine_process()
        failures = 0
        time.sleep(START_GRACE_PERIOD_S)


async def serve_health_endpoint(state) -> None:
    from aiohttp import web

    async def health(request):
        return web.json_response({
            "ok": True,
            "status": state.status if hasattr(state, "status") else "STARTING",
            "uptime_s": state.uptime_s if hasattr(state, "uptime_s") else 0,
            "mt5_connected": getattr(state, "mt5_connected", False),
            "timestamp": int(time.time()),
        })

    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    host = getattr(settings, "IPC_HOST", "127.0.0.1")
    port = getattr(settings, "IPC_HEALTH_PORT", 8766)
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("watchdog /health endpoint live on http://{}:{}/health", host, port)


if __name__ == "__main__":
    run_supervisor()
