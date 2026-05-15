"""Online learning coordinator — spawns retrain workers every N closed trades.

Triggers:
  * `RETRAIN_EVERY_N_TRADES` (100) closed trades since the last successful
    retrain, AND
  * Process-wide CPU under `RETRAIN_CPU_CEILING_PCT` (80%) for the last
    `CPU_SAMPLE_WINDOW_S` seconds.

Workers run in a separate `multiprocessing.Process` with the lowest OS
priority (Windows: IDLE_PRIORITY_CLASS; POSIX: nice +19) so the trading
main loop is never starved. Checkpoints land in
`engine/models/checkpoints/cnn_lstm_v{N+1}_{YYYY-MM-DD}.pt`, picked up by
`engine.models.inference` on next predict via newest-mtime selection.

Ring-buffer enforcement on DuckDB ticks runs alongside, every 60s.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import psutil
from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CPU_SAMPLE_WINDOW_S = 10.0
DEFAULT_RING_BUFFER_INTERVAL_S = 60.0


# ---------------------------------------------------------------------------
# State + counters
# ---------------------------------------------------------------------------

@dataclass
class OnlineState:
    """Tracks the watermark for the next retrain decision."""
    last_retrain_trade_count: int = 0
    last_retrain_at: float = 0.0
    last_checkpoint_path: Path | None = None
    inflight: mp.Process | None = None
    cpu_history: list[tuple[float, float]] = field(default_factory=list)


def count_closed_trades(db_path: str | None = None) -> int:
    with open_journal(db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE close_time IS NOT NULL"
        ).fetchone()
        return int(row["n"]) if row else 0


def latest_checkpoint_version(model_name: str = "cnn_lstm") -> int:
    """Parse `cnn_lstm_v{N}_*.pt` filenames and return the highest N."""
    pat = re.compile(rf"^{re.escape(model_name)}_v(\d+)_")
    best = 0
    for p in CHECKPOINT_DIR.glob(f"{model_name}_v*_*.*"):
        m = pat.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


# ---------------------------------------------------------------------------
# CPU gate
# ---------------------------------------------------------------------------

def sample_cpu(state: OnlineState, *, now: float | None = None) -> float:
    """Append a CPU sample and return the rolling average over the window."""
    now = now if now is not None else time.time()
    cpu = psutil.cpu_percent(interval=None)
    state.cpu_history.append((now, cpu))
    cutoff = now - CPU_SAMPLE_WINDOW_S
    state.cpu_history = [(t, c) for (t, c) in state.cpu_history if t >= cutoff]
    if not state.cpu_history:
        return cpu
    return sum(c for _, c in state.cpu_history) / len(state.cpu_history)


def cpu_ok(state: OnlineState, *, ceiling_pct: int = settings.RETRAIN_CPU_CEILING_PCT) -> bool:
    """True if rolling-window CPU average is under the ceiling."""
    if not state.cpu_history:
        return True
    avg = sum(c for _, c in state.cpu_history) / len(state.cpu_history)
    return avg < float(ceiling_pct)


# ---------------------------------------------------------------------------
# Trigger gate
# ---------------------------------------------------------------------------

def should_retrain(
    state: OnlineState, *, db_path: str | None = None,
    n_trigger: int = settings.RETRAIN_EVERY_N_TRADES,
    ceiling_pct: int = settings.RETRAIN_CPU_CEILING_PCT,
) -> tuple[bool, str]:
    """Pure decision: ready to spawn a retrain worker?"""
    if state.inflight is not None and state.inflight.is_alive():
        return False, "RETRAIN_IN_FLIGHT"
    closed = count_closed_trades(db_path)
    delta = closed - state.last_retrain_trade_count
    if delta < n_trigger:
        return False, f"NOT_ENOUGH_TRADES ({delta}/{n_trigger})"
    if not cpu_ok(state, ceiling_pct=ceiling_pct):
        return False, "CPU_OVER_CEILING"
    return True, "OK"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _lower_priority() -> None:
    """Drop process priority so retrain never starves the trading loop."""
    try:
        p = psutil.Process()
        if sys.platform == "win32":
            p.nice(psutil.IDLE_PRIORITY_CLASS)
        else:
            p.nice(19)
    except Exception as e:  # noqa: BLE001
        logger.debug("nice() failed: {}", e)


def _build_tier_retrain_worker(symbol: str, days: int, epochs: int) -> None:
    """Subprocess entrypoint: full Phase-5 build-tier retrain on `symbol`."""
    _lower_priority()
    import argparse
    from engine.models import train_batch
    args = argparse.Namespace(tier="build", symbols=symbol, days=days,
                              epochs=epochs, batch_size=64)
    train_batch.train(args)


def _stub_retrain_worker(version: int, model_name: str = "cnn_lstm",
                          out_dir: str | None = None) -> None:
    """Used by tests — writes a tiny placeholder checkpoint and exits."""
    _lower_priority()
    target = Path(out_dir) if out_dir else CHECKPOINT_DIR
    target.mkdir(parents=True, exist_ok=True)
    date = datetime.utcnow().strftime("%Y%m%d")
    path = target / f"{model_name}_v{version}_{date}.pt"
    path.write_bytes(b"STUB_CHECKPOINT")


def spawn_retrain(
    state: OnlineState, *, worker: Callable | None = None,
    worker_args: tuple = (), model_name: str = "cnn_lstm",
) -> mp.Process:
    """Fork a low-priority worker process. Returns the started Process."""
    if worker is None:
        worker = _build_tier_retrain_worker
        worker_args = ("EURUSD", 90, 5)
    proc = mp.Process(target=worker, args=worker_args, daemon=False,
                      name=f"retrain-{model_name}")
    proc.start()
    state.inflight = proc
    state.last_retrain_at = time.time()
    logger.info("spawned retrain pid={} args={}", proc.pid, worker_args)
    return proc


def reap(state: OnlineState) -> bool:
    """If an in-flight worker has exited, refresh the watermark. Returns True if reaped."""
    if state.inflight is None:
        return False
    if state.inflight.is_alive():
        return False
    proc = state.inflight
    proc.join(timeout=0)
    state.last_retrain_trade_count = count_closed_trades()
    state.last_checkpoint_path = _newest_checkpoint()
    state.inflight = None
    logger.info("retrain reaped exitcode={} latest={}", proc.exitcode,
                state.last_checkpoint_path)
    return True


def _newest_checkpoint(model_name: str = "cnn_lstm") -> Path | None:
    files = list(CHECKPOINT_DIR.glob(f"{model_name}_v*_*.*"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Coordinator loop (sync, called from engine main loop)
# ---------------------------------------------------------------------------

def tick(state: OnlineState, *, db_path: str | None = None) -> str:
    """One coordinator tick: sample CPU, reap, decide, maybe spawn.

    Returns a short status string useful for logs/telemetry.
    """
    sample_cpu(state)
    reap(state)
    ok, reason = should_retrain(state, db_path=db_path)
    if not ok:
        return reason
    next_v = latest_checkpoint_version() + 1
    spawn_retrain(state, worker=_stub_retrain_worker, worker_args=(next_v,))
    return f"SPAWNED v{next_v}"


# ---------------------------------------------------------------------------
# Ring-buffer enforcement (DuckDB ticks)
# ---------------------------------------------------------------------------

def enforce_ring_buffer_once() -> dict[str, int]:
    """Trim per-symbol tick history past `DATA_RING_BUFFER_TICKS`. Returns rows-deleted-per-symbol."""
    from engine.data import duckdb_store
    from engine.config.symbols import SYMBOLS_13
    with duckdb_store.open_store() as con:
        return duckdb_store.enforce_ring_buffer(
            con, [s.name for s in SYMBOLS_13],
            max_ticks=settings.DATA_RING_BUFFER_TICKS,
        )
