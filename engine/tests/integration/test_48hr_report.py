"""Phase 16 — 48-hour soak audit.

The actual 48-hour run is operator-driven; this test fabricates the on-disk
artefacts a healthy soak leaves behind and asserts the audit returns PASS,
then mutates one signal at a time to confirm each criterion fails
independently. That gives us a working CI gate without paying for two
days of wall-clock per change.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.data.sqlite_journal import init_db, open_journal


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _seed_trades(n: int, *, start: datetime, end: datetime) -> None:
    init_db()
    span = (end - start).total_seconds()
    with open_journal() as con:
        for i in range(n):
            ts = start + timedelta(seconds=span * (i + 1) / (n + 1))
            con.execute(
                """
                INSERT INTO trades(mt5_ticket, symbol, direction, entry_price,
                                   lot_size, sl, tp, open_time, close_time,
                                   close_reason, exit_price, pnl)
                VALUES(?, 'EURUSD', 'BUY', 1.07, 0.01, 1.06, 1.08, ?, ?,
                       'TP', 1.075, 5.0)
                """,
                (5_000_000 + i, ts.isoformat(), ts.isoformat()),
            )
        con.commit()


def _seed_notification_counts() -> None:
    from engine.utils.telemetry import NOTIFICATION_EVENTS, record_notification
    for e in NOTIFICATION_EVENTS:
        record_notification(e)


def _write_telemetry(path: Path, *, start: datetime, end: datetime,
                     rss_mb: float = 320.0, sys_cpu: float = 3.0,
                     proc_cpu: float = 8.0, n: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    span = (end - start).total_seconds()
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            ts = start + timedelta(seconds=span * (i + 1) / (n + 1))
            row = {
                "ts": ts.isoformat(),
                "rss_mb": rss_mb, "proc_cpu_pct": proc_cpu,
                "sys_cpu_pct": sys_cpu, "open_positions": 1, "bus_subscribers": 1,
            }
            fh.write(json.dumps(row) + "\n")


def _write_clean_log(path: Path, *, start: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{start.strftime('%Y-%m-%d %H:%M:%S')} | INFO | engine ready\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def soak_env(tmp_path, monkeypatch):
    """Spin up an isolated SQLite + telemetry/log paths and reload modules."""
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "journal.sqlite"))
    from engine.config import settings as _s
    importlib.reload(_s)
    from engine.data import sqlite_journal
    importlib.reload(sqlite_journal)
    from engine.utils import telemetry
    importlib.reload(telemetry)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=48)
    return {
        "tmp": tmp_path, "start": start, "end": end,
        "telemetry_path": tmp_path / "telemetry.jsonl",
        "log_path":       tmp_path / "engine.log",
        "telemetry": telemetry,
    }


def test_full_pass_path(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(8, start=soak_env["start"], end=soak_env["end"])
    _seed_notification_counts()
    _write_telemetry(soak_env["telemetry_path"], start=soak_env["start"], end=soak_env["end"])
    _write_clean_log(soak_env["log_path"], start=soak_env["start"])

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    assert verdict["passed"], verdict
    assert verdict["observed"]["trades"] == 8
    assert verdict["observed"]["exceptions"] == 0
    assert verdict["criteria"]["all_notifications_seen"]


def test_fails_when_too_few_trades(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(3, start=soak_env["start"], end=soak_env["end"])  # below threshold of 6
    _seed_notification_counts()
    _write_telemetry(soak_env["telemetry_path"], start=soak_env["start"], end=soak_env["end"])
    _write_clean_log(soak_env["log_path"], start=soak_env["start"])

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    assert not verdict["passed"]
    assert verdict["criteria"]["trades_executed_ok"] is False
    # Other criteria stay green so the report pinpoints the failure.
    assert verdict["criteria"]["no_unhandled_excepts"] is True
    assert verdict["criteria"]["memory_p99_ok"] is True


def test_fails_on_unhandled_exception(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(8, start=soak_env["start"], end=soak_env["end"])
    _seed_notification_counts()
    _write_telemetry(soak_env["telemetry_path"], start=soak_env["start"], end=soak_env["end"])
    mid = soak_env["start"] + timedelta(hours=1)
    soak_env["log_path"].write_text(
        f"{mid.strftime('%Y-%m-%d %H:%M:%S')} | ERROR | UNHANDLED Traceback in mt5_link\n",
        encoding="utf-8",
    )

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    assert not verdict["passed"]
    assert verdict["criteria"]["no_unhandled_excepts"] is False
    assert verdict["observed"]["exceptions"] >= 1


def test_fails_on_memory_blowup(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(8, start=soak_env["start"], end=soak_env["end"])
    _seed_notification_counts()
    _write_telemetry(soak_env["telemetry_path"],
                     start=soak_env["start"], end=soak_env["end"],
                     rss_mb=900.0)  # over the 600MB ceiling
    _write_clean_log(soak_env["log_path"], start=soak_env["start"])

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    assert not verdict["passed"]
    assert verdict["criteria"]["memory_p99_ok"] is False
    assert verdict["observed"]["rss_p99_mb"] == 900.0


def test_fails_when_notification_event_missing(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(8, start=soak_env["start"], end=soak_env["end"])
    # Skip ENGINE_ERROR so the criterion fails.
    for e in t.NOTIFICATION_EVENTS:
        if e == "ENGINE_ERROR":
            continue
        t.record_notification(e)
    _write_telemetry(soak_env["telemetry_path"], start=soak_env["start"], end=soak_env["end"])
    _write_clean_log(soak_env["log_path"], start=soak_env["start"])

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    assert not verdict["passed"]
    assert verdict["criteria"]["all_notifications_seen"] is False
    assert "ENGINE_ERROR" in verdict["observed"]["missing_notifications"]


def test_render_report_round_trip(soak_env):
    t = soak_env["telemetry"]
    _seed_trades(8, start=soak_env["start"], end=soak_env["end"])
    _seed_notification_counts()
    _write_telemetry(soak_env["telemetry_path"], start=soak_env["start"], end=soak_env["end"])
    _write_clean_log(soak_env["log_path"], start=soak_env["start"])

    verdict = t.audit_run(
        soak_env["start"], soak_env["end"],
        telemetry_path=soak_env["telemetry_path"],
        log_path=soak_env["log_path"],
        check_supabase=False,
    )
    text = t.render_report(verdict)
    assert "PASS" in text
    assert "trades" in text
    assert "all_notifications_seen" in text


def test_sampler_writes_jsonl_row(tmp_path):
    from engine.utils import telemetry as tmod
    importlib.reload(tmod)
    out = tmp_path / "telemetry.jsonl"
    s = tmod.Sampler(path=out)
    sample = s.tick()
    assert out.exists()
    line = out.read_text(encoding="utf-8").strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["rss_mb"] > 0
    assert obj["ts"] == sample.ts
    assert set(obj) >= set(asdict(sample))
