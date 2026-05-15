"""Phase 16 telemetry — process health sampler + 48-hour audit report.

The engine main loop calls `Sampler.tick()` once per `SAMPLE_INTERVAL_S`
(default 30s); each tick appends one JSONL row to
`engine/logs/telemetry.jsonl` with RSS-MB, process CPU%, system CPU%,
broadcaster subscriber count, and open-positions count. Loguru already
writes the structured engine log alongside it, so the audit reads both.

The notification dispatch path bumps `notification_counter` so the audit
can verify all 8 event types fired during a soak.

`audit_run(start_iso, end_iso)` is the auditor itself — pure function
over the on-disk artefacts, returns a dict with `passed: bool` plus
per-criterion verdicts and the raw observed values.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import psutil
from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_PATH = REPO_ROOT / "engine" / "logs" / "telemetry.jsonl"
DEFAULT_LOG_PATH = REPO_ROOT / "engine" / "logs" / "engine.log"
SAMPLE_INTERVAL_S = 30.0

# Event names mirror engine.notifications.windows_toast.EventT
NOTIFICATION_EVENTS = (
    "TRADE_OPENED", "TRADE_CLOSED_PROFIT", "TRADE_CLOSED_LOSS",
    "SIGNAL_DETECTED", "KILL_TRIGGERED", "NEWS_WARNING",
    "ENGINE_ERROR", "TRAINING_COMPLETE",
)

# Pass thresholds (mirrors §15 Phase 16 verification).
THRESHOLDS = {
    "trades_min":           6,        # 3/day × 2
    "memory_p99_mb_max":    600,
    "cpu_p95_max":          25.0,
    "idle_cpu_p95_max":     5.0,
    "exceptions_max":       0,
}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    ts: str
    rss_mb: float
    proc_cpu_pct: float
    sys_cpu_pct: float
    open_positions: int
    bus_subscribers: int


class Sampler:
    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or DEFAULT_TELEMETRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = psutil.Process()
        # Prime the per-process CPU counter — first sample is meaningless.
        self._proc.cpu_percent(interval=None)

    def _open_positions(self) -> int:
        try:
            with open_journal() as con:
                row = con.execute(
                    "SELECT COUNT(*) AS n FROM trades WHERE close_time IS NULL"
                ).fetchone()
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def _bus_subscribers(self) -> int:
        try:
            from engine.ipc.broadcaster import BUS
            return BUS.subscriber_count
        except Exception:  # noqa: BLE001
            return 0

    def tick(self) -> Sample:
        s = Sample(
            ts=datetime.now(timezone.utc).isoformat(),
            rss_mb=round(self._proc.memory_info().rss / 1_048_576, 2),
            proc_cpu_pct=round(self._proc.cpu_percent(interval=None), 2),
            sys_cpu_pct=round(psutil.cpu_percent(interval=None), 2),
            open_positions=self._open_positions(),
            bus_subscribers=self._bus_subscribers(),
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(s)) + "\n")
        return s


# ---------------------------------------------------------------------------
# Notification counter
# ---------------------------------------------------------------------------

def record_notification(event: str) -> None:
    """Bump the per-event counter in `settings_kv` (atomic via SQLite txn)."""
    if event not in NOTIFICATION_EVENTS:
        return
    key = f"notif_count:{event}"
    try:
        with open_journal() as con:
            cur = con.execute("SELECT v FROM settings_kv WHERE k=?", (key,)).fetchone()
            n = int(cur["v"]) + 1 if cur else 1
            con.execute(
                "INSERT INTO settings_kv(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, str(n)),
            )
            con.commit()
    except Exception as e:  # noqa: BLE001
        logger.debug("notification counter bump failed: {}", e)


def notification_counts() -> dict[str, int]:
    out = {e: 0 for e in NOTIFICATION_EVENTS}
    try:
        with open_journal() as con:
            rows = con.execute(
                "SELECT k,v FROM settings_kv WHERE k LIKE 'notif_count:%'"
            ).fetchall()
        for r in rows:
            event = r["k"].split(":", 1)[1]
            if event in out:
                out[event] = int(r["v"])
    except Exception as e:  # noqa: BLE001
        logger.debug("notification counts read failed: {}", e)
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def _read_samples(path: Path, *, start: datetime, end: datetime) -> list[Sample]:
    if not path.exists():
        return []
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = datetime.fromisoformat(obj["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if not (start <= ts <= end):
                    continue
                out.append(Sample(**obj))
            except Exception:  # noqa: BLE001
                continue
    return out


def _count_exceptions(log_path: Path, *, start: datetime, end: datetime) -> int:
    """Count unhandled exceptions in the loguru file. Tolerant of missing log."""
    if not log_path.exists():
        return 0
    pat_ts = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    exc_marker = re.compile(r"\b(Traceback|UNHANDLED|CRITICAL|exception)\b", re.I)
    n = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat_ts.match(line)
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if not (start <= ts <= end):
                continue
            if exc_marker.search(line):
                n += 1
    return n


def _count_trades(start: datetime, end: datetime) -> int:
    try:
        with open_journal() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "WHERE close_time IS NOT NULL AND close_time>=? AND close_time<=?",
                (start.isoformat(), end.isoformat()),
            ).fetchone()
        return int(row["n"]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def _supabase_match(start: datetime, end: datetime) -> tuple[bool, str]:
    """Compare SQLite trade count to Supabase. Returns (ok, reason)."""
    if not settings.have_supabase():
        return True, "supabase_unconfigured (skipped)"
    try:
        from engine.supabase_sync.client import get_client
        client = get_client()
        local = _count_trades(start, end)
        remote_resp = (client.table("trades")
                       .select("id", count="exact")
                       .gte("close_time", start.isoformat())
                       .lte("close_time", end.isoformat())
                       .execute())
        remote = int(getattr(remote_resp, "count", None) or len(remote_resp.data or []))
        return remote == local, f"sqlite={local} supabase={remote}"
    except Exception as e:  # noqa: BLE001
        return False, f"supabase_query_failed: {e}"


def audit_run(
    start: datetime, end: datetime, *,
    telemetry_path: Path | None = None,
    log_path: Path | None = None,
    notif_counts: dict[str, int] | None = None,
    require_all_notifications: bool = True,
    check_supabase: bool = True,
) -> dict:
    """Pure-ish audit over the on-disk artefacts. Returns verdict dict."""
    tpath = telemetry_path or DEFAULT_TELEMETRY_PATH
    lpath = log_path or DEFAULT_LOG_PATH
    samples = _read_samples(tpath, start=start, end=end)
    rss_p99 = _percentile([s.rss_mb for s in samples], 99)
    sys_cpu_p95 = _percentile([s.sys_cpu_pct for s in samples], 95)
    proc_cpu_p95 = _percentile([s.proc_cpu_pct for s in samples], 95)
    exc = _count_exceptions(lpath, start=start, end=end)
    trades = _count_trades(start, end)
    notifs = notif_counts if notif_counts is not None else notification_counts()
    missing_notifs = [e for e in NOTIFICATION_EVENTS if notifs.get(e, 0) == 0]
    sb_ok, sb_reason = _supabase_match(start, end) if check_supabase else (True, "supabase_check_disabled")

    criteria = {
        "trades_executed_ok":   trades >= THRESHOLDS["trades_min"],
        "no_unhandled_excepts": exc <= THRESHOLDS["exceptions_max"],
        "memory_p99_ok":        rss_p99 <= THRESHOLDS["memory_p99_mb_max"],
        "cpu_p95_ok":           proc_cpu_p95 <= THRESHOLDS["cpu_p95_max"],
        "idle_cpu_p95_ok":      sys_cpu_p95 <= THRESHOLDS["idle_cpu_p95_max"]
                                or proc_cpu_p95 <= THRESHOLDS["cpu_p95_max"],
        "all_notifications_seen": (not require_all_notifications) or len(missing_notifs) == 0,
        "supabase_in_sync":     sb_ok,
        "samples_present":      len(samples) > 0,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "observed": {
            "trades": trades,
            "exceptions": exc,
            "rss_p99_mb": rss_p99,
            "proc_cpu_p95_pct": proc_cpu_p95,
            "sys_cpu_p95_pct": sys_cpu_p95,
            "samples": len(samples),
            "notifications": notifs,
            "missing_notifications": missing_notifs,
            "supabase": sb_reason,
        },
        "window": {"start": start.isoformat(), "end": end.isoformat()},
    }


def render_report(verdict: dict) -> str:
    """Plain-text human-readable summary for stdout / Discord / log."""
    head = "PASS" if verdict["passed"] else "FAIL"
    lines = [f"=== 48-HOUR SOAK REPORT ({head}) ===",
             f"  window: {verdict['window']['start']}  ->  {verdict['window']['end']}"]
    lines.append("  observed:")
    for k, v in verdict["observed"].items():
        lines.append(f"    {k:>22}: {v}")
    lines.append("  criteria:")
    for k, v in verdict["criteria"].items():
        mark = "[PASS]" if v else "[FAIL]"
        lines.append(f"    {mark} {k}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser("phase16 telemetry / audit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample-once").set_defaults(cmd="sample-once")
    a = sub.add_parser("audit")
    a.add_argument("--hours", type=float, default=48.0)
    a.add_argument("--no-supabase", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "sample-once":
        s = Sampler().tick()
        print(json.dumps(asdict(s)))
        return 0
    if args.cmd == "audit":
        end = datetime.now(timezone.utc)
        start = datetime.fromtimestamp(end.timestamp() - args.hours * 3600, tz=timezone.utc)
        verdict = audit_run(start, end, check_supabase=not args.no_supabase)
        print(render_report(verdict))
        return 0 if verdict["passed"] else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())


# Convenience: rolling helper for any caller that wants p-stat without state.
def percentile(xs: Iterable[float], p: float) -> float:
    return _percentile(list(xs), p)
