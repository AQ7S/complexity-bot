from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
from loguru import logger

from engine.config import settings


DEFAULT_EVENT_DB_PATH = "./engine/data/store/events.duckdb"


EVENT_TYPES = (
    "SIGNAL_DETECTED", "SMC_FILTER_REJECT", "CONSENSUS_RESULT", "CLAUDE_DECISION",
    "TRADE_OPENED", "TRADE_CLOSED", "SPREAD_BLOCK", "CIRCUIT_BREAK",
    "NEWS_PAUSE", "RETRAIN_START", "RETRAIN_COMPLETE", "ENGINE_START",
    "MT5_RECONNECT", "OFFLINE_MODE_ON", "OFFLINE_MODE_OFF", "WATCHDOG_RESTART",
)


def _resolve_path(db_path: str | None = None) -> str:
    if db_path:
        return db_path
    return getattr(settings, "EVENT_LOG_PATH", DEFAULT_EVENT_DB_PATH)


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE SEQUENCE IF NOT EXISTS seq_engine_events START 1;
        CREATE TABLE IF NOT EXISTS engine_events (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_engine_events'),
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            data_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON engine_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_type ON engine_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_symbol ON engine_events(symbol);
        """
    )


@contextmanager
def open_event_log(db_path: str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    target = _resolve_path(db_path)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(target)
    try:
        _ensure_schema(con)
        yield con
    finally:
        con.close()


def log_event(
    event_type: str,
    symbol: str | None = None,
    data: dict[str, Any] | None = None,
    *,
    db_path: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(data or {}, default=str, separators=(",", ":"))
    try:
        with open_event_log(db_path) as con:
            con.execute(
                "INSERT INTO engine_events (timestamp, event_type, symbol, data_json) VALUES (?, ?, ?, ?)",
                [ts, str(event_type), symbol, payload],
            )
    except Exception as e:
        logger.warning("event_log insert failed type={} err={}", event_type, e)


def query_recent_events(
    *,
    limit: int = 200,
    event_type: str | None = None,
    symbol: str | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT id, timestamp, event_type, symbol, data_json FROM engine_events"
    clauses: list[str] = []
    params: list[Any] = []
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        with open_event_log(db_path) as con:
            rows = con.execute(sql, params).fetchall()
    except Exception as e:
        logger.warning("event_log query failed: {}", e)
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row[4]) if row[4] else {}
        except json.JSONDecodeError:
            payload = {}
        out.append({
            "id": int(row[0]),
            "timestamp": row[1],
            "event_type": row[2],
            "symbol": row[3],
            "data": payload,
        })
    return out


def purge_older_than(days: int, *, db_path: str | None = None) -> int:
    cutoff = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sql = "DELETE FROM engine_events WHERE timestamp < ?"
    try:
        with open_event_log(db_path) as con:
            res = con.execute(sql, [cutoff]).fetchone()
            return int(res[0]) if res else 0
    except Exception as e:
        logger.warning("event_log purge failed: {}", e)
        return 0


def bulk_log(rows: Iterable[tuple[str, str | None, dict[str, Any]]], *, db_path: str | None = None) -> int:
    payload: list[tuple[str, str, str | None, str]] = []
    now = datetime.now(timezone.utc).isoformat()
    for et, sym, data in rows:
        payload.append((now, et, sym, json.dumps(data, default=str, separators=(",", ":"))))
    if not payload:
        return 0
    try:
        with open_event_log(db_path) as con:
            con.executemany(
                "INSERT INTO engine_events (timestamp, event_type, symbol, data_json) VALUES (?, ?, ?, ?)",
                payload,
            )
            return len(payload)
    except Exception as e:
        logger.warning("event_log bulk_log failed: {}", e)
        return 0
