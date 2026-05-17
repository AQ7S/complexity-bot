"""SQLite trade journal + Claude audit (per master plan §7).

Single connection per process is fine: SQLite locking is enabled with WAL
journal mode so concurrent reads from the UI side don't block writes.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from engine.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mt5_ticket INTEGER UNIQUE,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  lot_size REAL NOT NULL,
  sl REAL NOT NULL,
  tp REAL NOT NULL,
  pnl REAL,
  rr_achieved REAL,
  open_time TEXT NOT NULL,
  close_time TEXT,
  close_reason TEXT,
  signal_confluence INTEGER,
  claude_decision TEXT,
  claude_confidence INTEGER,
  claude_reasoning TEXT,
  synced_supabase INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claude_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id INTEGER,
  symbol TEXT NOT NULL,
  context_json TEXT NOT NULL,
  decision TEXT NOT NULL,
  confidence INTEGER NOT NULL,
  reasoning TEXT NOT NULL,
  risk_adjustment REAL NOT NULL,
  ts TEXT NOT NULL,
  synced_supabase INTEGER DEFAULT 0,
  FOREIGN KEY(trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS model_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  version TEXT NOT NULL,
  ts TEXT NOT NULL,
  accuracy REAL,
  loss REAL,
  sharpe REAL,
  trades_trained_on INTEGER,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS signals_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  smc_signal TEXT,
  cnn_signal TEXT,
  rl_signal TEXT,
  killzone_ok INTEGER,
  news_clear INTEGER,
  consensus_count INTEGER,
  outcome TEXT
);

CREATE TABLE IF NOT EXISTS settings_kv (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS price_alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  threshold REAL NOT NULL,
  enabled INTEGER DEFAULT 1,
  triggered_at TEXT
);

CREATE TABLE IF NOT EXISTS shadow_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL NOT NULL,
  sl_price REAL NOT NULL,
  tp_price REAL NOT NULL,
  claude_decision TEXT,
  claude_confidence INTEGER,
  confluence_score INTEGER,
  model_version TEXT,
  bars_held INTEGER DEFAULT 0,
  close_time TEXT,
  exit_price REAL,
  hypothetical_outcome TEXT,
  pnl_r REAL,
  pnl_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_shadow_open ON shadow_trades(hypothetical_outcome);

CREATE TABLE IF NOT EXISTS calibration_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  ece_score REAL NOT NULL,
  n_trades INTEGER NOT NULL,
  bin_data_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calibration_ts ON calibration_history(timestamp DESC);

CREATE TABLE IF NOT EXISTS signal_features (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id INTEGER,
  shadow_id INTEGER,
  ts TEXT NOT NULL,
  symbol TEXT NOT NULL,
  features_json TEXT NOT NULL,
  FOREIGN KEY(trade_id) REFERENCES trades(id),
  FOREIGN KEY(shadow_id) REFERENCES shadow_trades(id)
);
CREATE INDEX IF NOT EXISTS idx_signal_features_trade ON signal_features(trade_id);
CREATE INDEX IF NOT EXISTS idx_signal_features_shadow ON signal_features(shadow_id);

CREATE TABLE IF NOT EXISTS trade_excursions (
  trade_id INTEGER PRIMARY KEY,
  max_mae_pips REAL,
  max_mfe_pips REAL,
  time_to_mae_s INTEGER,
  time_to_mfe_s INTEGER,
  FOREIGN KEY(trade_id) REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS claude_overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  param TEXT NOT NULL,
  old_value TEXT,
  new_value TEXT NOT NULL,
  rationale TEXT,
  expires_at TEXT,
  active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_overrides_active ON claude_overrides(active, ts DESC);

CREATE TABLE IF NOT EXISTS strategy_pnl (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  trades INTEGER DEFAULT 0,
  wins INTEGER DEFAULT 0,
  losses INTEGER DEFAULT 0,
  pnl_usd REAL DEFAULT 0,
  pnl_r REAL DEFAULT 0,
  notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_strategy_pnl ON strategy_pnl(strategy, ts DESC);
"""


def _resolve_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        db_path = settings.SQLITE_PATH
    return Path(db_path).resolve()


def init_db(db_path: str | Path | None = None) -> None:
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA_SQL)
        con.execute("PRAGMA journal_mode=WAL;")


@contextmanager
def open_journal(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = _resolve_path(db_path)
    init_db(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def insert_claude_decision(
    con: sqlite3.Connection,
    *,
    trade_id: int | None,
    symbol: str,
    context: dict,
    decision: str,
    confidence: int,
    reasoning: str,
    risk_adjustment: float,
) -> int:
    cur = con.execute(
        """
        INSERT INTO claude_decisions
          (trade_id, symbol, context_json, decision, confidence, reasoning, risk_adjustment, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_id, symbol, json.dumps(context), decision, int(confidence),
            reasoning, float(risk_adjustment), _now_iso(),
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def insert_signal_log(
    con: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    smc_signal: str | None,
    cnn_signal: str | None,
    rl_signal: str | None,
    killzone_ok: bool,
    news_clear: bool,
    consensus_count: int,
    outcome: str,
) -> int:
    cur = con.execute(
        """
        INSERT INTO signals_log
          (ts, symbol, timeframe, smc_signal, cnn_signal, rl_signal,
           killzone_ok, news_clear, consensus_count, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_iso(), symbol, timeframe, smc_signal, cnn_signal, rl_signal,
            int(bool(killzone_ok)), int(bool(news_clear)), int(consensus_count), outcome,
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def fetch_unsynced_claude_decisions(con: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM claude_decisions WHERE synced_supabase=0 ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_claude_decisions_synced(con: sqlite3.Connection, ids: list[int]) -> None:
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    con.execute(
        f"UPDATE claude_decisions SET synced_supabase=1 WHERE id IN ({placeholders})",
        ids,
    )
    con.commit()
