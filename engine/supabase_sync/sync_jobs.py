"""Batch sync jobs from local SQLite → Supabase."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from loguru import logger

from engine.data.sqlite_journal import (
    fetch_unsynced_claude_decisions, mark_claude_decisions_synced, open_journal,
)
from engine.supabase_sync.client import get_client


def _claude_row_to_supabase(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_id": row.get("trade_id"),
        "symbol": row["symbol"],
        "context_json": json.loads(row["context_json"]) if isinstance(row["context_json"], str) else row["context_json"],
        "decision": row["decision"],
        "confidence": int(row["confidence"]),
        "reasoning": row["reasoning"],
        "risk_adjustment": float(row["risk_adjustment"]),
        "ts": row["ts"],
    }


def sync_claude_decisions(limit: int = 100) -> int:
    """Push unsynced rows; mark them synced on success. Returns rows pushed."""
    client = get_client()
    with open_journal() as con:
        rows = fetch_unsynced_claude_decisions(con, limit=limit)
        if not rows:
            return 0
        payload = [_claude_row_to_supabase(r) for r in rows]
        client.table("claude_decisions").insert(payload).execute()
        mark_claude_decisions_synced(con, [r["id"] for r in rows])
        logger.info("Supabase: synced {} claude_decisions", len(rows))
        return len(rows)


def insert_claude_decision_remote(row: dict[str, Any]) -> dict | None:
    """One-shot insert (used by the live test). Returns the inserted record."""
    client = get_client()
    res = client.table("claude_decisions").insert(_claude_row_to_supabase(row)).execute()
    return (res.data or [None])[0]
