"""Feature snapshotting for live retraining.

The online LightGBM retrainer needs the exact feature vector that drove
each historical signal — not the OHLCV bars, but the post-pipeline 50-
column inference input. Recomputing those features 5000 trades later
would be slow AND vulnerable to subtle indicator-library version drift.

Strategy: snapshot the feature vector at the moment of signal admission,
join it to the trade row when the trade closes, and use the joined table
as the LightGBM training set.

Two hooks:

  * `snapshot_for_signal(features, symbol, trade_id=None, shadow_id=None)` —
    called by `consensus.evaluate()` immediately after a signal passes
    the gate, before the order is sent.
  * `attach_to_trade(snapshot_id, trade_id)` — called by the order
    router once an actual `mt5_ticket` is assigned, linking the snapshot
    to the durable trade row.

Both write to the `signal_features` table created in the journal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from engine.data.sqlite_journal import open_journal


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_id: int
    ts: str
    symbol: str
    n_features: int
    trade_id: int | None = None
    shadow_id: int | None = None


def snapshot_for_signal(
    features: Iterable[float],
    *,
    symbol: str,
    trade_id: int | None = None,
    shadow_id: int | None = None,
    db_path: str | None = None,
) -> int:
    """Persist a feature vector for later retrain joins. Returns the row id."""
    vec = [float(v) for v in features]
    if not vec:
        return 0
    payload = json.dumps(vec, separators=(",", ":"))
    ts = datetime.now(timezone.utc).isoformat()
    with open_journal(db_path) as con:
        cur = con.execute(
            "INSERT INTO signal_features (trade_id, shadow_id, ts, symbol, features_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (trade_id, shadow_id, ts, symbol, payload),
        )
        con.commit()
        return int(cur.lastrowid or 0)


def attach_to_trade(snapshot_id: int, trade_id: int, *, db_path: str | None = None) -> None:
    """Link a previously written snapshot to a durable trade row."""
    if snapshot_id <= 0 or trade_id <= 0:
        return
    with open_journal(db_path) as con:
        con.execute(
            "UPDATE signal_features SET trade_id=? WHERE id=?",
            (int(trade_id), int(snapshot_id)),
        )
        con.commit()


def attach_to_shadow(snapshot_id: int, shadow_id: int, *, db_path: str | None = None) -> None:
    if snapshot_id <= 0 or shadow_id <= 0:
        return
    with open_journal(db_path) as con:
        con.execute(
            "UPDATE signal_features SET shadow_id=? WHERE id=?",
            (int(shadow_id), int(snapshot_id)),
        )
        con.commit()


def load_snapshot(snapshot_id: int, *, db_path: str | None = None) -> FeatureSnapshot | None:
    with open_journal(db_path) as con:
        row = con.execute(
            "SELECT id, ts, symbol, trade_id, shadow_id, features_json "
            "FROM signal_features WHERE id=?",
            (int(snapshot_id),),
        ).fetchone()
    if row is None:
        return None
    try:
        n = len(json.loads(row["features_json"]))
    except (TypeError, json.JSONDecodeError):
        n = 0
    return FeatureSnapshot(
        snapshot_id=int(row["id"]),
        ts=str(row["ts"]),
        symbol=str(row["symbol"]),
        n_features=n,
        trade_id=row["trade_id"],
        shadow_id=row["shadow_id"],
    )


def count_snapshots(*, db_path: str | None = None) -> int:
    with open_journal(db_path) as con:
        row = con.execute("SELECT COUNT(*) AS n FROM signal_features").fetchone()
    return int(row["n"]) if row else 0
