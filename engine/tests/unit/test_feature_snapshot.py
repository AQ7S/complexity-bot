"""Tests for feature snapshotting (Tier 5.1)."""
from __future__ import annotations

import tempfile

from engine.data import sqlite_journal
from engine.learning.feature_snapshot import (
    attach_to_shadow,
    attach_to_trade,
    count_snapshots,
    load_snapshot,
    snapshot_for_signal,
)


def _journal():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    with sqlite_journal.open_journal(tmp.name):
        pass
    return tmp.name


def test_snapshot_creates_row():
    db = _journal()
    sid = snapshot_for_signal([1.0, 2.0, 3.0], symbol="EURUSD#", db_path=db)
    assert sid > 0
    assert count_snapshots(db_path=db) == 1


def test_snapshot_empty_features_returns_zero():
    db = _journal()
    sid = snapshot_for_signal([], symbol="EURUSD#", db_path=db)
    assert sid == 0
    assert count_snapshots(db_path=db) == 0


def test_load_snapshot_returns_dataclass():
    db = _journal()
    sid = snapshot_for_signal([1.0] * 50, symbol="GOLD#", db_path=db)
    snap = load_snapshot(sid, db_path=db)
    assert snap is not None
    assert snap.symbol == "GOLD#"
    assert snap.n_features == 50


def test_attach_to_trade_updates_link():
    db = _journal()
    sid = snapshot_for_signal([0.5] * 10, symbol="EURUSD#", db_path=db)
    attach_to_trade(sid, trade_id=42, db_path=db)
    snap = load_snapshot(sid, db_path=db)
    assert snap is not None
    assert snap.trade_id == 42


def test_attach_to_shadow_updates_link():
    db = _journal()
    sid = snapshot_for_signal([0.1] * 10, symbol="GOLD#", db_path=db)
    attach_to_shadow(sid, shadow_id=99, db_path=db)
    snap = load_snapshot(sid, db_path=db)
    assert snap is not None
    assert snap.shadow_id == 99


def test_attach_with_zero_id_is_safe():
    db = _journal()
    sid = snapshot_for_signal([0.0] * 5, symbol="EURUSD#", db_path=db)
    attach_to_trade(sid, trade_id=0, db_path=db)
    attach_to_shadow(0, shadow_id=99, db_path=db)
    snap = load_snapshot(sid, db_path=db)
    assert snap is not None
    assert snap.trade_id is None


def test_load_unknown_id_returns_none():
    db = _journal()
    assert load_snapshot(99999, db_path=db) is None


def test_count_increments_per_signal():
    db = _journal()
    for i in range(5):
        snapshot_for_signal([float(i)] * 3, symbol="EURUSD#", db_path=db)
    assert count_snapshots(db_path=db) == 5
