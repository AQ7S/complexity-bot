"""Tests for the MAE/MFE excursion tracker (Tier 4.4)."""
from __future__ import annotations

import tempfile

import pytest

from engine.data import sqlite_journal
from engine.learning.excursion_tracker import ExcursionTracker, load_excursion


def _journal_path() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    with sqlite_journal.open_journal(tmp.name) as _:
        pass
    return tmp.name


def test_open_initialises_state():
    t = ExcursionTracker()
    t.open(trade_id=1, entry_price=1.10, direction="BUY", point_size=0.0001, now=0.0)
    snap = t.snapshot(1)
    assert snap is not None
    assert snap.max_mae_pips == 0.0
    assert snap.max_mfe_pips == 0.0


def test_adverse_excursion_for_buy_when_price_drops():
    t = ExcursionTracker()
    t.open(trade_id=1, entry_price=1.10, direction="BUY", point_size=0.0001, now=0.0)
    t.update(1, 1.0980, now=10.0)  # 20 pips against
    snap = t.snapshot(1)
    assert snap is not None
    assert snap.max_mae_pips == pytest.approx(20.0, abs=0.5)
    assert snap.time_to_mae_s == 10


def test_favourable_excursion_for_buy_when_price_rises():
    t = ExcursionTracker()
    t.open(trade_id=1, entry_price=1.10, direction="BUY", point_size=0.0001, now=0.0)
    t.update(1, 1.1020, now=5.0)
    t.update(1, 1.1050, now=15.0)
    snap = t.snapshot(1)
    assert snap is not None
    assert snap.max_mfe_pips == pytest.approx(50.0, abs=0.5)
    assert snap.time_to_mfe_s == 15


def test_sell_excursion_signs_reversed():
    t = ExcursionTracker()
    t.open(trade_id=1, entry_price=1.10, direction="SELL", point_size=0.0001, now=0.0)
    t.update(1, 1.1030, now=10.0)  # 30 pips against a SELL
    t.update(1, 1.0950, now=20.0)  # 50 pips in favour of a SELL
    snap = t.snapshot(1)
    assert snap.max_mae_pips == pytest.approx(30.0, abs=0.5)
    assert snap.max_mfe_pips == pytest.approx(50.0, abs=0.5)


def test_close_persists_and_clears():
    db = _journal_path()
    t = ExcursionTracker()
    t.open(trade_id=7, entry_price=1.20, direction="BUY", point_size=0.0001, now=0.0)
    t.update(7, 1.1980, now=12.0)
    final = t.close(7, db_path=db)
    assert final is not None
    assert t.snapshot(7) is None
    persisted = load_excursion(7, db_path=db)
    assert persisted is not None
    assert persisted.max_mae_pips == pytest.approx(20.0, abs=0.5)


def test_close_unknown_trade_id_returns_none():
    t = ExcursionTracker()
    assert t.close(999) is None


def test_update_unknown_trade_id_is_safe():
    t = ExcursionTracker()
    t.update(999, 1.0)
    assert t.snapshot(999) is None
