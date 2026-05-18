"""Tests for Claude API cost tracker (Tier 8.9)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.notifications.claude_cost import (
    HARD_BUDGET_USD,
    SOFT_BUDGET_USD,
    daily_history,
    estimate_cost,
    is_over_budget,
    read_today,
    record_call,
    reset_today,
)


@pytest.fixture()
def tmp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield str(Path(d) / "j.sqlite")


def test_estimate_cost_sonnet_4_6():
    cost = estimate_cost(1_000_000, 0, model="claude-sonnet-4-6")
    assert cost == pytest.approx(3.00)
    cost = estimate_cost(0, 1_000_000, model="claude-sonnet-4-6")
    assert cost == pytest.approx(15.00)


def test_estimate_cost_unknown_falls_back_to_opus():
    a = estimate_cost(1_000_000, 0, model="totally-fake-model")
    b = estimate_cost(1_000_000, 0, model="claude-opus-4-7")
    assert a == pytest.approx(b)


def test_record_call_increments_counters(tmp_db):
    rec = record_call(model="claude-sonnet-4-6", input_tokens=1000,
                       output_tokens=200, db_path=tmp_db)
    assert rec.input_tokens == 1000
    assert rec.output_tokens == 200
    assert rec.calls == 1
    assert rec.cost_usd > 0
    rec2 = record_call(model="claude-sonnet-4-6", input_tokens=500,
                        output_tokens=100, db_path=tmp_db)
    assert rec2.input_tokens == 1500
    assert rec2.calls == 2


def test_read_today_empty(tmp_db):
    rec = read_today(db_path=tmp_db)
    assert rec.input_tokens == 0
    assert rec.cost_usd == 0.0


def test_is_over_budget_hard(tmp_db):
    # Crank up tokens until we blow past HARD budget.
    needed_input = int((HARD_BUDGET_USD / 3.00) * 1_000_000) + 1
    record_call(model="claude-sonnet-4-6", input_tokens=needed_input,
                 output_tokens=0, db_path=tmp_db)
    assert is_over_budget(hard=True, db_path=tmp_db)


def test_is_over_budget_soft_only(tmp_db):
    needed = int(((SOFT_BUDGET_USD + 0.01) / 3.00) * 1_000_000)
    record_call(model="claude-sonnet-4-6", input_tokens=needed,
                 output_tokens=0, db_path=tmp_db)
    assert is_over_budget(hard=False, db_path=tmp_db)


def test_reset_today_zeroes_counters(tmp_db):
    record_call(model="claude-sonnet-4-6", input_tokens=1000,
                 output_tokens=200, db_path=tmp_db)
    reset_today(db_path=tmp_db)
    rec = read_today(db_path=tmp_db)
    assert rec.cost_usd == 0.0
    assert rec.calls == 0


def test_daily_history_returns_requested_days(tmp_db):
    hist = daily_history(days=5, db_path=tmp_db)
    assert len(hist) == 5
    assert all(r.cost_usd == 0.0 for r in hist)
