"""Tests for stress-test replay (Tier 4.3)."""
from __future__ import annotations

import pytest

from engine.strategy.stress_test import (
    SCENARIOS,
    StressPosition,
    replay_all,
    replay_scenario,
)


def _long_eurusd(notional: float = 50_000) -> StressPosition:
    return StressPosition("EURUSD#", "BUY", notional)


def test_all_five_scenarios_present():
    expected = {"BLACK_MONDAY_1987", "FLASH_CRASH_2010", "CHF_DEPEG_2015",
                "COVID_MARCH_2020", "SVB_MARCH_2023"}
    assert set(SCENARIOS) == expected


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        replay_scenario("DOES_NOT_EXIST", [], starting_equity=10_000)  # type: ignore[arg-type]


def test_black_monday_inflicts_loss():
    r = replay_scenario("BLACK_MONDAY_1987", [_long_eurusd(50_000)], starting_equity=10_000)
    assert r.pnl_usd < 0
    assert r.drawdown_pct > 0
    assert r.equity_after < r.equity_before


def test_short_position_profits_from_crash():
    r = replay_scenario("BLACK_MONDAY_1987",
                        [StressPosition("EURUSD#", "SELL", 10_000)],
                        starting_equity=10_000)
    assert r.pnl_usd > 0
    assert r.drawdown_pct == 0


def test_intraday_kill_breached_at_high_loss():
    # 50k notional on -18% wildcard shock = $9,000 loss on $10,000 equity → -90%
    r = replay_scenario("BLACK_MONDAY_1987", [_long_eurusd(50_000)], starting_equity=10_000)
    assert r.drawdown_pct > 0.03
    assert r.intraday_kill_breached
    assert r.weekly_kill_breached


def test_margin_call_when_equity_wiped():
    r = replay_scenario("COVID_MARCH_2020",
                        [_long_eurusd(100_000)], starting_equity=1_000)
    assert r.equity_after < 0
    assert r.margin_call


def test_per_symbol_pnl_reported():
    positions = [
        StressPosition("EURUSD#", "BUY", 10_000),
        StressPosition("GOLD#", "BUY", 10_000),
    ]
    r = replay_scenario("COVID_MARCH_2020", positions, starting_equity=20_000)
    assert "EURUSD#" in r.per_symbol_pnl
    assert "GOLD#" in r.per_symbol_pnl
    # GOLD gets a +4% shock during COVID per the scenario.
    assert r.per_symbol_pnl["GOLD#"] > 0


def test_replay_all_returns_dict_keyed_by_scenario():
    reports = replay_all([_long_eurusd(10_000)], starting_equity=10_000)
    assert len(reports) == 5
    for name, r in reports.items():
        assert r.scenario == name


def test_empty_positions_no_damage():
    r = replay_scenario("BLACK_MONDAY_1987", [], starting_equity=10_000)
    assert r.pnl_usd == 0.0
    assert not r.margin_call
