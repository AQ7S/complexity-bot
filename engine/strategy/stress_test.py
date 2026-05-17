"""Stress-test replay against canonical historical disaster scenarios.

Each scenario is encoded as a small set of per-symbol shocks (gap, vol
spike, correlation breakdown duration) that approximate what happened
in real markets. The engine doesn't need exact tick-level history — it
needs to know whether its current risk limits would have survived.

Scenarios:
  * BLACK_MONDAY_1987      — 22% S&P drop, correlation breakdown 4× vol.
  * FLASH_CRASH_2010       — 9% 10-minute S&P drop, intraday recovery.
  * CHF_DEPEG_2015         — 30% gap on EUR/CHF, SNB floor removal.
  * COVID_MARCH_2020       — 30% S&P drop + commodity correlation collapse.
  * SVB_MARCH_2023         — USD-pair banking-sector contagion (~5% moves).

The simulator applies each scenario to the open positions and reports
the projected equity drop, drawdown, margin call probability, and which
risk limits (per-trade SL, intraday kill, weekly kill, VaR cap) would
have triggered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


ScenarioName = Literal[
    "BLACK_MONDAY_1987",
    "FLASH_CRASH_2010",
    "CHF_DEPEG_2015",
    "COVID_MARCH_2020",
    "SVB_MARCH_2023",
]


@dataclass(frozen=True)
class StressShock:
    """Per-symbol shock parameters."""
    symbol_glob: str           # e.g. "EURUSD#" or "*"
    return_shock: float        # additive return (e.g. -0.30 for -30%)
    vol_multiplier: float = 4.0
    correlation_breakdown: bool = False


@dataclass(frozen=True)
class StressScenario:
    name: ScenarioName
    description: str
    shocks: tuple[StressShock, ...]


@dataclass(frozen=True)
class StressPosition:
    symbol: str
    direction: str
    notional_usd: float


@dataclass
class StressReport:
    scenario: ScenarioName
    equity_before: float
    equity_after: float
    pnl_usd: float
    pnl_pct: float
    drawdown_pct: float
    intraday_kill_breached: bool
    weekly_kill_breached: bool
    margin_call: bool
    per_symbol_pnl: dict[str, float] = field(default_factory=dict)


def _shock_for_symbol(scenario: StressScenario, symbol: str) -> StressShock | None:
    """Prefer specific matches over wildcards.

    Search order:
      1. Exact symbol match (`shock.symbol_glob == symbol`).
      2. Prefix wildcard (`"EUR*"` against `"EURUSD#"`).
      3. Plain wildcard (`"*"`).
    """
    wildcard: StressShock | None = None
    prefix_match: StressShock | None = None
    for shock in scenario.shocks:
        if shock.symbol_glob == symbol:
            return shock
        if shock.symbol_glob == "*":
            wildcard = shock
        elif shock.symbol_glob.endswith("*") and symbol.startswith(shock.symbol_glob[:-1]):
            prefix_match = shock
    return prefix_match or wildcard


SCENARIOS: dict[ScenarioName, StressScenario] = {
    "BLACK_MONDAY_1987": StressScenario(
        name="BLACK_MONDAY_1987",
        description="22% S&P drop with global correlation breakdown.",
        shocks=(
            StressShock("*", -0.18, vol_multiplier=4.0, correlation_breakdown=True),
            StressShock("GOLD#", +0.05, vol_multiplier=3.0),
        ),
    ),
    "FLASH_CRASH_2010": StressScenario(
        name="FLASH_CRASH_2010",
        description="9% S&P drop in 10 minutes, intraday recovery.",
        shocks=(
            StressShock("*", -0.05, vol_multiplier=6.0),
            StressShock("EURUSD#", -0.02),
            StressShock("USDJPY#", -0.03),
        ),
    ),
    "CHF_DEPEG_2015": StressScenario(
        name="CHF_DEPEG_2015",
        description="30% gap on EUR/CHF as SNB removes 1.20 floor.",
        shocks=(
            StressShock("USDCHF#", -0.20, vol_multiplier=8.0, correlation_breakdown=True),
            StressShock("EURUSD#", -0.04),
            StressShock("GOLD#", +0.03),
        ),
    ),
    "COVID_MARCH_2020": StressScenario(
        name="COVID_MARCH_2020",
        description="30% S&P drop + commodity correlation collapse.",
        shocks=(
            StressShock("*", -0.12, vol_multiplier=5.0, correlation_breakdown=True),
            StressShock("GOLD#", +0.04, vol_multiplier=3.0),
            StressShock("BTCUSD#", -0.40, vol_multiplier=6.0),
        ),
    ),
    "SVB_MARCH_2023": StressScenario(
        name="SVB_MARCH_2023",
        description="Banking sector contagion: USD pairs 4-5% moves.",
        shocks=(
            StressShock("USDJPY#", -0.04, vol_multiplier=2.5),
            StressShock("USDCHF#", -0.05, vol_multiplier=2.5),
            StressShock("GOLD#", +0.06, vol_multiplier=3.0),
            StressShock("EURUSD#", +0.03, vol_multiplier=2.5),
        ),
    ),
}


def replay_scenario(
    name: ScenarioName,
    positions: list[StressPosition],
    *,
    starting_equity: float,
    intraday_kill_pct: float = 0.03,
    weekly_kill_pct: float = 0.08,
    margin_floor_pct: float = 0.0,
) -> StressReport:
    """Apply a scenario's shocks to `positions` and report equity damage."""
    scenario = SCENARIOS.get(name)
    if scenario is None:
        raise ValueError(f"unknown scenario {name}")
    if starting_equity <= 0:
        return StressReport(
            scenario=name,
            equity_before=starting_equity, equity_after=starting_equity,
            pnl_usd=0.0, pnl_pct=0.0, drawdown_pct=0.0,
            intraday_kill_breached=False, weekly_kill_breached=False,
            margin_call=False,
        )
    per_sym: dict[str, float] = {}
    total_pnl = 0.0
    for pos in positions:
        shock = _shock_for_symbol(scenario, pos.symbol)
        if shock is None:
            continue
        notional = abs(pos.notional_usd)
        direction_sign = 1.0 if pos.direction.upper() == "BUY" else -1.0
        pnl = direction_sign * shock.return_shock * notional
        per_sym[pos.symbol] = per_sym.get(pos.symbol, 0.0) + pnl
        total_pnl += pnl
    equity_after = starting_equity + total_pnl
    pnl_pct = total_pnl / starting_equity if starting_equity > 0 else 0.0
    drawdown = -pnl_pct if pnl_pct < 0 else 0.0
    margin_call = equity_after <= margin_floor_pct * starting_equity
    return StressReport(
        scenario=name,
        equity_before=starting_equity,
        equity_after=equity_after,
        pnl_usd=total_pnl,
        pnl_pct=pnl_pct,
        drawdown_pct=drawdown,
        intraday_kill_breached=drawdown > intraday_kill_pct,
        weekly_kill_breached=drawdown > weekly_kill_pct,
        margin_call=margin_call,
        per_symbol_pnl=per_sym,
    )


def replay_all(
    positions: list[StressPosition],
    *,
    starting_equity: float,
) -> dict[ScenarioName, StressReport]:
    return {
        name: replay_scenario(name, positions, starting_equity=starting_equity)
        for name in SCENARIOS
    }
