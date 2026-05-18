"""Pluggable trading strategies (Tier 3.6).

Each module here defines exactly one Strategy subclass. The orchestrator
imports the public list `ALL_STRATEGIES` to wire them all up at startup.
"""
from __future__ import annotations

from engine.strategy.strategies.scalping import ScalpingStrategy
from engine.strategy.strategies.day_trading import DayTradingStrategy
from engine.strategy.strategies.swing import SwingStrategy
from engine.strategy.strategies.mean_reversion import MeanReversionStrategy
from engine.strategy.strategies.breakout import BreakoutStrategy
from engine.strategy.strategies.carry import CarryStrategy
from engine.strategy.strategies.pairs_trading import PairsTradingStrategy


def all_strategies() -> list:
    return [
        DayTradingStrategy(),
        ScalpingStrategy(),
        SwingStrategy(),
        MeanReversionStrategy(),
        BreakoutStrategy(),
        CarryStrategy(),
        PairsTradingStrategy(),
    ]


__all__ = [
    "all_strategies",
    "ScalpingStrategy", "DayTradingStrategy", "SwingStrategy",
    "MeanReversionStrategy", "BreakoutStrategy", "CarryStrategy",
    "PairsTradingStrategy",
]
