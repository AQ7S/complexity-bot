"""Process-wide StrategyOrchestrator singleton.

Created on first access. Lives for the lifetime of the engine process.
Holding the orchestrator in a singleton (rather than passing it through
every function signature) lets the IPC command handler and the broadcast
loop share state without circular imports.
"""
from __future__ import annotations

import threading

from engine.strategy.orchestrator import StrategyOrchestrator
from engine.strategy.strategies import all_strategies


_lock = threading.Lock()
_instance: StrategyOrchestrator | None = None


def get_orchestrator() -> StrategyOrchestrator:
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = StrategyOrchestrator(all_strategies())
        return _instance


def reset_for_tests() -> None:
    """Drop the singleton so each test gets a fresh orchestrator."""
    global _instance
    with _lock:
        _instance = None
