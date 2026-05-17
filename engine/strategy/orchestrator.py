"""Strategy orchestrator (Tier 3.6 / Tier 6.1).

Allocates the daily risk budget across active strategies based on
rolling Sharpe + applies per-strategy circuit breakers (5 consecutive
losses → 4h pause, 3 losing days → 24h shadow-only).

Allocation rule:
    weight_i = max(0, sharpe_i) / Σ_j max(0, sharpe_j)
clipped to a floor (so probationary strategies can still produce
samples) and a ceiling (so one dominant style cannot starve the others).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

from engine.strategy.base import Strategy, StrategyContext, StrategySignal


StrategyMode = Literal["ON", "SHADOW", "OFF"]


MIN_WEIGHT_FLOOR = 0.05
MAX_WEIGHT_CEILING = 0.50
CIRCUIT_BREAKER_LOSS_STREAK = 5
CIRCUIT_BREAKER_PAUSE_S = 4 * 3600   # 4 hours
DAILY_LOSING_STREAK_LIMIT = 3
DAILY_SHADOW_PAUSE_S = 24 * 3600


@dataclass
class StrategyHealth:
    name: str
    consecutive_losses: int = 0
    consecutive_losing_days: int = 0
    paused_until: float = 0.0      # epoch seconds; 0 = not paused
    shadow_only_until: float = 0.0
    rolling_sharpe: float = 0.0
    trades_today: int = 0
    pnl_today_usd: float = 0.0
    last_close_date: str = ""
    operator_mode: StrategyMode = "ON"   # operator override from UI

    def is_paused(self, now: float | None = None) -> bool:
        if self.operator_mode == "OFF":
            return True
        n = now if now is not None else time.time()
        return n < self.paused_until

    def is_shadow_only(self, now: float | None = None) -> bool:
        if self.operator_mode == "SHADOW":
            return True
        n = now if now is not None else time.time()
        return n < self.shadow_only_until

    def current_state(self, now: float | None = None) -> str:
        if self.operator_mode == "OFF":
            return "DISABLED"
        n = now if now is not None else time.time()
        if n < self.paused_until:
            return "PAUSED"
        if self.operator_mode == "SHADOW" or n < self.shadow_only_until:
            return "SHADOW"
        return "ACTIVE"


@dataclass
class OrchestratorTickResult:
    signals: list[StrategySignal] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    skipped_paused: list[str] = field(default_factory=list)
    skipped_shadow: list[str] = field(default_factory=list)


class StrategyOrchestrator:
    def __init__(
        self,
        strategies: Iterable[Strategy],
        *,
        total_risk_pct: float = 0.02,
    ) -> None:
        self.strategies: list[Strategy] = list(strategies)
        self.total_risk_pct = float(total_risk_pct)
        self.health: dict[str, StrategyHealth] = {
            s.name: StrategyHealth(name=s.name) for s in self.strategies
        }

    # ---- Allocation ----
    def allocate_budget(self) -> dict[str, float]:
        active = [s for s in self.strategies if not self.health[s.name].is_paused()]
        if not active:
            return {s.name: 0.0 for s in self.strategies}

        positives = {s.name: max(0.0, self.health[s.name].rolling_sharpe) for s in active}
        total_pos = sum(positives.values())
        if total_pos <= 1e-12:
            equal = 1.0 / len(active)
            weights: dict[str, float] = {s.name: equal for s in active}
        else:
            weights = {n: v / total_pos for n, v in positives.items()}

        # Floor pass: lift below-floor weights; redistribute the deficit
        # away from above-floor weights proportionally.
        deficit = 0.0
        for k in weights:
            if weights[k] < MIN_WEIGHT_FLOOR:
                deficit += MIN_WEIGHT_FLOOR - weights[k]
                weights[k] = MIN_WEIGHT_FLOOR
        if deficit > 0:
            above = {k: weights[k] - MIN_WEIGHT_FLOOR for k in weights if weights[k] > MIN_WEIGHT_FLOOR}
            total_above = sum(above.values())
            if total_above > 0:
                for k in above:
                    weights[k] -= deficit * (above[k] / total_above)

        # Ceiling pass: cap above-ceiling weights; redistribute the
        # excess to below-ceiling weights proportionally to their headroom.
        excess = 0.0
        for k in weights:
            if weights[k] > MAX_WEIGHT_CEILING:
                excess += weights[k] - MAX_WEIGHT_CEILING
                weights[k] = MAX_WEIGHT_CEILING
        if excess > 0:
            below = {k: MAX_WEIGHT_CEILING - weights[k] for k in weights if weights[k] < MAX_WEIGHT_CEILING}
            total_below = sum(below.values())
            if total_below > 0:
                for k in below:
                    share = excess * (below[k] / total_below)
                    weights[k] = min(MAX_WEIGHT_CEILING, weights[k] + share)

        for s in self.strategies:
            if s.name not in weights:
                weights[s.name] = 0.0
        return weights

    # ---- Tick ----
    def tick(self, contexts: list[StrategyContext]) -> OrchestratorTickResult:
        weights = self.allocate_budget()
        out = OrchestratorTickResult(weights=weights)
        for s in self.strategies:
            h = self.health[s.name]
            if h.is_paused():
                out.skipped_paused.append(s.name)
                continue
            if h.is_shadow_only():
                # Strategy still produces signals but main loop must route
                # them to shadow only. The orchestrator marks via empty
                # rationale_tags-prefix; the caller checks `is_shadow_only`.
                out.skipped_shadow.append(s.name)
            for ctx in contexts:
                if not s.accepts_symbol(ctx.symbol):
                    continue
                if ctx.timeframe not in s.timeframes:
                    continue
                try:
                    sig = s.detect(ctx)
                except Exception:  # noqa: BLE001
                    sig = None
                if sig is not None:
                    out.signals.append(sig)
        return out

    # ---- Trade-result feedback ----
    def record_trade_close(
        self,
        strategy_name: str,
        *,
        pnl_usd: float,
        sharpe_update: float | None = None,
        now: float | None = None,
    ) -> None:
        if strategy_name not in self.health:
            return
        h = self.health[strategy_name]
        n = now if now is not None else time.time()
        today = datetime.fromtimestamp(n, tz=timezone.utc).date().isoformat()
        if h.last_close_date != today:
            # Day rollover.
            if h.last_close_date and h.pnl_today_usd < 0:
                h.consecutive_losing_days += 1
                if h.consecutive_losing_days >= DAILY_LOSING_STREAK_LIMIT:
                    h.shadow_only_until = n + DAILY_SHADOW_PAUSE_S
                    h.consecutive_losing_days = 0
            elif h.last_close_date:
                h.consecutive_losing_days = 0
            h.last_close_date = today
            h.trades_today = 0
            h.pnl_today_usd = 0.0
        h.trades_today += 1
        h.pnl_today_usd += pnl_usd
        if pnl_usd <= 0:
            h.consecutive_losses += 1
            if h.consecutive_losses >= CIRCUIT_BREAKER_LOSS_STREAK:
                h.paused_until = n + CIRCUIT_BREAKER_PAUSE_S
                h.consecutive_losses = 0
        else:
            h.consecutive_losses = 0
        if sharpe_update is not None:
            h.rolling_sharpe = float(sharpe_update)

    def reset_pauses(self) -> None:
        for h in self.health.values():
            h.paused_until = 0.0
            h.shadow_only_until = 0.0
            h.consecutive_losses = 0
            h.consecutive_losing_days = 0

    # ---- Operator controls + UI snapshot ----
    def set_mode(self, strategy_name: str, mode: StrategyMode) -> bool:
        """Operator override from the UI: ON, SHADOW, or OFF. Returns True on success."""
        if strategy_name not in self.health:
            return False
        self.health[strategy_name].operator_mode = mode
        if mode == "ON":
            # Clear any latent breaker so the operator's intent takes effect immediately.
            self.health[strategy_name].paused_until = 0.0
            self.health[strategy_name].shadow_only_until = 0.0
        return True

    def snapshot(self, *, now: float | None = None) -> dict:
        """Build the IPC `strategy_status` payload."""
        weights = self.allocate_budget()
        frames: list[dict] = []
        n = now if now is not None else time.time()
        for s in self.strategies:
            h = self.health[s.name]
            style = getattr(s, "style", "")
            frames.append({
                "name": s.name,
                "style": style,
                "state": h.current_state(n),
                "weight": float(weights.get(s.name, 0.0)),
                "rolling_sharpe": float(h.rolling_sharpe),
                "consecutive_losses": int(h.consecutive_losses),
                "trades_today": int(h.trades_today),
                "pnl_today_usd": float(h.pnl_today_usd),
                "paused_until_ts": int(h.paused_until * 1000) if h.paused_until > 0 else 0,
                "shadow_only_until_ts": int(h.shadow_only_until * 1000) if h.shadow_only_until > 0 else 0,
            })
        return {
            "total_risk_pct": float(self.total_risk_pct),
            "strategies": frames,
        }
