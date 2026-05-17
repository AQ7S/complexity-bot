"""Auto-retrain dispatcher (Tier 5).

Brings together the four moving parts of the live retrain loop:

  1. Drift detection (Page-Hinkley + ADWIN on rolling Sharpe).
  2. N-trade trigger (every RETRAIN_EVERY_N_TRADES closed trades).
  3. CPU ceiling — skip when CPU is hot.
  4. LightGBM retrain → champion-challenger gate → conditional promotion.

The dispatcher exposes a single `tick()` method that the engine main
loop calls once per closed trade (or once per minute). On every tick
it updates the drift detector with the most recent rolling-Sharpe
sample and decides whether to spawn a retrain.

The actual retrain itself runs on a background thread/process so the
trading loop is never blocked. We hold a single `_running` flag so
overlapping retrains can't race.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from engine.config import settings
from engine.learning.champion_challenger import (
    PairedSignal,
    PromotionDecision,
    evaluate_promotion,
)
from engine.learning.drift_detector import ADWINDetector, PageHinkleyDetector
from engine.models.online_lgbm_trainer import (
    RetrainOutcome,
    retrain_now,
    should_retrain,
)


@dataclass
class DispatcherState:
    trades_since_last: int = 0
    last_retrain_ts: float = 0.0
    last_outcome: RetrainOutcome | None = None
    last_promotion: PromotionDecision | None = None
    is_running: bool = False
    drift_events: int = 0


@dataclass
class RetrainDispatcher:
    """Orchestrates the auto-retrain decision and execution."""

    n_trade_trigger: int = settings.RETRAIN_EVERY_N_TRADES
    cpu_ceiling_pct: int = settings.RETRAIN_CPU_CEILING_PCT
    drift_threshold: float = 5.0
    cooldown_s: float = 300.0    # min seconds between retrains

    state: DispatcherState = field(default_factory=DispatcherState)
    _ph: PageHinkleyDetector = field(default_factory=lambda: PageHinkleyDetector(threshold=5.0, min_delta=0.005))
    _adwin: ADWINDetector = field(default_factory=lambda: ADWINDetector(delta=0.002))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_sharpe_sample(self, sharpe: float) -> None:
        """Feed the rolling-Sharpe stream to both drift detectors."""
        self._ph.add(float(sharpe))
        self._adwin.add(float(sharpe))

    def drift_alarm(self) -> bool:
        return self._ph.drift_detected() or self._adwin.drift_detected()

    def acknowledge_drift(self) -> None:
        if self._ph.drift_detected():
            self._ph.reset()
        if self._adwin.drift_detected():
            self._adwin.acknowledge()

    def record_closed_trade(self) -> None:
        self.state.trades_since_last += 1

    def _spawn_retrain(
        self,
        *,
        cpu_pct: float,
        paired_signals_provider: Callable[[], list[PairedSignal]] | None,
        promotion_callback: Callable[[RetrainOutcome, PromotionDecision], None] | None,
        retrain_fn: Callable[..., RetrainOutcome],
    ) -> None:
        def _job() -> None:
            try:
                outcome = retrain_fn()
                self.state.last_outcome = outcome
                logger.info(
                    "retrain_dispatcher: retrain finished — checkpoint={} loss={:.4f} skipped={}",
                    outcome.checkpoint, outcome.best_val_logloss, outcome.skipped,
                )
                if outcome.skipped or not outcome.checkpoint:
                    return
                if paired_signals_provider is not None:
                    paired = paired_signals_provider()
                    decision = evaluate_promotion(paired)
                    self.state.last_promotion = decision
                    logger.info("retrain_dispatcher: champion-challenger — {}", decision.reason)
                    if promotion_callback is not None:
                        try:
                            promotion_callback(outcome, decision)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("promotion_callback raised: {}", e)
            finally:
                self.state.is_running = False
                self.state.last_retrain_ts = time.time()
                self.state.trades_since_last = 0
                self.acknowledge_drift()

        t = threading.Thread(target=_job, name="retrain", daemon=True)
        t.start()

    def tick(
        self,
        *,
        cpu_pct: float = 0.0,
        paired_signals_provider: Callable[[], list[PairedSignal]] | None = None,
        promotion_callback: Callable[[RetrainOutcome, PromotionDecision], None] | None = None,
        retrain_fn: Callable[..., RetrainOutcome] | None = None,
    ) -> bool:
        """Returns True iff a retrain was spawned during this tick."""
        with self._lock:
            if self.state.is_running:
                return False
            if (time.time() - self.state.last_retrain_ts) < self.cooldown_s:
                return False
            alarm = self.drift_alarm()
            if alarm:
                self.state.drift_events += 1
            decide = should_retrain(
                closed_trades_since_last=self.state.trades_since_last,
                drift_alarm=alarm,
                cpu_pct=cpu_pct,
                every_n=self.n_trade_trigger,
                cpu_ceiling=self.cpu_ceiling_pct,
            )
            if not decide:
                return False
            self.state.is_running = True

        # Outside the lock to keep the trading loop responsive.
        self._spawn_retrain(
            cpu_pct=cpu_pct,
            paired_signals_provider=paired_signals_provider,
            promotion_callback=promotion_callback,
            retrain_fn=retrain_fn or retrain_now,
        )
        return True
