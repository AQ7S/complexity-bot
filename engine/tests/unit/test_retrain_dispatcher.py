"""Tests for the auto-retrain dispatcher (Tier 5.2)."""
from __future__ import annotations

import time

from engine.learning.champion_challenger import PairedSignal, PromotionDecision
from engine.learning.retrain_dispatcher import RetrainDispatcher
from engine.models.online_lgbm_trainer import RetrainOutcome


def _stub_retrain(**kwargs) -> RetrainOutcome:
    return RetrainOutcome(
        skipped=False, reason="ok",
        n_train=400, n_val=100,
        checkpoint="/tmp/fake_lgbm.txt",
        best_val_logloss=0.85, elapsed_s=12.0,
    )


def _stub_skipped(**kwargs) -> RetrainOutcome:
    return RetrainOutcome(skipped=True, reason="not enough rows", elapsed_s=0.1)


def test_no_retrain_below_threshold():
    d = RetrainDispatcher(n_trade_trigger=100, cooldown_s=0.0)
    for _ in range(50):
        d.record_closed_trade()
    fired = d.tick(retrain_fn=_stub_retrain)
    assert not fired


def test_retrain_fires_at_threshold():
    d = RetrainDispatcher(n_trade_trigger=10, cooldown_s=0.0)
    for _ in range(10):
        d.record_closed_trade()
    fired = d.tick(retrain_fn=_stub_retrain)
    assert fired
    # Wait briefly for the background thread to finish.
    for _ in range(20):
        if not d.state.is_running:
            break
        time.sleep(0.05)
    assert d.state.last_outcome is not None
    assert d.state.last_outcome.checkpoint == "/tmp/fake_lgbm.txt"
    assert d.state.trades_since_last == 0


def test_drift_triggers_early_retrain():
    d = RetrainDispatcher(n_trade_trigger=100, cooldown_s=0.0, drift_threshold=1.0)
    # Feed a sharpe shock that the PH detector will catch.
    for _ in range(50):
        d.add_sharpe_sample(0.0)
    for _ in range(30):
        d.add_sharpe_sample(-2.0)
    assert d.drift_alarm()
    fired = d.tick(retrain_fn=_stub_retrain)
    assert fired


def test_cpu_ceiling_blocks_retrain():
    d = RetrainDispatcher(n_trade_trigger=5, cooldown_s=0.0, cpu_ceiling_pct=10)
    for _ in range(10):
        d.record_closed_trade()
    fired = d.tick(retrain_fn=_stub_retrain, cpu_pct=90.0)
    assert not fired


def test_cooldown_blocks_back_to_back():
    d = RetrainDispatcher(n_trade_trigger=5, cooldown_s=600.0)
    for _ in range(10):
        d.record_closed_trade()
    fired1 = d.tick(retrain_fn=_stub_retrain)
    assert fired1
    for _ in range(20):
        if not d.state.is_running:
            break
        time.sleep(0.05)
    for _ in range(10):
        d.record_closed_trade()
    fired2 = d.tick(retrain_fn=_stub_retrain)
    assert not fired2  # cooldown still in force


def test_promotion_callback_called_on_success():
    d = RetrainDispatcher(n_trade_trigger=5, cooldown_s=0.0)
    for _ in range(5):
        d.record_closed_trade()

    pairs = [PairedSignal(0.0, 0.01) for _ in range(150)]
    captured: list[PromotionDecision] = []

    def cb(outcome, decision):
        captured.append(decision)

    d.tick(
        retrain_fn=_stub_retrain,
        paired_signals_provider=lambda: pairs,
        promotion_callback=cb,
    )
    for _ in range(20):
        if not d.state.is_running:
            break
        time.sleep(0.05)
    assert len(captured) == 1
    assert isinstance(captured[0], PromotionDecision)


def test_skipped_retrain_does_not_promote():
    d = RetrainDispatcher(n_trade_trigger=5, cooldown_s=0.0)
    for _ in range(5):
        d.record_closed_trade()
    pairs = [PairedSignal(0.0, 0.01) for _ in range(150)]
    captured: list[PromotionDecision] = []
    d.tick(
        retrain_fn=_stub_skipped,
        paired_signals_provider=lambda: pairs,
        promotion_callback=lambda o, dec: captured.append(dec),
    )
    for _ in range(20):
        if not d.state.is_running:
            break
        time.sleep(0.05)
    assert captured == []
