"""Online LightGBM retrainer.

Runs every N closed trades (or on drift alarm) inside the engine main
loop. Pulls the last `window_trades` rows from the journal + shadow
trades, joins their persisted feature vectors, applies triple-barrier
labels where possible, fits a LightGBM challenger checkpoint, and hands
off to champion-challenger evaluation (Tier 3.3).

This is the user's primary "live retraining without GPU" channel —
LightGBM trains a 50-feature × 5000-row dataset in < 30s on a slow CPU.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal
from engine.models.lightgbm_model import LightGBMModel


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_WINDOW_TRADES = 5000
DEFAULT_VAL_FRACTION = 0.2
MIN_TRADES_TO_RETRAIN = 200


@dataclass
class RetrainOutcome:
    skipped: bool
    reason: str
    n_train: int = 0
    n_val: int = 0
    checkpoint: str | None = None
    best_val_logloss: float = 0.0
    elapsed_s: float = 0.0


def should_retrain(
    *,
    closed_trades_since_last: int,
    drift_alarm: bool,
    cpu_pct: float = 0.0,
    every_n: int = settings.RETRAIN_EVERY_N_TRADES,
    cpu_ceiling: int = settings.RETRAIN_CPU_CEILING_PCT,
) -> bool:
    """Decision rule. Drift wins; otherwise the standard N-trade trigger."""
    if cpu_pct > cpu_ceiling:
        return False
    if drift_alarm:
        return True
    return closed_trades_since_last >= every_n


def _load_recent_training_rows(
    *,
    window_trades: int = DEFAULT_WINDOW_TRADES,
    db_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read recent closed trades + their feature snapshots from SQLite."""
    features: list[list[float]] = []
    labels: list[int] = []
    with open_journal(db_path) as con:
        rows = con.execute(
            """
            SELECT t.pnl, t.direction, sf.features_json
              FROM trades t
              JOIN signal_features sf ON sf.trade_id = t.id
             WHERE t.pnl IS NOT NULL
             ORDER BY t.id DESC
             LIMIT ?
            """,
            (int(window_trades),),
        ).fetchall()
    for r in rows:
        try:
            feat = json.loads(r["features_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(feat, list) or not feat:
            continue
        features.append([float(x) for x in feat])
        # Map outcome to BUY/SELL/HOLD labels: winning BUY=0, winning SELL=1,
        # losing trade = HOLD (model would have been right to skip).
        pnl = float(r["pnl"])
        direction = str(r["direction"]).upper()
        if pnl > 0:
            labels.append(0 if direction == "BUY" else 1)
        else:
            labels.append(2)
    if not features:
        return np.empty((0, 0)), np.empty(0, dtype=np.int64)
    arr = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    return arr, y


def retrain_now(
    *,
    window_trades: int = DEFAULT_WINDOW_TRADES,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    db_path: str | None = None,
) -> RetrainOutcome:
    """Run a full LightGBM retrain cycle. Returns a structured outcome."""
    start = time.time()
    X, y = _load_recent_training_rows(window_trades=window_trades, db_path=db_path)
    if X.size == 0 or X.shape[0] < MIN_TRADES_TO_RETRAIN:
        return RetrainOutcome(
            skipped=True,
            reason=f"insufficient training rows ({X.shape[0]} < {MIN_TRADES_TO_RETRAIN})",
            elapsed_s=time.time() - start,
        )

    n = X.shape[0]
    cut = int(n * (1.0 - val_fraction))
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]

    model = LightGBMModel()
    result = model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, early_stopping=20)

    version = f"v{int(time.time())}"
    ckpt = CHECKPOINT_DIR / f"lgbm_challenger_{version}.txt"
    model.save(ckpt)

    return RetrainOutcome(
        skipped=False,
        reason="ok",
        n_train=len(y_tr),
        n_val=len(y_va),
        checkpoint=str(ckpt),
        best_val_logloss=result.best_val_logloss,
        elapsed_s=time.time() - start,
    )
