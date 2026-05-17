"""LightGBM gradient-boosted tree model — CPU-friendly primary alternative.

The deep CNN-LSTM is great for representation learning but needs a GPU
for retraining at a useful cadence. LightGBM (Ke et al. 2017) is a
gradient-boosted decision tree library that:

  * Trains on CPU in seconds for typical financial feature counts (50–200
    features, 50k–500k rows).
  * Frequently beats deep nets on tabular data per the M4/M5 competitions.
  * Outputs calibrated probabilities natively.
  * Is small enough to retrain *continuously* without leaving the engine
    machine — the answer to the operator's "no Colab, no GPU, retrain
    live" constraint.

This module wraps LightGBM with a uniform `.train()`, `.predict_proba()`,
`.save()`, `.load()` API matching the CNN-LSTM contract enough that the
consensus engine can swap them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = ("BUY", "SELL", "HOLD")


@dataclass
class LGBMTrainingResult:
    best_iteration: int
    best_val_logloss: float
    feature_importances: dict[int, float]


class LightGBMModel:
    """Sklearn-style 3-class classifier backed by lightgbm.LGBMClassifier."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "n_estimators": 300,
        }
        if params:
            self.params.update(params)
        self.booster = None
        self.feature_names: list[str] | None = None
        self.version: str | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping: int = 20,
    ) -> LGBMTrainingResult:
        import lightgbm as lgb  # noqa: PLC0415
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("val")
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        if X_val is not None and y_val is not None and early_stopping > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping, verbose=False))
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        n_round = int(self.params.get("n_estimators", 300))
        self.booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=n_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        best_iter = self.booster.best_iteration or n_round
        best_logloss = 0.0
        if "val" in self.booster.best_score:
            best_logloss = float(list(self.booster.best_score["val"].values())[0])
        elif "train" in self.booster.best_score:
            best_logloss = float(list(self.booster.best_score["train"].values())[0])
        importances = self.booster.feature_importance(importance_type="gain")
        return LGBMTrainingResult(
            best_iteration=int(best_iter),
            best_val_logloss=best_logloss,
            feature_importances={i: float(v) for i, v in enumerate(importances)},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model not trained yet")
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("model not trained yet")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p))

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMModel":
        import lightgbm as lgb  # noqa: PLC0415
        m = cls()
        m.booster = lgb.Booster(model_file=str(path))
        return m
