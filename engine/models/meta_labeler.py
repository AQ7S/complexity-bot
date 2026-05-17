"""Meta-labeling (López de Prado, AFML ch. 3).

After the primary model M1 emits a directional prediction (BUY or SELL), a
secondary binary model M2 decides whether to *take* the trade. M2's training
signal is "did the trade actually win", conditioned on M1's prediction.

Effect: M1 picks direction; M2 raises the precision of executed signals at
the cost of fewer trades. Net result is typically lower turnover, higher
win rate, and tunable precision/recall trade-off.

We implement a minimal, dependency-free logistic-regression meta-labeler
(numpy + manual gradient descent) so this module is testable without sklearn.
A drop-in replacement using sklearn or LightGBM can subclass `MetaLabeler`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


@dataclass
class MetaLabelerModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        x = (features - self.feature_mean) / self.feature_std
        z = x @ self.weights + self.bias
        return _sigmoid(z)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int64)


def train_meta_labeler(
    features: np.ndarray,
    outcomes: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 0.05,
    l2: float = 1e-3,
    threshold: float = 0.5,
) -> MetaLabelerModel:
    """Fit a logistic-regression M2 on (features, win_flag).

    `features`  shape (N, F) — typically [M1_confidence] concatenated with the
                same feature vector the primary model saw.
    `outcomes`  shape (N,)   — 1 if the M1-predicted trade actually won, else 0.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and outcomes length mismatch")

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xn = (X - mu) / sd
    n, d = Xn.shape

    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        z = Xn @ w + b
        p = _sigmoid(z)
        grad_w = (Xn.T @ (p - y)) / n + l2 * w
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return MetaLabelerModel(
        weights=w, bias=b,
        feature_mean=mu, feature_std=sd,
        threshold=threshold,
    )


def apply_meta_label(
    model: MetaLabelerModel,
    primary_pred: int,
    features: np.ndarray,
) -> tuple[int, float]:
    """Return (final_pred, take_probability).

    If `primary_pred` is HOLD (2), pass through. Otherwise consult M2: if
    take-probability >= threshold, keep primary_pred; else return HOLD.
    """
    if primary_pred == 2:
        return 2, 0.0
    proba = float(model.predict_proba(features.reshape(1, -1))[0])
    if proba >= model.threshold:
        return primary_pred, proba
    return 2, proba
