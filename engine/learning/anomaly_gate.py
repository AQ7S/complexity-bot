"""Distribution anomaly gate (Isolation Forest fallback to Mahalanobis).

The #1 cause of catastrophic loss in deployed ML trading: the live
feature vector wanders far outside the manifold the model was trained
on, but the model still confidently predicts BUY/SELL. Anomaly
detection on the input vector is a cheap, effective safety net:

    if anomaly_score(features_now) > threshold:
        reject signal — out of distribution

We use Isolation Forest (sklearn) when available; when not, we fall
back to a Mahalanobis-distance gate (Σ⁻¹-weighted distance from the
training-set mean). Either way the gate is hot-swappable.

The gate is fit on the saved `signal_features` rows from the journal
during the same training cadence as the LightGBM retrain — so it's
always in sync with the model's actual training distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyGate:
    """Abstract gate interface. Concrete impls below."""
    mean: np.ndarray
    cov_inv: np.ndarray
    threshold: float
    method: str = "mahalanobis"
    sk_model: object | None = None

    def score(self, feature_vector: np.ndarray) -> float:
        """Higher score = more anomalous."""
        if self.sk_model is not None:
            # sklearn returns higher = MORE normal; invert.
            x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
            return float(-self.sk_model.score_samples(x)[0])  # type: ignore[attr-defined]
        diff = np.asarray(feature_vector, dtype=np.float64) - self.mean
        d = float(diff @ self.cov_inv @ diff)
        return math.sqrt(max(d, 0.0))

    def is_anomalous(self, feature_vector: np.ndarray) -> bool:
        return self.score(feature_vector) > self.threshold


def fit_mahalanobis_gate(
    training_features: np.ndarray,
    *,
    quantile_threshold: float = 0.99,
    ridge: float = 1e-6,
) -> AnomalyGate:
    """Fit a Mahalanobis-distance gate on `training_features` (n_rows × n_dims).

    The threshold is set at the `quantile_threshold` of in-sample
    distances — so on the training data, only ~1% of samples are flagged.
    """
    X = np.asarray(training_features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 50:
        return AnomalyGate(
            mean=np.zeros(X.shape[1] if X.ndim == 2 else 1),
            cov_inv=np.eye(X.shape[1] if X.ndim == 2 else 1),
            threshold=float("inf"),
            method="mahalanobis_insufficient_data",
        )
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * ridge
    cov_inv = np.linalg.pinv(cov)
    diffs = X - mean
    dists = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))
    threshold = float(np.quantile(dists, quantile_threshold))
    return AnomalyGate(
        mean=mean, cov_inv=cov_inv, threshold=threshold,
        method="mahalanobis",
    )


def fit_isolation_forest_gate(
    training_features: np.ndarray,
    *,
    contamination: float = 0.01,
    random_state: int = 17,
) -> AnomalyGate:
    """Fit an Isolation Forest gate using sklearn.

    Falls back to Mahalanobis when sklearn is unavailable.
    """
    X = np.asarray(training_features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 50:
        return fit_mahalanobis_gate(X)
    try:
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415
    except ImportError:
        return fit_mahalanobis_gate(X)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    model.fit(X)
    # Use the contamination point on the training score distribution.
    scores = -model.score_samples(X)
    threshold = float(np.quantile(scores, 1.0 - contamination))
    gate = AnomalyGate(
        mean=X.mean(axis=0),
        cov_inv=np.eye(X.shape[1]),
        threshold=threshold,
        method="isolation_forest",
        sk_model=model,
    )
    return gate
