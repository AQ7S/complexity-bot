"""SHAP-lite trade attribution.

After every closed trade with the LightGBM model: compute SHAP values
for the features that drove the prediction. Aggregating across many
closed trades reveals which features are responsible for losses — the
final feedback loop closing the pipeline:

    feature → prediction → trade → outcome → SHAP → "this feature
    causes most losses" → operator prunes or transforms

We do not require the optional `shap` package — LightGBM has built-in
TreeSHAP support via `predict(..., pred_contrib=True)`. The output is
shape (n_samples, n_features+1); the last column is the model's base
value (the bias).

Aggregation helpers:
  * `mean_abs_attribution()`  — average |SHAP| per feature across trades
                                (= feature importance by attribution)
  * `attribution_for_losers()` — same but only for trades where pnl < 0
  * `top_loss_drivers()`       — sorted descending by mean |SHAP| on losers,
                                 returning (feature_index, magnitude)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SHAPRow:
    trade_id: int
    pnl: float
    base_value: float
    contributions: np.ndarray  # shape (n_features,)


def compute_shap_for_trades(
    model,
    feature_matrix: np.ndarray,
    *,
    trade_ids: list[int] | None = None,
    pnls: list[float] | None = None,
) -> list[SHAPRow]:
    """Compute TreeSHAP contributions for each row in `feature_matrix`.

    `model` must expose `.booster` (a LightGBM Booster) or implement
    `predict(X, pred_contrib=True)` itself. Returns one SHAPRow per row.
    """
    if feature_matrix.size == 0:
        return []
    booster = getattr(model, "booster", model)
    raw = booster.predict(feature_matrix, pred_contrib=True)
    # LightGBM returns either (n, n_features+1) for binary or
    # (n, n_classes * (n_features+1)) for multiclass. We sum across
    # classes to get a single per-feature attribution (signed = which
    # class label was tipped).
    arr = np.asarray(raw, dtype=np.float64)
    n = feature_matrix.shape[0]
    if arr.ndim == 1:
        arr = arr.reshape(n, -1)
    n_features = feature_matrix.shape[1]
    cols_per_class = n_features + 1
    if arr.shape[1] % cols_per_class != 0:
        # Fall back: treat as a single class.
        per_class = [arr]
    else:
        n_classes = arr.shape[1] // cols_per_class
        per_class = [
            arr[:, k * cols_per_class:(k + 1) * cols_per_class]
            for k in range(n_classes)
        ]
    summed = np.zeros((n, cols_per_class), dtype=np.float64)
    for p in per_class:
        summed += p
    summed /= float(len(per_class))

    rows: list[SHAPRow] = []
    for i in range(n):
        rows.append(SHAPRow(
            trade_id=int(trade_ids[i]) if trade_ids and i < len(trade_ids) else i,
            pnl=float(pnls[i]) if pnls and i < len(pnls) else 0.0,
            base_value=float(summed[i, -1]),
            contributions=summed[i, :-1].copy(),
        ))
    return rows


def mean_abs_attribution(rows: list[SHAPRow]) -> np.ndarray:
    """Return per-feature mean |SHAP|. Shape (n_features,)."""
    if not rows:
        return np.array([])
    stacked = np.vstack([np.abs(r.contributions) for r in rows])
    return stacked.mean(axis=0)


def attribution_for_losers(rows: list[SHAPRow]) -> np.ndarray:
    losers = [r for r in rows if r.pnl < 0]
    return mean_abs_attribution(losers)


def top_loss_drivers(
    rows: list[SHAPRow],
    *,
    top_n: int = 5,
    feature_names: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Return the top `top_n` features driving losses, sorted descending."""
    mean = attribution_for_losers(rows)
    if mean.size == 0:
        return []
    order = np.argsort(mean)[::-1][:top_n]
    out: list[tuple[str, float]] = []
    for i in order:
        name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
        out.append((name, float(mean[i])))
    return out


def consistency_check(rows: list[SHAPRow], predictions: np.ndarray) -> float:
    """SHAP completeness: contributions + base_value should equal the prediction.

    Returns the mean absolute error across rows. Should be near zero on a
    healthy LightGBM model. Useful as a smoke test in CI.
    """
    if not rows or len(rows) != len(predictions):
        return float("inf")
    errs = []
    for r, p in zip(rows, predictions):
        reconstructed = float(r.contributions.sum() + r.base_value)
        errs.append(abs(reconstructed - float(p)))
    return float(np.mean(errs))
