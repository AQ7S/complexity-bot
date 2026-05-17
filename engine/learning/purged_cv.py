"""Purged + embargoed walk-forward k-fold CV (López de Prado, AFML ch. 7).

Standard k-fold leaks information across folds when financial features are
autocorrelated. Two corrections:

  * Purge — drop any training sample whose label horizon overlaps the test
    fold's time range. (Removes look-ahead bias from overlapping windows.)
  * Embargo — drop any training sample within `embargo_pct` of the test fold
    boundary. (Removes serial-correlation leakage from neighborhood bars.)

The output is (train_idx, test_idx) for each fold; callers wire those into
their training loop. The CV is *the only* validation method that survives
peer review in quantitative finance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    score: float
    extra: dict


def purged_kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    label_horizons: np.ndarray | None = None,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for each of `n_splits` folds.

    `label_horizons[i]` is the index of the last bar whose information feeds
    into sample `i`'s label. If None, defaults to `i` (no overlap).

    `embargo_pct` is a fraction of `n_samples`: training samples within that
    many bars of either test fold boundary are dropped.
    """
    if n_samples < n_splits:
        raise ValueError(f"n_samples ({n_samples}) < n_splits ({n_splits})")
    if not 0 <= embargo_pct < 0.5:
        raise ValueError("embargo_pct must be in [0, 0.5)")
    if label_horizons is None:
        label_horizons = np.arange(n_samples, dtype=np.int64)
    else:
        label_horizons = np.asarray(label_horizons, dtype=np.int64)
        if len(label_horizons) != n_samples:
            raise ValueError("label_horizons length must equal n_samples")

    indices = np.arange(n_samples, dtype=np.int64)
    fold_bounds = np.array_split(indices, n_splits)
    embargo = int(round(embargo_pct * n_samples))

    for k, test_idx in enumerate(fold_bounds):
        if len(test_idx) == 0:
            continue
        test_start = int(test_idx[0])
        test_end = int(test_idx[-1])
        # Purge: training samples whose label horizon falls inside the test
        # range are dropped.
        purged_mask = (label_horizons >= test_start) & (label_horizons <= test_end)
        # Embargo: drop training samples within `embargo` of test boundaries.
        emb_lo = max(0, test_start - embargo)
        emb_hi = min(n_samples - 1, test_end + embargo)
        embargo_mask = (indices >= emb_lo) & (indices <= emb_hi)
        # Exclude all in-test bars too.
        in_test = (indices >= test_start) & (indices <= test_end)
        drop = purged_mask | embargo_mask | in_test
        train_idx = indices[~drop]
        yield k, train_idx, test_idx


def walk_forward_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_factory: Callable[[], object],
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    label_horizons: np.ndarray | None = None,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> list[FoldResult]:
    """Walk-forward evaluator. `model_factory()` must return an object with
    `.fit(X, y)` and `.predict(X)` (sklearn-compatible).

    Default scorer = accuracy.
    """
    if scorer is None:
        scorer = lambda yt, yp: float(np.mean(yt == yp))  # noqa: E731
    results: list[FoldResult] = []
    for k, train_idx, test_idx in purged_kfold_indices(
        n_samples=len(X), n_splits=n_splits,
        label_horizons=label_horizons, embargo_pct=embargo_pct,
    ):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])  # type: ignore[attr-defined]
        preds = model.predict(X[test_idx])  # type: ignore[attr-defined]
        score = scorer(y[test_idx], preds)
        results.append(FoldResult(
            fold=k, n_train=len(train_idx), n_test=len(test_idx),
            score=score, extra={},
        ))
    return results


def aggregate_folds(results: list[FoldResult]) -> dict[str, float]:
    """Return mean, std, min, max over fold scores."""
    if not results:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_folds": 0}
    scores = np.array([r.score for r in results], dtype=np.float64)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "n_folds": len(scores),
    }
