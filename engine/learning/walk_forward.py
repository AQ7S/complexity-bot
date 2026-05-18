"""Walk-Forward Parameter Optimization (WFO).

Static strategy parameters (z_threshold, min_confluence, pt_mult, etc.)
guarantee that yesterday's tuned configuration is today's stale one.
Top quant shops re-tune nightly; we don't need that cadence, but going
30 days without re-tuning costs Sharpe.

Algorithm (anchored or rolling):
    for each rolling window W of length L:
        on the in-sample portion of W:
            grid_search over the param grid using purged_kfold CV,
            pick the configuration with the best mean OOS metric
        on the out-of-sample window immediately following W:
            evaluate the picked config and record its OOS metric
    aggregate: the WFO performance is the *concatenation* of all OOS
    windows — a fair forward-looking estimate of how the rolling
    re-tune procedure would perform.

This is fundamentally different from a single purged CV: WFO measures
the strategy + the *tuner*, not just the strategy at one frozen config.

We do not depend on sklearn — the optimizer is a generic grid search
over a user-supplied parameter grid + a user-supplied evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class WFOSplit:
    window_index: int
    in_sample_start: int
    in_sample_end: int
    out_of_sample_start: int
    out_of_sample_end: int


@dataclass(frozen=True)
class WFOResult:
    splits: list[WFOSplit]
    best_params_per_window: list[dict]
    out_of_sample_metrics: list[float]
    mean_oos_metric: float
    std_oos_metric: float


def anchored_walk_forward_splits(
    n_periods: int,
    *,
    is_min: int,
    oos_size: int,
    step: int | None = None,
) -> list[WFOSplit]:
    """Anchored WFO: in-sample window grows; OOS shifts forward each step.

    Anchor at 0, IS = [0, t), OOS = [t, t+oos_size). Step `t` forward by
    `step` (default = oos_size, i.e. non-overlapping OOS windows).
    """
    if n_periods < is_min + oos_size:
        return []
    step = step or oos_size
    splits: list[WFOSplit] = []
    t = is_min
    idx = 0
    while t + oos_size <= n_periods:
        splits.append(WFOSplit(
            window_index=idx,
            in_sample_start=0, in_sample_end=t,
            out_of_sample_start=t, out_of_sample_end=t + oos_size,
        ))
        idx += 1
        t += step
    return splits


def rolling_walk_forward_splits(
    n_periods: int,
    *,
    is_size: int,
    oos_size: int,
    step: int | None = None,
) -> list[WFOSplit]:
    """Rolling WFO: fixed-length sliding IS window; OOS immediately follows."""
    if n_periods < is_size + oos_size:
        return []
    step = step or oos_size
    splits: list[WFOSplit] = []
    t = is_size
    idx = 0
    while t + oos_size <= n_periods:
        splits.append(WFOSplit(
            window_index=idx,
            in_sample_start=t - is_size, in_sample_end=t,
            out_of_sample_start=t, out_of_sample_end=t + oos_size,
        ))
        idx += 1
        t += step
    return splits


def grid_search(
    param_grid: dict[str, Iterable],
    *,
    is_returns_fn: Callable[[dict], np.ndarray],
    metric: Callable[[np.ndarray], float] | None = None,
) -> tuple[dict, float]:
    """Exhaustive grid search. `is_returns_fn(params) -> per-period returns
    on the IS window`; we maximize `metric(returns)`. Default metric is
    Sharpe ratio.
    """
    if metric is None:
        def _sharpe(x: np.ndarray) -> float:
            if x.size < 2:
                return 0.0
            sd = float(np.std(x, ddof=1))
            return 0.0 if sd < 1e-12 else float(np.mean(x)) / sd
        metric = _sharpe
    keys = list(param_grid.keys())
    grids = [list(param_grid[k]) for k in keys]
    best_score = -float("inf")
    best_params: dict = {}
    for combo in product(*grids):
        params = dict(zip(keys, combo))
        returns = is_returns_fn(params)
        score = float(metric(returns))
        if score > best_score:
            best_score = score
            best_params = dict(params)
    return best_params, best_score


def walk_forward_optimize(
    splits: list[WFOSplit],
    param_grid: dict[str, Iterable],
    *,
    returns_for_window_fn: Callable[[dict, int, int], np.ndarray],
    metric: Callable[[np.ndarray], float] | None = None,
) -> WFOResult:
    """End-to-end walk-forward optimization.

    `returns_for_window_fn(params, start, end) -> ndarray of per-period
    returns the strategy produced on [start, end) with the given params`.
    This indirection keeps the optimizer agnostic to the strategy.
    """
    if not splits:
        return WFOResult(splits=[], best_params_per_window=[],
                         out_of_sample_metrics=[],
                         mean_oos_metric=0.0, std_oos_metric=0.0)
    if metric is None:
        def _sharpe(x: np.ndarray) -> float:
            if x.size < 2:
                return 0.0
            sd = float(np.std(x, ddof=1))
            return 0.0 if sd < 1e-12 else float(np.mean(x)) / sd
        metric = _sharpe
    best_params_list: list[dict] = []
    oos_metrics: list[float] = []
    for sp in splits:
        best_params, _ = grid_search(
            param_grid,
            is_returns_fn=lambda p: returns_for_window_fn(
                p, sp.in_sample_start, sp.in_sample_end,
            ),
            metric=metric,
        )
        oos_returns = returns_for_window_fn(
            best_params, sp.out_of_sample_start, sp.out_of_sample_end,
        )
        oos_metrics.append(float(metric(oos_returns)))
        best_params_list.append(best_params)
    arr = np.array(oos_metrics, dtype=np.float64)
    return WFOResult(
        splits=splits,
        best_params_per_window=best_params_list,
        out_of_sample_metrics=oos_metrics,
        mean_oos_metric=float(arr.mean()),
        std_oos_metric=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    )
