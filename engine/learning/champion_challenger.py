"""Champion–Challenger promotion with statistical significance gate.

A new retrain does NOT auto-promote. It runs in parallel ("challenger")
with the currently deployed model ("champion") against the same signal
stream. After enough paired observations, a one-sided paired bootstrap
test decides whether the challenger is *statistically* better — not just
nominally better on the sample.

Promotion criteria (all must hold):

  1. challenger.sharpe ≥ champion.sharpe × `min_sharpe_uplift` (default 1.10)
  2. paired-bootstrap p-value (challenger > champion) < `p_threshold` (default 0.05)
  3. challenger.win_rate ≥ `min_win_rate` (default 0.50)
  4. paired sample size ≥ `n_min` (default 100)

These criteria are conservative — most contenders fail. That's the point:
the gate exists to *prevent* premature promotion of models that look good
on the most recent sample but won't generalise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedSignal:
    """Per-signal paired outcome between champion and challenger."""
    champion_return: float
    challenger_return: float


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    n_paired: int
    champion_sharpe: float
    challenger_sharpe: float
    sharpe_ratio: float
    p_value: float
    challenger_win_rate: float


def _sharpe(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(returns)) / sd * float(np.sqrt(periods_per_year))


def paired_bootstrap_pvalue(
    diffs: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int | None = 7,
) -> float:
    """One-sided test: P(mean(diff) ≤ 0 | data) under bootstrap resampling."""
    if diffs.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diffs[idx]))
    return float(np.mean(means <= 0))


def evaluate_promotion(
    pairs: list[PairedSignal],
    *,
    n_min: int = 100,
    p_threshold: float = 0.05,
    min_sharpe_uplift: float = 1.10,
    min_win_rate: float = 0.50,
) -> PromotionDecision:
    """Decide whether the challenger should replace the champion."""
    if len(pairs) < n_min:
        return PromotionDecision(
            promote=False,
            reason=f"insufficient samples: {len(pairs)} < {n_min}",
            n_paired=len(pairs),
            champion_sharpe=0.0, challenger_sharpe=0.0,
            sharpe_ratio=0.0, p_value=1.0, challenger_win_rate=0.0,
        )
    champ = np.array([p.champion_return for p in pairs], dtype=np.float64)
    chall = np.array([p.challenger_return for p in pairs], dtype=np.float64)
    diffs = chall - champ

    s_champ = _sharpe(champ)
    s_chall = _sharpe(chall)
    ratio = s_chall / s_champ if abs(s_champ) > 1e-9 else float("inf") if s_chall > 0 else 0.0
    p_val = paired_bootstrap_pvalue(diffs)
    wr = float(np.mean(chall > 0))

    if s_champ <= 0:
        if s_chall > 0 and p_val < p_threshold and wr >= min_win_rate:
            return PromotionDecision(
                promote=True,
                reason="champion underwater; challenger profitable + significant",
                n_paired=len(pairs), champion_sharpe=s_champ,
                challenger_sharpe=s_chall, sharpe_ratio=ratio,
                p_value=p_val, challenger_win_rate=wr,
            )
        return PromotionDecision(
            promote=False,
            reason=f"champion underwater but challenger not significantly better (p={p_val:.3f})",
            n_paired=len(pairs), champion_sharpe=s_champ,
            challenger_sharpe=s_chall, sharpe_ratio=ratio,
            p_value=p_val, challenger_win_rate=wr,
        )

    if ratio < min_sharpe_uplift:
        return PromotionDecision(
            promote=False,
            reason=f"sharpe uplift {ratio:.2f} < required {min_sharpe_uplift:.2f}",
            n_paired=len(pairs), champion_sharpe=s_champ,
            challenger_sharpe=s_chall, sharpe_ratio=ratio,
            p_value=p_val, challenger_win_rate=wr,
        )
    if p_val >= p_threshold:
        return PromotionDecision(
            promote=False,
            reason=f"paired bootstrap p={p_val:.3f} ≥ {p_threshold}",
            n_paired=len(pairs), champion_sharpe=s_champ,
            challenger_sharpe=s_chall, sharpe_ratio=ratio,
            p_value=p_val, challenger_win_rate=wr,
        )
    if wr < min_win_rate:
        return PromotionDecision(
            promote=False,
            reason=f"challenger win rate {wr:.2%} < required {min_win_rate:.2%}",
            n_paired=len(pairs), champion_sharpe=s_champ,
            challenger_sharpe=s_chall, sharpe_ratio=ratio,
            p_value=p_val, challenger_win_rate=wr,
        )

    return PromotionDecision(
        promote=True,
        reason="all gates passed",
        n_paired=len(pairs), champion_sharpe=s_champ,
        challenger_sharpe=s_chall, sharpe_ratio=ratio,
        p_value=p_val, challenger_win_rate=wr,
    )
