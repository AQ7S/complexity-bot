"""Probability of Backtest Overfitting (PBO), Deflated Sharpe Ratio,
and Combinatorially Symmetric Cross-Validation (CSCV) — López de Prado,
"Advances in Financial Machine Learning" (AFML) ch. 11 + "Pseudo-
Mathematics and Financial Charlatanism" (Bailey et al. 2014).

Why these exist:
    The instant we try N strategy variants and pick the "best" one, its
    apparent performance is biased upward. The more we tried, the bigger
    the bias. These three diagnostics quantify and correct for it:

      * CSCV       — splits the trial × period return matrix into balanced
                     in-sample / out-of-sample partitions, ranks strategies
                     in IS, compares ranks in OOS. Output is the empirical
                     rank-flip distribution.
      * PBO        — fraction of CSCV partitions where the best IS strategy
                     under-performed in OOS. PBO > 0.5 means the selection
                     procedure is *worse than random*.
      * Deflated   — adjusts an observed Sharpe ratio for (a) number of
        Sharpe     trials attempted, (b) skewness/kurtosis of returns,
                     (c) sample length. Produces a probability that the
                     true Sharpe exceeds zero.

Both are now gating requirements for any model/strategy promotion via
Tier 3.3 champion-challenger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_trials: int
    n_partitions: int
    logits: list[float]
    best_is_strategy_indices: list[int]


@dataclass(frozen=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    n_trials: int
    n_observations: int


def _phi_cdf(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Beasley-Springer-Moro inverse-normal."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -_phi_inv(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3
    return float(t - num / den)


def _sharpe(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(x)) / sd


def cscv_partitions(n_periods: int, s: int = 16) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate CSCV (in-sample, out-of-sample) period index pairs.

    Splits `n_periods` into `s` equal blocks (truncating remainder), then
    enumerates every C(s, s/2) way to take half the blocks as IS and the
    complement as OOS. `s` must be even; `s=16` gives 12,870 partitions
    (a strong-enough sample for the PBO estimator).
    """
    if s % 2 != 0:
        raise ValueError("s must be even")
    if n_periods < s:
        raise ValueError(f"n_periods ({n_periods}) < s ({s})")
    block_size = n_periods // s
    blocks = [np.arange(i * block_size, (i + 1) * block_size, dtype=np.int64)
              for i in range(s)]
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    half = s // 2
    for combo in combinations(range(s), half):
        is_set = set(combo)
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(s) if i not in is_set])
        partitions.append((is_idx, oos_idx))
    return partitions


def compute_pbo(
    returns_matrix: np.ndarray,
    *,
    s: int = 16,
    rank_metric: str = "sharpe",
) -> PBOResult:
    """Compute Probability of Backtest Overfitting.

    `returns_matrix` shape: (n_periods, n_trials). Each column is one
    strategy's per-period returns. Selection rule = "pick the strategy
    with the best in-sample metric"; PBO = P(selected strategy ranks in
    bottom half of OOS).
    """
    arr = np.asarray(returns_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return PBOResult(pbo=1.0, n_trials=int(arr.shape[1] if arr.ndim == 2 else 0),
                         n_partitions=0, logits=[], best_is_strategy_indices=[])
    n_periods, n_trials = arr.shape
    if rank_metric != "sharpe":
        raise ValueError("only rank_metric='sharpe' implemented")
    partitions = cscv_partitions(n_periods, s=s)
    logits: list[float] = []
    best_is: list[int] = []
    for is_idx, oos_idx in partitions:
        is_sharpe = np.array([_sharpe(arr[is_idx, k]) for k in range(n_trials)])
        oos_sharpe = np.array([_sharpe(arr[oos_idx, k]) for k in range(n_trials)])
        best_strategy = int(np.argmax(is_sharpe))
        best_is.append(best_strategy)
        # OOS rank of the IS-selected strategy (rank 1 = best).
        oos_ranks = (-oos_sharpe).argsort().argsort() + 1
        r = int(oos_ranks[best_strategy])
        # Logit-transform the relative rank in [0, 1].
        omega = r / (n_trials + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))
    # PBO = fraction of partitions where the IS-best landed in the bottom half OOS.
    pbo = float(np.mean(np.array(logits) > 0))
    return PBOResult(
        pbo=pbo, n_trials=n_trials, n_partitions=len(partitions),
        logits=logits, best_is_strategy_indices=best_is,
    )


def expected_max_sharpe(n_trials: int, *, mean_sharpe: float = 0.0,
                       std_sharpe: float = 1.0) -> float:
    """Expected maximum of N independent N(μ, σ²) Sharpes — for Deflated Sharpe.

    Uses the asymptotic order-statistic approximation from Bailey-Lopez de Prado:
        E[max] ≈ μ + σ × ((1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N·e)))
    with γ ≈ 0.5772 (Euler-Mascheroni).
    """
    if n_trials <= 1:
        return mean_sharpe
    GAMMA = 0.5772156649
    e = math.e
    a = _phi_inv(1.0 - 1.0 / n_trials)
    b = _phi_inv(1.0 - 1.0 / (n_trials * e))
    return mean_sharpe + std_sharpe * ((1.0 - GAMMA) * a + GAMMA * b)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    sharpe_trial_mean: float = 0.0,
    sharpe_trial_std: float = 1.0,
) -> DeflatedSharpeResult:
    """Bailey-López-de-Prado Deflated Sharpe Ratio.

    Inputs:
      * `returns`            — observed return stream (per-period)
      * `n_trials`           — number of strategy variants you tried before
                               picking the current one (selection bias)
      * `sharpe_trial_mean`  — mean Sharpe across all trial variants (if known)
      * `sharpe_trial_std`   — std-dev of Sharpe across all trial variants

    Returns the deflated Sharpe + the probability that the *true* Sharpe is
    above zero. Reject anything with p_value > 0.05 — the apparent edge is
    likely a selection artifact.
    """
    arr = np.asarray(returns, dtype=np.float64)
    n = arr.size
    if n < 30:
        return DeflatedSharpeResult(
            observed_sharpe=_sharpe(arr), deflated_sharpe=0.0,
            p_value=1.0, n_trials=n_trials, n_observations=n,
        )
    sr = _sharpe(arr)
    # Per-period skewness / kurtosis.
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd < 1e-12:
        return DeflatedSharpeResult(
            observed_sharpe=0.0, deflated_sharpe=0.0,
            p_value=1.0, n_trials=n_trials, n_observations=n,
        )
    z = (arr - mu) / sd
    g3 = float(np.mean(z ** 3))                 # skewness
    g4 = float(np.mean(z ** 4) - 3.0)           # excess kurtosis
    # Expected-max Sharpe under N trials of unit variance.
    sr_max = expected_max_sharpe(n_trials, mean_sharpe=sharpe_trial_mean,
                                 std_sharpe=sharpe_trial_std)
    # Standard error of the *observed* Sharpe accounting for non-normality.
    var_sr = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr) / (n - 1)
    se_sr = math.sqrt(max(var_sr, 1e-12))
    if se_sr <= 0:
        return DeflatedSharpeResult(
            observed_sharpe=sr, deflated_sharpe=0.0,
            p_value=1.0, n_trials=n_trials, n_observations=n,
        )
    # Probability the true Sharpe exceeds the expected-max under H0.
    p_significant = _phi_cdf((sr - sr_max) / se_sr)
    deflated = (sr - sr_max) / se_sr
    return DeflatedSharpeResult(
        observed_sharpe=sr,
        deflated_sharpe=deflated,
        p_value=1.0 - p_significant,
        n_trials=n_trials,
        n_observations=n,
    )


def is_overfit(pbo: PBOResult, *, threshold: float = 0.5) -> bool:
    """Reject the selection rule when PBO exceeds 50% — picking is worse than random."""
    return pbo.pbo > threshold


def passes_deflated_sharpe(result: DeflatedSharpeResult, *, alpha: float = 0.05) -> bool:
    """Promotion gate: the deflated Sharpe must be significant at `alpha`."""
    return result.p_value <= alpha and result.deflated_sharpe > 0.0
