"""Cointegration testing + pairs-trading primitives (Tier 8.3).

This module is the math under the new `PairsTradingStrategy`. We do not
require statsmodels — Engle-Granger is implementable from scratch with
numpy (OLS + augmented Dickey-Fuller via DF tabulated critical values).

Why it matters for retail:
    Two cointegrated FX pairs (e.g. EUR/USD and GBP/USD) drift apart and
    back together in the long run. Trading the *spread* (long one short
    the other in the ratio determined by the OLS hedge ratio) has far
    lower volatility than either leg alone — so transaction costs eat
    a smaller share of the edge. This is one of the few structural
    retail edges left after market-maker spreads.

Pipeline:
    1. Verify each individual series is non-stationary (ADF can't reject
       unit root at 5%).
    2. Fit OLS: y = α + β·x. β is the hedge ratio.
    3. Test the residuals for stationarity via ADF.
    4. If residuals are stationary → the pair is cointegrated.
    5. Trade the residual's mean-reversion: z = (residual − μ) / σ;
       enter at |z| > z_entry, exit at |z| < z_exit, stop at |z| > z_stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# Approximate ADF critical values (Hamilton 1994, no-constant variant).
# Sample sizes: 25, 50, 100, 250, 500.
_ADF_CRITICAL = {
    0.01: [-2.66, -2.62, -2.60, -2.58, -2.58],
    0.05: [-1.95, -1.95, -1.95, -1.95, -1.95],
    0.10: [-1.60, -1.61, -1.61, -1.62, -1.62],
}
_ADF_SAMPLE_SIZES = [25, 50, 100, 250, 500]


@dataclass(frozen=True)
class ADFResult:
    statistic: float
    critical_5pct: float
    n: int
    is_stationary: bool


@dataclass(frozen=True)
class CointegrationResult:
    is_cointegrated: bool
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    adf_stat: float
    adf_critical_5pct: float
    n: int


def _adf_critical_value(n: int, alpha: float = 0.05) -> float:
    crit = _ADF_CRITICAL.get(alpha)
    if crit is None:
        raise ValueError(f"unsupported alpha {alpha}")
    if n <= _ADF_SAMPLE_SIZES[0]:
        return crit[0]
    if n >= _ADF_SAMPLE_SIZES[-1]:
        return crit[-1]
    for i, ss in enumerate(_ADF_SAMPLE_SIZES[1:], start=1):
        if n <= ss:
            return float(crit[i])
    return float(crit[-1])


def augmented_dickey_fuller(
    series: np.ndarray,
    *,
    lags: int = 1,
    alpha: float = 0.05,
) -> ADFResult:
    """Minimal ADF test on `series`. Tests H0 = unit root (non-stationary)
    via OLS on Δy = ρ·y_{-1} + Σ φ_i·Δy_{-i} + ε. Reject H0 (i.e. series
    is stationary) when the test statistic < critical value at `alpha`.
    """
    y = np.asarray(series, dtype=np.float64).flatten()
    n = y.size
    if n < 20:
        return ADFResult(statistic=0.0, critical_5pct=-1.95, n=n, is_stationary=False)
    dy = np.diff(y)
    lagged = y[:-1]
    cols = [lagged]
    for i in range(1, lags + 1):
        if dy.size - i <= 0:
            break
        cols.append(np.concatenate([np.zeros(i), dy[:-i]]))
    X = np.column_stack(cols)
    target = dy
    if X.shape[0] != target.shape[0]:
        m = min(X.shape[0], target.shape[0])
        X = X[-m:]
        target = target[-m:]
    # OLS via lstsq.
    coefs, _resid, _rank, _sv = np.linalg.lstsq(X, target, rcond=None)
    pred = X @ coefs
    resid = target - pred
    rss = float(np.sum(resid ** 2))
    df = max(X.shape[0] - X.shape[1], 1)
    sigma2 = rss / df
    # Standard error of the first coefficient (ρ on lagged level).
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return ADFResult(statistic=0.0, critical_5pct=-1.95, n=n, is_stationary=False)
    se_rho = float(np.sqrt(max(sigma2 * XtX_inv[0, 0], 1e-12)))
    if se_rho <= 0:
        return ADFResult(statistic=0.0, critical_5pct=-1.95, n=n, is_stationary=False)
    t_stat = float(coefs[0]) / se_rho
    crit = _adf_critical_value(n, alpha=alpha)
    return ADFResult(
        statistic=t_stat,
        critical_5pct=crit,
        n=n,
        is_stationary=t_stat < crit,
    )


def engle_granger_cointegration(
    y: np.ndarray,
    x: np.ndarray,
    *,
    alpha: float = 0.05,
) -> CointegrationResult:
    """Two-step Engle-Granger cointegration test.

    Returns a `CointegrationResult` with hedge ratio, residual moments,
    and the stationarity verdict on the residuals.
    """
    y_arr = np.asarray(y, dtype=np.float64).flatten()
    x_arr = np.asarray(x, dtype=np.float64).flatten()
    n = min(y_arr.size, x_arr.size)
    if n < 30:
        return CointegrationResult(
            is_cointegrated=False, hedge_ratio=0.0, intercept=0.0,
            spread_mean=0.0, spread_std=0.0, adf_stat=0.0,
            adf_critical_5pct=-1.95, n=n,
        )
    y_arr = y_arr[-n:]
    x_arr = x_arr[-n:]
    # OLS: y = α + β x.
    X = np.column_stack([np.ones(n), x_arr])
    coefs, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
    intercept, beta = float(coefs[0]), float(coefs[1])
    residuals = y_arr - intercept - beta * x_arr
    adf = augmented_dickey_fuller(residuals, lags=1, alpha=alpha)
    return CointegrationResult(
        is_cointegrated=bool(adf.is_stationary),
        hedge_ratio=beta,
        intercept=intercept,
        spread_mean=float(residuals.mean()),
        spread_std=float(residuals.std(ddof=1)) if residuals.size > 1 else 0.0,
        adf_stat=adf.statistic,
        adf_critical_5pct=adf.critical_5pct,
        n=n,
    )


PairSide = Literal["LONG_Y_SHORT_X", "SHORT_Y_LONG_X", "FLAT"]


@dataclass(frozen=True)
class PairsSignal:
    side: PairSide
    z_score: float
    spread: float
    hedge_ratio: float
    notes: str = ""


def pairs_trade_signal(
    y_recent: np.ndarray,
    x_recent: np.ndarray,
    coint: CointegrationResult,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
) -> PairsSignal:
    """Given a fresh (y, x) bar pair and a fitted cointegration model,
    decide whether to enter / exit / hold the pair.

    `y_recent` / `x_recent` are usually just the *latest closes* (length 1),
    but accept arrays so the helper can be called from a backtester loop.
    """
    if not coint.is_cointegrated or coint.spread_std <= 0:
        return PairsSignal("FLAT", 0.0, 0.0, coint.hedge_ratio, "not cointegrated")
    yv = float(np.asarray(y_recent).flatten()[-1])
    xv = float(np.asarray(x_recent).flatten()[-1])
    spread = yv - coint.intercept - coint.hedge_ratio * xv
    z = (spread - coint.spread_mean) / coint.spread_std
    if abs(z) > z_stop:
        return PairsSignal("FLAT", z, spread, coint.hedge_ratio, "stop hit")
    if z > z_entry:
        return PairsSignal("SHORT_Y_LONG_X", z, spread, coint.hedge_ratio,
                            "spread overstretched up — mean reversion short")
    if z < -z_entry:
        return PairsSignal("LONG_Y_SHORT_X", z, spread, coint.hedge_ratio,
                            "spread overstretched down — mean reversion long")
    if abs(z) < z_exit:
        return PairsSignal("FLAT", z, spread, coint.hedge_ratio, "spread reverted — exit")
    return PairsSignal("FLAT", z, spread, coint.hedge_ratio, "within band — hold")
