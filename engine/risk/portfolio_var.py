"""Portfolio Value-at-Risk and Conditional Value-at-Risk.

Per-trade risk caps are necessary but not sufficient: five 2%-risk
trades on correlated assets can lose 10% jointly even though each
individual trade only risks 2%. Portfolio VaR enforces a cap on the
*joint* potential loss across all open positions, computed from
historical returns + the rolling correlation matrix.

Definitions:
    VaR_α  = the loss threshold that won't be exceeded with probability α.
    CVaR_α = average loss in the worst (1−α) tail.

Hard portfolio cap (default):  VaR_95 ≤ 5% of equity.
The risk manager refuses to admit a new position that would push the
*projected* portfolio VaR over this cap.

Two implementations:
  * Historical (non-parametric) — uses an empirical distribution of
    portfolio returns reconstructed from per-symbol historical returns
    + position weights. Robust to fat tails. Default.
  * Parametric (Gaussian) — analytical via covariance + z-score. Faster,
    but misleading on fat-tailed series. Provided for diagnostic use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VAR_CAP_DEFAULT = 0.05      # 5% of equity
CONFIDENCE_DEFAULT = 0.95
LOOKBACK_DEFAULT_DAYS = 250


@dataclass(frozen=True)
class Position:
    symbol: str
    direction: str            # "BUY" | "SELL"
    notional_usd: float       # signed by direction: positive long, negative short


@dataclass
class VaRReport:
    var_pct: float
    cvar_pct: float
    var_usd: float
    cvar_usd: float
    method: str
    confidence: float
    n_observations: int


def _position_weights(positions: list[Position], equity: float) -> dict[str, float]:
    if equity <= 0:
        return {}
    out: dict[str, float] = {}
    for p in positions:
        w = p.notional_usd / equity
        if p.direction.upper() == "SELL":
            w = -abs(w)
        else:
            w = abs(w)
        out[p.symbol] = out.get(p.symbol, 0.0) + w
    return out


def historical_var(
    returns_by_symbol: dict[str, np.ndarray],
    positions: list[Position],
    equity: float,
    *,
    confidence: float = CONFIDENCE_DEFAULT,
) -> VaRReport:
    """Historical (non-parametric) VaR / CVaR at the given confidence."""
    if equity <= 0 or not positions or not returns_by_symbol:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "historical", confidence, 0)
    weights = _position_weights(positions, equity)
    syms = [s for s in weights if s in returns_by_symbol]
    if not syms:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "historical", confidence, 0)
    # Align lengths to the shortest series.
    n = min(len(returns_by_symbol[s]) for s in syms)
    if n < 5:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "historical", confidence, n)
    port_returns = np.zeros(n, dtype=np.float64)
    for s in syms:
        port_returns += weights[s] * returns_by_symbol[s][-n:]
    losses = -port_returns  # positive = loss
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return VaRReport(
        var_pct=max(0.0, var),
        cvar_pct=max(0.0, cvar),
        var_usd=max(0.0, var) * equity,
        cvar_usd=max(0.0, cvar) * equity,
        method="historical",
        confidence=confidence,
        n_observations=n,
    )


def parametric_var(
    returns_by_symbol: dict[str, np.ndarray],
    positions: list[Position],
    equity: float,
    *,
    confidence: float = CONFIDENCE_DEFAULT,
) -> VaRReport:
    """Gaussian (variance-covariance) VaR / CVaR."""
    from math import erfc, sqrt
    if equity <= 0 or not positions or not returns_by_symbol:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "parametric", confidence, 0)
    weights = _position_weights(positions, equity)
    syms = [s for s in weights if s in returns_by_symbol]
    if not syms:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "parametric", confidence, 0)
    n = min(len(returns_by_symbol[s]) for s in syms)
    if n < 5:
        return VaRReport(0.0, 0.0, 0.0, 0.0, "parametric", confidence, n)
    mat = np.vstack([returns_by_symbol[s][-n:] for s in syms])  # (S, n)
    cov = np.cov(mat, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    w_vec = np.array([weights[s] for s in syms], dtype=np.float64)
    port_var_per_period = float(w_vec @ cov @ w_vec)
    port_std = float(np.sqrt(max(port_var_per_period, 0.0)))
    # One-sided normal: z s.t. Phi(z) = confidence.
    from math import sqrt as msqrt
    def _normal_ppf(p: float) -> float:
        # Beasley-Springer-Moro approximation.
        if p < 0.5:
            return -_normal_ppf(1.0 - p)
        t = msqrt(-2.0 * np.log(1.0 - p))
        return float(t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                     / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t ** 3))
    z = _normal_ppf(confidence)
    var = z * port_std
    pdf = float(np.exp(-z * z / 2.0) / msqrt(2.0 * np.pi))
    cvar = (pdf / (1.0 - confidence)) * port_std
    return VaRReport(
        var_pct=max(0.0, var),
        cvar_pct=max(0.0, cvar),
        var_usd=max(0.0, var) * equity,
        cvar_usd=max(0.0, cvar) * equity,
        method="parametric",
        confidence=confidence,
        n_observations=n,
    )


def var_breach_predictor(
    new_position: Position,
    existing_positions: list[Position],
    returns_by_symbol: dict[str, np.ndarray],
    equity: float,
    *,
    var_cap: float = VAR_CAP_DEFAULT,
    confidence: float = CONFIDENCE_DEFAULT,
) -> tuple[bool, VaRReport]:
    """Project portfolio VaR with `new_position` added. Returns (would_breach, report)."""
    projected = existing_positions + [new_position]
    report = historical_var(returns_by_symbol, projected, equity, confidence=confidence)
    return (report.var_pct > var_cap), report
