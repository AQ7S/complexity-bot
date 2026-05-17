"""Risk-of-Ruin (RoR) calculator.

Given the strategy's current win rate, average reward:risk ratio, risk
per trade as fraction of equity, and the operator's "ruin" threshold
(typically max drawdown the broker / operator can tolerate), compute the
probability that equity hits the ruin threshold within a target number
of trades.

Two approximations:

  * **Analytic (gambler's ruin):** for a fixed fractional risk f and
    fixed reward:risk ratio b, the probability of eventually being
    ruined from N units to 0 in a biased random walk is:
        if expected return ≤ 0: P_ruin = 1
        else:                    P_ruin ≈ ((1-p)/p)^N / b
    using "units" = floor(max_drawdown_pct / risk_per_trade).
    Standard reference: Vince, "Mathematics of Money Management".

  * **Monte Carlo (more honest for variable b):** simulate N_paths
    independent trade sequences, each of `horizon_trades` steps, and
    count the fraction that touch the ruin threshold. The MC version
    correctly handles fractional Kelly and equity-relative risk.

The Bayesian update helper `update_with_trades()` lets the engine
re-estimate RoR after every N closed trades as the win rate / payoff
estimates sharpen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuinEstimate:
    p_ruin: float
    method: str
    n_paths: int = 0
    horizon_trades: int = 0
    expected_final_equity_pct: float | None = None


def analytic_ruin(
    *,
    win_rate: float,
    rr_ratio: float,
    risk_per_trade_pct: float,
    max_drawdown_pct: float,
) -> RuinEstimate:
    """Closed-form approximation for fixed risk + fixed payoff."""
    if not 0 < win_rate < 1:
        return RuinEstimate(p_ruin=1.0, method="analytic")
    if risk_per_trade_pct <= 0 or max_drawdown_pct <= 0:
        return RuinEstimate(p_ruin=1.0, method="analytic")
    expectancy = win_rate * rr_ratio - (1 - win_rate)
    if expectancy <= 0:
        return RuinEstimate(p_ruin=1.0, method="analytic")
    # Units of risk between current equity and ruin.
    n_units = max(1, int(math.floor(max_drawdown_pct / risk_per_trade_pct)))
    # Vince's biased random-walk approximation.
    q_over_p = (1.0 - win_rate) / win_rate
    p_ruin = (q_over_p ** n_units) / max(rr_ratio, 1e-9)
    p_ruin = max(0.0, min(1.0, p_ruin))
    return RuinEstimate(p_ruin=p_ruin, method="analytic")


def monte_carlo_ruin(
    *,
    win_rate: float,
    rr_ratio: float,
    risk_per_trade_pct: float,
    max_drawdown_pct: float,
    horizon_trades: int = 1000,
    n_paths: int = 5000,
    seed: int | None = 17,
) -> RuinEstimate:
    """Monte Carlo simulation of the equity path under fixed fractional risk.

    Each trade: equity *= (1 + risk_per_trade_pct * rr_ratio) on win,
                       *= (1 - risk_per_trade_pct) on loss.
    Ruin = equity falls by `max_drawdown_pct` from its starting value.
    """
    if not 0 < win_rate < 1:
        return RuinEstimate(p_ruin=1.0, method="monte_carlo",
                            n_paths=n_paths, horizon_trades=horizon_trades)
    rng = np.random.default_rng(seed)
    ruined = 0
    final_equities: list[float] = []
    ruin_eq = 1.0 - max_drawdown_pct
    for _ in range(n_paths):
        eq = 1.0
        is_ruined = False
        for _ in range(horizon_trades):
            if rng.random() < win_rate:
                eq *= 1.0 + risk_per_trade_pct * rr_ratio
            else:
                eq *= 1.0 - risk_per_trade_pct
            if eq <= ruin_eq:
                is_ruined = True
                break
        if is_ruined:
            ruined += 1
        final_equities.append(eq)
    return RuinEstimate(
        p_ruin=ruined / n_paths,
        method="monte_carlo",
        n_paths=n_paths,
        horizon_trades=horizon_trades,
        expected_final_equity_pct=float(np.mean(final_equities) - 1.0),
    )


def update_with_trades(
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    wins: int = 0,
    losses: int = 0,
    rr_ratio: float = 2.0,
    risk_per_trade_pct: float = 0.02,
    max_drawdown_pct: float = 0.20,
    method: str = "analytic",
) -> RuinEstimate:
    """Bayesian update of win-rate (Beta conjugate prior) → RoR estimate.

    Posterior win rate = (α + wins) / (α + β + wins + losses).
    """
    alpha_post = prior_alpha + max(0, wins)
    beta_post = prior_beta + max(0, losses)
    p_win = alpha_post / (alpha_post + beta_post)
    if method == "monte_carlo":
        return monte_carlo_ruin(
            win_rate=p_win, rr_ratio=rr_ratio,
            risk_per_trade_pct=risk_per_trade_pct,
            max_drawdown_pct=max_drawdown_pct,
        )
    return analytic_ruin(
        win_rate=p_win, rr_ratio=rr_ratio,
        risk_per_trade_pct=risk_per_trade_pct,
        max_drawdown_pct=max_drawdown_pct,
    )
