from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal


N_BINS = 10
MIN_TRADES_FOR_ECE = 30


@dataclass(frozen=True)
class CalibrationBin:
    bin_start: float
    bin_end: float
    n: int
    avg_confidence: float
    win_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "bin_start": self.bin_start,
            "bin_end": self.bin_end,
            "n": float(self.n),
            "avg_confidence": self.avg_confidence,
            "win_rate": self.win_rate,
        }


@dataclass
class CalibrationResult:
    ece_score: float
    n_trades: int
    bins: list[CalibrationBin]
    overconfident: bool


def _bucketize(confidences: list[float], outcomes: list[int]) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    for k in range(N_BINS):
        lo = k / N_BINS
        hi = (k + 1) / N_BINS
        members = [
            (c, o) for c, o in zip(confidences, outcomes)
            if (lo <= c < hi) or (k == N_BINS - 1 and c == 1.0)
        ]
        if not members:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        avg_conf = sum(c for c, _ in members) / len(members)
        wr = sum(o for _, o in members) / len(members)
        bins.append(CalibrationBin(lo, hi, len(members), avg_conf, wr))
    return bins


def compute_ece(confidences: list[float], outcomes: list[int]) -> CalibrationResult:
    n = len(confidences)
    if n == 0 or n != len(outcomes):
        return CalibrationResult(0.0, 0, [], False)
    bins = _bucketize(confidences, outcomes)
    ece = 0.0
    for b in bins:
        if b.n == 0:
            continue
        ece += (b.n / n) * abs(b.avg_confidence - b.win_rate)
    overconfident = ece > settings.ECE_OVERCONFIDENT_THRESHOLD
    return CalibrationResult(ece_score=ece, n_trades=n, bins=bins, overconfident=overconfident)


def compute_brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Mean squared error of probabilistic predictions vs outcomes.

    Captures *sharpness* (resolution) in addition to calibration. A model
    that always predicts 0.5 has perfect calibration but Brier = 0.25 — bad.
    A confident-and-correct model gets Brier near 0.
    """
    if not confidences or len(confidences) != len(outcomes):
        return 0.0
    total = 0.0
    for c, o in zip(confidences, outcomes):
        total += (c - o) ** 2
    return total / len(confidences)


def hosmer_lemeshow_test(
    confidences: list[float],
    outcomes: list[int],
    n_groups: int = 10,
) -> tuple[float, float]:
    """Hosmer–Lemeshow goodness-of-fit test.

    Returns (chi_square_statistic, approx_p_value). Low p-value (<0.05) means
    the calibration deviation across bins is statistically significant — the
    model's predicted probabilities don't match observed frequencies.

    The p-value is computed against a chi-square distribution with
    (n_groups - 2) degrees of freedom via a survival-function approximation.
    """
    n = len(confidences)
    if n == 0 or n != len(outcomes):
        return 0.0, 1.0
    pairs = sorted(zip(confidences, outcomes))
    group_size = max(1, n // n_groups)
    chi2 = 0.0
    actual_groups = 0
    for g in range(n_groups):
        lo = g * group_size
        hi = (g + 1) * group_size if g < n_groups - 1 else n
        if hi <= lo:
            continue
        bucket = pairs[lo:hi]
        m = len(bucket)
        observed_wins = sum(o for _, o in bucket)
        expected_wins = sum(c for c, _ in bucket)
        expected_losses = m - expected_wins
        if expected_wins <= 1e-9 or expected_losses <= 1e-9:
            continue
        chi2 += ((observed_wins - expected_wins) ** 2) / expected_wins
        chi2 += (((m - observed_wins) - expected_losses) ** 2) / expected_losses
        actual_groups += 1
    df = max(1, actual_groups - 2)
    # Survival function of chi-square via regularized upper incomplete gamma.
    # Wilson-Hilferty approximation gives a good p-value for df in [1, 30].
    if chi2 <= 0:
        return 0.0, 1.0
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    # Standard normal survival function (1 - Phi(z)).
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return chi2, float(min(1.0, max(0.0, p)))


def reliability_diagram(
    confidences: list[float],
    outcomes: list[int],
) -> list[dict[str, float]]:
    """Bin-by-bin (avg_confidence, win_rate, n) — direct input for UI plotting."""
    bins = _bucketize(confidences, outcomes)
    return [
        {
            "bin_start": b.bin_start,
            "bin_end": b.bin_end,
            "n": b.n,
            "avg_confidence": b.avg_confidence,
            "win_rate": b.win_rate,
            "gap": b.avg_confidence - b.win_rate,
        }
        for b in bins
    ]


def _load_calibration_inputs(*, db_path: str | None = None) -> tuple[list[float], list[int]]:
    confs: list[float] = []
    outcomes: list[int] = []
    with open_journal(db_path) as con:
        rows = con.execute(
            """
            SELECT claude_confidence, hypothetical_outcome
              FROM shadow_trades
             WHERE hypothetical_outcome IN ('WIN','LOSS')
               AND claude_confidence IS NOT NULL
             ORDER BY id DESC
             LIMIT 500
            """
        ).fetchall()
    for r in rows:
        conf = float(r["claude_confidence"]) / 100.0
        win = 1 if r["hypothetical_outcome"] == "WIN" else 0
        confs.append(conf)
        outcomes.append(win)
    return confs, outcomes


def _persist(result: CalibrationResult, *, db_path: str | None = None) -> None:
    if result.n_trades == 0:
        return
    payload = json.dumps([b.as_dict() for b in result.bins], separators=(",", ":"))
    ts = datetime.now(timezone.utc).isoformat()
    with open_journal(db_path) as con:
        con.execute(
            "INSERT INTO calibration_history (timestamp, ece_score, n_trades, bin_data_json) "
            "VALUES (?, ?, ?, ?)",
            (ts, float(result.ece_score), int(result.n_trades), payload),
        )
        con.commit()


def recompute_and_persist(*, db_path: str | None = None) -> CalibrationResult:
    confs, outcomes = _load_calibration_inputs(db_path=db_path)
    if len(confs) < MIN_TRADES_FOR_ECE:
        return CalibrationResult(0.0, len(confs), [], False)
    result = compute_ece(confs, outcomes)
    _persist(result, db_path=db_path)
    if result.overconfident:
        logger.warning(
            "Calibration: ECE={:.3f} on {} trades — model is OVERCONFIDENT (threshold {:.2f})",
            result.ece_score, result.n_trades, settings.ECE_OVERCONFIDENT_THRESHOLD,
        )
    else:
        logger.info(
            "Calibration: ECE={:.3f} on {} trades (well calibrated)",
            result.ece_score, result.n_trades,
        )
    return result


def latest_calibration(*, db_path: str | None = None) -> CalibrationResult | None:
    with open_journal(db_path) as con:
        row = con.execute(
            "SELECT ece_score, n_trades, bin_data_json FROM calibration_history "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    try:
        bins_raw = json.loads(row["bin_data_json"])
    except (TypeError, json.JSONDecodeError):
        bins_raw = []
    bins = [
        CalibrationBin(
            bin_start=float(b["bin_start"]), bin_end=float(b["bin_end"]),
            n=int(b["n"]), avg_confidence=float(b["avg_confidence"]),
            win_rate=float(b["win_rate"]),
        )
        for b in bins_raw
    ]
    return CalibrationResult(
        ece_score=float(row["ece_score"]), n_trades=int(row["n_trades"]),
        bins=bins, overconfident=float(row["ece_score"]) > settings.ECE_OVERCONFIDENT_THRESHOLD,
    )


def closed_shadow_trade_count(*, db_path: str | None = None) -> int:
    with open_journal(db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM shadow_trades WHERE hypothetical_outcome IN ('WIN','LOSS')"
        ).fetchone()
    return int(row["n"]) if row else 0
