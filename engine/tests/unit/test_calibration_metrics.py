"""Tests for extended calibration metrics (Tier 1.4)."""
from __future__ import annotations

import math

from engine.learning.calibration import (
    compute_brier_score,
    compute_ece,
    hosmer_lemeshow_test,
    reliability_diagram,
)


def test_brier_perfect_zero():
    confs = [1.0, 0.0, 1.0, 0.0]
    outs = [1, 0, 1, 0]
    b = compute_brier_score(confs, outs)
    assert b == 0.0


def test_brier_worst_one():
    confs = [1.0, 0.0]
    outs = [0, 1]
    b = compute_brier_score(confs, outs)
    assert b == 1.0


def test_brier_uniform_predictions_quarter():
    confs = [0.5] * 20
    outs = [0, 1] * 10
    b = compute_brier_score(confs, outs)
    assert math.isclose(b, 0.25, abs_tol=1e-9)


def test_hl_well_calibrated_high_pvalue():
    rng_confs = [i / 100.0 for i in range(100)]
    rng_outs = [1 if i / 100.0 > 0.5 else 0 for i in range(100)]
    # Roughly calibrated: predicted prob ~ outcome — but coarse.
    chi2, p = hosmer_lemeshow_test(rng_confs, rng_outs, n_groups=5)
    assert chi2 >= 0
    assert 0.0 <= p <= 1.0


def test_hl_miscalibrated_low_pvalue():
    # Always predicts 0.9 but only wins half the time → poor calibration.
    confs = [0.9] * 100
    outs = ([1] * 50) + ([0] * 50)
    chi2, p = hosmer_lemeshow_test(confs, outs, n_groups=5)
    assert chi2 > 5.0
    assert p < 0.5


def test_reliability_diagram_returns_ten_bins():
    confs = [i / 100.0 for i in range(100)]
    outs = [i % 2 for i in range(100)]
    rd = reliability_diagram(confs, outs)
    assert len(rd) == 10
    for bin_ in rd:
        assert "bin_start" in bin_
        assert "win_rate" in bin_
        assert "gap" in bin_


def test_ece_zero_on_perfect_calibration():
    confs = [0.1] * 10 + [0.9] * 10
    outs = [0] * 9 + [1] + [1] * 9 + [0]
    res = compute_ece(confs, outs)
    assert res.n_trades == 20
    assert res.ece_score < 0.05
