"""Tests for the Claude meta-policy (Tier 3.5)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from engine.data import sqlite_journal
from engine.learning.claude_meta_policy import (
    ALLOWED_PARAMS,
    Override,
    active_overrides,
    apply_overrides,
    propose_overrides,
    reset_overrides,
)


def _isolated_journal():
    """Create a brand-new isolated SQLite journal and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    path = tmp.name
    with sqlite_journal.open_journal(path) as con:
        # Schema is auto-applied by open_journal.
        pass
    return path


def test_unknown_param_dropped():
    db = _isolated_journal()

    def claude(_payload):
        return json.dumps([{"param": "non_existent", "new_value": 99, "rationale": "x"}])

    overrides = propose_overrides([], claude_caller=claude)
    assert overrides == []
    apply_overrides(overrides, db_path=db)
    assert active_overrides(db_path=db) == {}


def test_param_clamped_to_range():
    db = _isolated_journal()

    def claude(_payload):
        return json.dumps([{"param": "consensus.min_agree", "new_value": 99, "rationale": "max"}])

    overrides = propose_overrides([], claude_caller=claude)
    assert len(overrides) == 1
    assert overrides[0].new_value == 6  # clamp at upper bound


def test_bool_param_coerced():
    db = _isolated_journal()

    def claude(_payload):
        return json.dumps([
            {"param": "risk.smc_filter_required", "new_value": "true", "rationale": "test"},
            {"param": "consensus.po3_bonus_enabled", "new_value": False, "rationale": "off"},
        ])

    overrides = propose_overrides([], claude_caller=claude)
    assert any(o.new_value is True for o in overrides)
    assert any(o.new_value is False for o in overrides)


def test_apply_and_active_roundtrip():
    db = _isolated_journal()

    def claude(_payload):
        return json.dumps([{"param": "consensus.min_agree", "new_value": 4, "rationale": "tighten"}])

    overrides = propose_overrides([], claude_caller=claude)
    n = apply_overrides(overrides, db_path=db)
    assert n == 1
    active = active_overrides(db_path=db)
    assert active.get("consensus.min_agree") == 4


def test_reset_disables_all():
    db = _isolated_journal()

    def claude(_payload):
        return json.dumps([{"param": "spread.acceptable_multiplier", "new_value": 2.5, "rationale": ""}])

    overrides = propose_overrides([], claude_caller=claude)
    apply_overrides(overrides, db_path=db)
    assert active_overrides(db_path=db)
    n = reset_overrides(db_path=db)
    assert n >= 1
    assert active_overrides(db_path=db) == {}


def test_malformed_response_returns_empty():
    overrides = propose_overrides([], claude_caller=lambda _p: "not json at all")
    assert overrides == []


def test_max_three_overrides_taken():
    def claude(_payload):
        # Send 5 valid; expect at most 3 to be accepted.
        return json.dumps([
            {"param": "consensus.min_agree", "new_value": 4, "rationale": ""},
            {"param": "spread.acceptable_multiplier", "new_value": 2.0, "rationale": ""},
            {"param": "claude_gate.min_confidence_normal", "new_value": 60, "rationale": ""},
            {"param": "risk.smc_filter_required", "new_value": True, "rationale": ""},
            {"param": "consensus.po3_bonus_enabled", "new_value": True, "rationale": ""},
        ])

    overrides = propose_overrides([], claude_caller=claude)
    assert len(overrides) == 3


def test_allowed_params_table_intact():
    # Anti-tamper guard: every required key still present.
    for key in [
        "consensus.min_agree", "consensus.po3_bonus_enabled",
        "claude_gate.min_confidence_normal", "claude_gate.min_confidence_fallback",
        "risk.smc_filter_required", "risk.killzone_strict_outside_overlap",
        "spread.acceptable_multiplier",
    ]:
        assert key in ALLOWED_PARAMS
