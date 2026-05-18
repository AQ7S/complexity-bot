"""Tests for hash-chained audit log (Tier 8.8)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engine.data.audit_log import (
    GENESIS_HASH,
    append_entry,
    read_chain,
    reset_cache_for_tests,
    verify_chain,
)


@pytest.fixture()
def tmp_log():
    reset_cache_for_tests()
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "audit.jsonl"
    reset_cache_for_tests()


def test_first_entry_uses_genesis(tmp_log):
    e = append_entry("test", {"value": 1}, path=tmp_log)
    assert e.prev_hash == GENESIS_HASH
    assert len(e.hash) == 64


def test_chain_links(tmp_log):
    e1 = append_entry("a", {"x": 1}, path=tmp_log)
    e2 = append_entry("b", {"y": 2}, path=tmp_log)
    assert e2.prev_hash == e1.hash


def test_verify_chain_intact(tmp_log):
    for i in range(5):
        append_entry("event", {"i": i}, path=tmp_log)
    assert verify_chain(tmp_log) is None


def test_verify_detects_tampering(tmp_log):
    for i in range(3):
        append_entry("event", {"i": i}, path=tmp_log)
    lines = tmp_log.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[1])
    obj["payload"]["i"] = 999
    lines[1] = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    tmp_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    broken_at = verify_chain(tmp_log)
    assert broken_at == 1


def test_read_chain_yields_entries(tmp_log):
    append_entry("a", {"i": 1}, path=tmp_log)
    append_entry("b", {"i": 2}, path=tmp_log)
    entries = list(read_chain(tmp_log))
    assert len(entries) == 2
    assert entries[0].kind == "a"
    assert entries[1].kind == "b"


def test_verify_on_missing_file_returns_none():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nope.jsonl"
        assert verify_chain(p) is None


def test_cache_survives_writes(tmp_log):
    e1 = append_entry("a", {"x": 1}, path=tmp_log)
    e2 = append_entry("b", {"y": 2}, path=tmp_log)
    e3 = append_entry("c", {"z": 3}, path=tmp_log)
    assert e2.prev_hash == e1.hash
    assert e3.prev_hash == e2.hash
