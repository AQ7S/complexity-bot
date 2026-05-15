"""Phase 15 — Fernet round-trip + SQLite-backed credential store."""
from __future__ import annotations

import importlib

import pytest
from cryptography.fernet import Fernet

from engine.utils import encryption


@pytest.fixture(autouse=True)
def _fixed_key(monkeypatch):
    monkeypatch.setenv("FERNET_KEY", Fernet.generate_key().decode("ascii"))
    importlib.reload(encryption)
    yield


def test_encrypt_decrypt_roundtrip():
    secret = "hunter2-MT5-pw"
    token = encryption.encrypt_to_str(secret)
    assert token != secret
    assert encryption.decrypt_from_str(token) == secret


def test_decrypt_rejects_garbage():
    with pytest.raises(ValueError):
        encryption.decrypt_from_str("not-a-real-token")


def test_store_and_fetch_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "journal.sqlite"))
    from engine.config import settings as _s
    importlib.reload(_s)
    from engine.data import sqlite_journal
    importlib.reload(sqlite_journal)
    importlib.reload(encryption)
    sqlite_journal.init_db()

    encryption.store_secret("ANTHROPIC_API_KEY", "sk-ant-test-XYZ")
    got = encryption.fetch_secret("ANTHROPIC_API_KEY")
    assert got == "sk-ant-test-XYZ"

    # Stored token must be unreadable from raw SQLite.
    with sqlite_journal.open_journal() as con:
        row = con.execute(
            "SELECT v FROM settings_kv WHERE k='sec:ANTHROPIC_API_KEY'"
        ).fetchone()
    assert "sk-ant" not in row["v"]
