"""Fernet wrapper for at-rest secrets in `settings_kv`.

The MT5 password and API keys never live in plaintext in SQLite. The Fernet
key itself is stored in `.env` (`FERNET_KEY`); if absent we generate one and
write it back so the operator's first launch is zero-touch.

Round-trip helpers `encrypt_to_str()` / `decrypt_from_str()` work over UTF-8
strings; payloads are base64 url-safe so they survive every storage layer
without escaping.
"""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from engine.config import settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def _ensure_key() -> bytes:
    """Return the Fernet key bytes; generate + persist one if missing."""
    raw = os.environ.get("FERNET_KEY") or ""
    if raw:
        return raw.encode("ascii")
    key = Fernet.generate_key()
    # Persist so subsequent launches reuse it. Append rather than rewrite to
    # avoid clobbering manual edits in .env.
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text(encoding="utf-8")
        if "FERNET_KEY=" not in existing:
            with ENV_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"\nFERNET_KEY={key.decode('ascii')}\n")
    os.environ["FERNET_KEY"] = key.decode("ascii")
    return key


def _cipher() -> Fernet:
    return Fernet(_ensure_key())


def encrypt_to_str(plaintext: str) -> str:
    return _cipher().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_from_str(token: str) -> str:
    try:
        return _cipher().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("decrypt failed (wrong FERNET_KEY?)") from e


# Convenience: SQLite-backed credential store. Engine-side getter checks
# settings_kv first, then falls back to the plaintext env var.

ENCRYPTED_KEYS = frozenset({
    "MT5_PASSWORD", "ANTHROPIC_API_KEY", "SUPABASE_SERVICE_ROLE_KEY",
    "DISCORD_WEBHOOK_URL", "DISCORD_ERROR_WEBHOOK_URL",
    "FOREX_CALENDAR_API_KEY", "FINNHUB_API_KEY", "JBLANKED_API_KEY",
})


def store_secret(key: str, plaintext: str) -> None:
    from engine.data.sqlite_journal import open_journal
    token = encrypt_to_str(plaintext)
    with open_journal() as con:
        con.execute(
            "INSERT INTO settings_kv(k,v) VALUES(?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (f"sec:{key}", token),
        )
        con.commit()


def fetch_secret(key: str) -> str | None:
    from engine.data.sqlite_journal import open_journal
    with open_journal() as con:
        row = con.execute(
            "SELECT v FROM settings_kv WHERE k=?", (f"sec:{key}",)
        ).fetchone()
    if row is None:
        # Fall back to env so existing .env-only setups keep working.
        return getattr(settings, key, None) or os.environ.get(key)
    try:
        return decrypt_from_str(row["v"])
    except ValueError:
        return None
