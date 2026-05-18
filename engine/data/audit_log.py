"""Hash-chained tamper-evident audit log.

Every signal admitted, decision made, fill received, override applied,
and kill triggered gets one line in `engine/logs/audit.jsonl`. Each
line embeds the hash of the previous line, so any silent edit to a
historical entry invalidates every entry after it.

Structure:
    {ts, kind, payload, prev_hash, hash}

`hash = sha256(prev_hash || canonical_json({ts, kind, payload}))`

This is a primitive Merkle list — not cryptographically signed (no
private key on disk), but enough to detect tampering by any non-
malicious-with-root-access actor. Combined with the SQLite trade
journal it gives a complete audit trail.

Verification: `verify_chain(path)` walks the file end-to-end and
returns the first index where the hash chain broke, or None if intact.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


GENESIS_HASH = "0" * 64
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "engine" / "logs" / "audit.jsonl"


@dataclass
class AuditEntry:
    ts: str
    kind: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


_lock = threading.Lock()
_last_hash_cache: dict[str, str] = {}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _compute_hash(prev: str, ts: str, kind: str, payload: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev.encode("utf-8"))
    h.update(_canonical({"ts": ts, "kind": kind, "payload": payload}).encode("utf-8"))
    return h.hexdigest()


def _read_last_hash(path: Path) -> str:
    """Scan the file's last line for the previous hash. Reasonably fast
    because we only seek the tail block."""
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, 8192)
            fh.seek(-block, os.SEEK_END)
            tail = fh.read(block).decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return GENESIS_HASH
        entry = json.loads(lines[-1])
        return str(entry.get("hash", GENESIS_HASH))
    except Exception:  # noqa: BLE001
        return GENESIS_HASH


def append_entry(
    kind: str,
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    ts: str | None = None,
) -> AuditEntry:
    """Append a single hashed entry. Thread-safe across process workers."""
    p = path or DEFAULT_LOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    ts_iso = ts or datetime.now(timezone.utc).isoformat()
    with _lock:
        prev = _last_hash_cache.get(str(p))
        if prev is None:
            prev = _read_last_hash(p)
        h = _compute_hash(prev, ts_iso, kind, payload)
        entry = AuditEntry(ts=ts_iso, kind=kind, payload=dict(payload),
                           prev_hash=prev, hash=h)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(_canonical({
                "ts": entry.ts, "kind": entry.kind, "payload": entry.payload,
                "prev_hash": entry.prev_hash, "hash": entry.hash,
            }) + "\n")
        _last_hash_cache[str(p)] = h
    return entry


def read_chain(path: Path | None = None) -> Iterator[AuditEntry]:
    p = path or DEFAULT_LOG_PATH
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield AuditEntry(
                    ts=str(obj["ts"]), kind=str(obj["kind"]),
                    payload=dict(obj.get("payload") or {}),
                    prev_hash=str(obj.get("prev_hash", GENESIS_HASH)),
                    hash=str(obj.get("hash", "")),
                )
            except Exception:  # noqa: BLE001
                continue


def verify_chain(path: Path | None = None) -> int | None:
    """Walk the chain; return the index of the first broken entry, or None.

    A broken entry is one whose recomputed hash doesn't match its stored
    hash, OR whose `prev_hash` doesn't match the predecessor's hash.
    """
    prev = GENESIS_HASH
    for i, entry in enumerate(read_chain(path)):
        if entry.prev_hash != prev:
            return i
        recomputed = _compute_hash(prev, entry.ts, entry.kind, entry.payload)
        if recomputed != entry.hash:
            return i
        prev = entry.hash
    return None


def reset_cache_for_tests() -> None:
    """Test helper — drop the cached last-hash so a new file starts at GENESIS."""
    with _lock:
        _last_hash_cache.clear()
