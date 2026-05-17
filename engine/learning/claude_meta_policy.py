"""Claude-trained meta-policy.

Anthropic's Claude API is the engine's cheapest continuous-learning
resource. It cannot do gradient descent, but it *can* do symbolic
reasoning over labeled examples — pattern matching that supplements
numeric optimization where data alone is too sparse to retrain on.

Algorithm:
  Every 60 minutes (if ≥50 trades have closed in that window):
    1. Collect the N worst losers.
    2. Send Claude a structured prompt asking which parameter deltas
       would have rejected those trades while preserving recent winners.
    3. Claude returns JSON: [{"param": str, "delta": ..., "rationale": str}, ...].
    4. Validate each delta against `ALLOWED_PARAMS` (anti-injection /
       anti-runaway). Unknown or out-of-range deltas are dropped.
    5. Persist each accepted override to `claude_overrides` and apply
       it as a soft override at the engine settings layer until the
       next iteration overrides it or a kill-switch resets all.

Containment principles:
  * Only whitelisted parameters can be tuned (`ALLOWED_PARAMS`).
  * Every override is reversible — `reset_overrides()` wipes all.
  * Shadow mode prevents real money exposure during exploration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

from engine.config import settings
from engine.data.sqlite_journal import open_journal


# (min, max) clamps per allowed parameter. Anything outside this range is
# silently truncated. Anything not in this map is dropped entirely.
ALLOWED_PARAMS: dict[str, tuple[Any, Any] | tuple[type, ...]] = {
    "consensus.min_agree":               (3, 6),
    "consensus.po3_bonus_enabled":       (bool,),
    "claude_gate.min_confidence_normal": (50, 80),
    "claude_gate.min_confidence_fallback": (60, 90),
    "risk.smc_filter_required":          (bool,),
    "risk.killzone_strict_outside_overlap": (bool,),
    "spread.acceptable_multiplier":      (1.5, 3.0),
}
DEFAULT_OVERRIDE_TTL_HOURS = 24


@dataclass(frozen=True)
class Override:
    param: str
    new_value: Any
    rationale: str
    expires_at: datetime


def _coerce_value(param: str, raw: Any) -> Any | None:
    """Apply the ALLOWED_PARAMS clamp. Returns None if the value is invalid."""
    spec = ALLOWED_PARAMS.get(param)
    if spec is None:
        return None
    if isinstance(spec, tuple) and len(spec) == 1 and isinstance(spec[0], type):
        cls = spec[0]
        if cls is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                if raw.lower() in {"true", "1", "yes"}:  return True
                if raw.lower() in {"false", "0", "no"}:  return False
            return None
        try:
            return cls(raw)
        except (TypeError, ValueError):
            return None
    lo, hi = spec
    try:
        v = type(lo)(raw)
    except (TypeError, ValueError):
        return None
    if v < lo or v > hi:
        return max(lo, min(hi, v))
    return v


def collect_recent_losers(
    *,
    n: int = 10,
    lookback_hours: int = 24,
    db_path: str | None = None,
) -> list[dict]:
    """Return the N most-recent losing closed trades as a structured dossier."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    out: list[dict] = []
    with open_journal(db_path) as con:
        rows = con.execute(
            """
            SELECT id, symbol, direction, entry_price, exit_price, pnl,
                   signal_confluence, claude_confidence, claude_reasoning,
                   close_reason, close_time
              FROM trades
             WHERE close_time >= ? AND pnl < 0
             ORDER BY pnl ASC
             LIMIT ?
            """,
            (cutoff, int(n)),
        ).fetchall()
        for r in rows:
            out.append({
                "trade_id": r["id"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "entry": r["entry_price"],
                "exit": r["exit_price"],
                "pnl": r["pnl"],
                "confluence": r["signal_confluence"],
                "claude_confidence": r["claude_confidence"],
                "claude_reasoning": r["claude_reasoning"],
                "close_reason": r["close_reason"],
                "close_time": r["close_time"],
            })
    return out


def _build_prompt(losers: list[dict], recent_winners: list[dict]) -> dict:
    return {
        "task": "propose_parameter_overrides",
        "losers": losers,
        "recent_winners_sample": recent_winners[:10],
        "allowed_params": {
            k: ("bool" if (isinstance(v, tuple) and len(v) == 1 and v[0] is bool) else list(v))
            for k, v in ALLOWED_PARAMS.items()
        },
        "instructions": (
            "Examine the losing trades. Suggest up to 3 minimal parameter "
            "deltas (each strictly within `allowed_params` clamps) that "
            "would have rejected most losers while preserving most winners. "
            "Respond as a strict JSON array of "
            '{"param": str, "new_value": value, "rationale": str}.'
        ),
    }


def _parse_claude_response(text: str) -> list[dict]:
    """Lenient JSON extraction — accept fenced/partial output."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict) and "overrides" in parsed:
        parsed = parsed["overrides"]
    if not isinstance(parsed, list):
        return []
    return parsed


def propose_overrides(
    losers: list[dict],
    recent_winners: list[dict] | None = None,
    *,
    claude_caller: Callable[[dict], str] | None = None,
    ttl_hours: int = DEFAULT_OVERRIDE_TTL_HOURS,
) -> list[Override]:
    """Send the dossier to Claude (or `claude_caller` mock) and parse the response.

    Returns a list of validated, clamped Override objects.
    """
    payload = _build_prompt(losers, recent_winners or [])
    if claude_caller is None:
        try:
            from engine.strategy.claude_gate import call_claude_raw
            text = call_claude_raw(json.dumps(payload, separators=(",", ":")))
        except Exception as e:  # noqa: BLE001
            logger.warning("Claude meta-policy: failed to call Claude — {}", e)
            return []
    else:
        text = claude_caller(payload)

    parsed = _parse_claude_response(text)
    expiry = datetime.now(timezone.utc) + timedelta(hours=int(ttl_hours))
    out: list[Override] = []
    for item in parsed[:3]:
        if not isinstance(item, dict):
            continue
        param = str(item.get("param", "")).strip()
        if param not in ALLOWED_PARAMS:
            continue
        new_value = _coerce_value(param, item.get("new_value"))
        if new_value is None:
            continue
        rationale = str(item.get("rationale", ""))[:600]
        out.append(Override(param=param, new_value=new_value, rationale=rationale, expires_at=expiry))
    return out


def apply_overrides(
    overrides: list[Override],
    *,
    db_path: str | None = None,
) -> int:
    """Persist accepted overrides to claude_overrides. Returns count applied."""
    if not overrides:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    with open_journal(db_path) as con:
        for o in overrides:
            old_row = con.execute(
                "SELECT new_value FROM claude_overrides "
                "WHERE param=? AND active=1 ORDER BY id DESC LIMIT 1",
                (o.param,),
            ).fetchone()
            old_value = old_row["new_value"] if old_row else None
            con.execute(
                "UPDATE claude_overrides SET active=0 WHERE param=? AND active=1",
                (o.param,),
            )
            con.execute(
                "INSERT INTO claude_overrides "
                "(ts, param, old_value, new_value, rationale, expires_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (now_iso, o.param, old_value, json.dumps(o.new_value),
                 o.rationale, o.expires_at.isoformat()),
            )
        con.commit()
    return len(overrides)


def active_overrides(*, db_path: str | None = None) -> dict[str, Any]:
    """Return the currently active param → value map (for runtime injection)."""
    out: dict[str, Any] = {}
    now_iso = datetime.now(timezone.utc).isoformat()
    with open_journal(db_path) as con:
        rows = con.execute(
            "SELECT param, new_value FROM claude_overrides "
            "WHERE active=1 AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY id DESC",
            (now_iso,),
        ).fetchall()
        for r in rows:
            try:
                out[r["param"]] = json.loads(r["new_value"])
            except (TypeError, json.JSONDecodeError):
                continue
    return out


def reset_overrides(*, db_path: str | None = None) -> int:
    """Disable all active overrides. Returns count."""
    with open_journal(db_path) as con:
        cur = con.execute("UPDATE claude_overrides SET active=0 WHERE active=1")
        con.commit()
        return cur.rowcount or 0
