"""Claude API spend tracker + soft/hard budget enforcement.

Every call to `claude_gate.decide()` and `claude_meta_policy.propose_*`
goes through this module first. We track input + output tokens per call,
compute a USD cost using the published Sonnet 4.6 pricing, accumulate
into `settings_kv` keyed by UTC date, and enforce two thresholds:

  * `SOFT_BUDGET_USD` — emit a NOTIFY at 80% utilization
  * `HARD_BUDGET_USD` — return SKIP on the next decision (consensus
                       treats this as `REJECTED_CLAUDE_BUDGET`)

The pricing table is centralised so a single edit covers a model
upgrade.

Stored counters per day:
    claude_cost:YYYY-MM-DD:input_tokens
    claude_cost:YYYY-MM-DD:output_tokens
    claude_cost:YYYY-MM-DD:calls
    claude_cost:YYYY-MM-DD:cost_usd
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from loguru import logger

from engine.data.sqlite_journal import open_journal


# USD per 1M tokens. Updated 2026-05.
PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # (input, output)
    "claude-sonnet-4-6":   (3.00, 15.00),
    "claude-opus-4-7":     (15.00, 75.00),
    "claude-haiku-4-5":    (1.00,  5.00),
    "claude-sonnet-4-7":   (3.00, 15.00),  # forward-compat default
}

SOFT_BUDGET_USD = 5.00
HARD_BUDGET_USD = 10.00


@dataclass(frozen=True)
class CostRecord:
    date_str: str
    input_tokens: int
    output_tokens: int
    calls: int
    cost_usd: float


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _key(date_str: str, field: str) -> str:
    return f"claude_cost:{date_str}:{field}"


def estimate_cost(input_tokens: int, output_tokens: int, *, model: str) -> float:
    pricing = PRICING_PER_M_TOKENS.get(model)
    if pricing is None:
        # Conservative fallback — use Opus pricing on unknown models.
        pricing = PRICING_PER_M_TOKENS["claude-opus-4-7"]
    inp, out = pricing
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def record_call(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    db_path: str | None = None,
    date_str: str | None = None,
) -> CostRecord:
    """Bump today's counters; return the updated record."""
    date_str = date_str or _today_iso()
    cost = estimate_cost(input_tokens, output_tokens, model=model)
    with open_journal(db_path) as con:
        for field, delta in (
            ("input_tokens",  int(input_tokens)),
            ("output_tokens", int(output_tokens)),
            ("calls",         1),
        ):
            k = _key(date_str, field)
            row = con.execute("SELECT v FROM settings_kv WHERE k=?", (k,)).fetchone()
            cur = int(row["v"]) if row else 0
            con.execute(
                "INSERT INTO settings_kv(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (k, str(cur + delta)),
            )
        cost_key = _key(date_str, "cost_usd")
        row = con.execute("SELECT v FROM settings_kv WHERE k=?", (cost_key,)).fetchone()
        running = float(row["v"]) if row else 0.0
        new_cost = running + cost
        con.execute(
            "INSERT INTO settings_kv(k,v) VALUES(?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (cost_key, f"{new_cost:.6f}"),
        )
        con.commit()
        # Re-read everything for the returned record.
        rec = read_today(date_str=date_str, db_path=db_path)
    if rec.cost_usd >= HARD_BUDGET_USD:
        logger.error("Claude HARD budget exceeded today: ${:.2f} >= ${:.2f}",
                     rec.cost_usd, HARD_BUDGET_USD)
    elif rec.cost_usd >= SOFT_BUDGET_USD:
        logger.warning("Claude SOFT budget reached today: ${:.2f} >= ${:.2f}",
                       rec.cost_usd, SOFT_BUDGET_USD)
    return rec


def read_today(*, date_str: str | None = None, db_path: str | None = None) -> CostRecord:
    date_str = date_str or _today_iso()
    with open_journal(db_path) as con:
        def _g(field: str) -> str | None:
            row = con.execute(
                "SELECT v FROM settings_kv WHERE k=?", (_key(date_str, field),),
            ).fetchone()
            return row["v"] if row else None
        return CostRecord(
            date_str=date_str,
            input_tokens=int(_g("input_tokens") or 0),
            output_tokens=int(_g("output_tokens") or 0),
            calls=int(_g("calls") or 0),
            cost_usd=float(_g("cost_usd") or 0.0),
        )


def is_over_budget(*, hard: bool = True, db_path: str | None = None) -> bool:
    rec = read_today(db_path=db_path)
    limit = HARD_BUDGET_USD if hard else SOFT_BUDGET_USD
    return rec.cost_usd >= limit


def reset_today(*, date_str: str | None = None, db_path: str | None = None) -> None:
    """Wipe today's counters — operator-controlled emergency override."""
    date_str = date_str or _today_iso()
    with open_journal(db_path) as con:
        for field in ("input_tokens", "output_tokens", "calls", "cost_usd"):
            con.execute("DELETE FROM settings_kv WHERE k=?", (_key(date_str, field),))
        con.commit()


def daily_history(days: int = 7, *, db_path: str | None = None) -> list[CostRecord]:
    """Return the last `days` daily records, most recent first."""
    out: list[CostRecord] = []
    today = datetime.now(timezone.utc).date()
    from datetime import timedelta
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        out.append(read_today(date_str=d, db_path=db_path))
    return out
