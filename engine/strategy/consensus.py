"""5-source consensus engine — Appendix G decision tree (verbatim).

The engine is a pure function of injected source predictions, so it can be
unit-tested without touching MT5, the models, or Anthropic. The optional
`claude_gate` callable is only invoked when consensus passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from engine.config import settings

Vote = Literal["BUY", "SELL", "HOLD"]
Decision = Literal["BUY", "SELL", "SKIP"]


@dataclass(frozen=True)
class CnnVote:
    label: Vote
    confidence: int   # 0–100


@dataclass(frozen=True)
class Sources:
    smc: Vote
    cnn: CnnVote | None        # None ⇒ model failed; fallback path engages
    rl: Vote | None            # None ⇒ agent failed; treated as HOLD
    killzone_ok: bool
    news_clear: bool
    ofi: Vote = "HOLD"         # Order Flow Imbalance vote (6th source)
    candle: Vote = "HOLD"      # TA-Lib candlestick pattern vote (7th source)
    po3: Vote = "HOLD"         # Power-of-Three sweep+reclaim signal (bonus, +1 to confluence if it agrees)


@dataclass
class ClaudeResponse:
    decision: Decision
    confidence: int
    reasoning: str
    risk_adjustment: float
    ok: bool = True
    error: str | None = None


@dataclass
class ConsensusResult:
    outcome: str                       # EXECUTED | REJECTED_<reason>
    direction: Vote = "HOLD"
    confluence: int = 0                # 0–5 agreeing-source count
    risk_pct: float = settings.RISK_PCT_PER_TRADE
    fallback: bool = False
    claude: ClaudeResponse | None = None
    reason: str | None = None
    detail: dict = field(default_factory=dict)


# Order of preconditions matters: short-circuit on first failure.
PRECONDITION_REJECTIONS = (
    "REJECTED_PAUSED",
    "REJECTED_KILL_ACTIVE",
    "REJECTED_SPREAD_WIDENED",
    "REJECTED_MAX_POSITIONS",
    "REJECTED_CORRELATION",
    "REJECTED_NEWS",
    "REJECTED_KILL_ZONE",
    "REJECTED_VPIN_TOXIC",
)


@dataclass(frozen=True)
class State:
    is_paused: bool = False
    kill_active: bool = False
    spread_widened: bool = False
    open_positions: int = 0
    correlated_open: int = 0
    vpin_toxic: bool = False           # True ⇒ order flow toxic (VPIN > threshold)


def _direction_from(votes: list[Vote]) -> Vote:
    """Majority among the non-HOLD votes; HOLD if tied or all HOLD."""
    counts = {"BUY": votes.count("BUY"), "SELL": votes.count("SELL")}
    if counts["BUY"] == counts["SELL"]:
        return "HOLD"
    return "BUY" if counts["BUY"] > counts["SELL"] else "SELL"


def evaluate(
    state: State,
    sources: Sources,
    *,
    claude_gate: Callable[[dict], ClaudeResponse] | None = None,
    claude_context: dict | None = None,
    enable_claude: bool = True,
    min_agree: int = settings.CONSENSUS_MIN_AGREE,
) -> ConsensusResult:
    # 1–7: short-circuit preconditions
    if state.is_paused:                              return ConsensusResult("REJECTED_PAUSED", reason="engine paused")
    if state.kill_active:                            return ConsensusResult("REJECTED_KILL_ACTIVE", reason="kill switch active")
    if state.spread_widened:                         return ConsensusResult("REJECTED_SPREAD_WIDENED", reason="spread > baseline×3")
    if state.open_positions >= settings.MAX_CONCURRENT_POSITIONS:
        return ConsensusResult("REJECTED_MAX_POSITIONS", reason=f"{state.open_positions} ≥ {settings.MAX_CONCURRENT_POSITIONS}")
    if state.correlated_open >= settings.MAX_CORRELATED_POSITIONS:
        return ConsensusResult("REJECTED_CORRELATION", reason=f"{state.correlated_open} correlated open")
    if not sources.news_clear:                       return ConsensusResult("REJECTED_NEWS", reason="news within window")
    if not sources.killzone_ok:                      return ConsensusResult("REJECTED_KILL_ZONE", reason="outside ICT kill zone")
    if state.vpin_toxic:                             return ConsensusResult("REJECTED_VPIN_TOXIC", reason="VPIN flow toxic")

    # 8: votes + direction + consensus count (7-vote model: smc, cnn, rl, ofi, candle + killzone + news)
    fallback = sources.cnn is None
    cnn_vote: Vote = "HOLD" if sources.cnn is None else sources.cnn.label
    rl_vote: Vote = "HOLD" if sources.rl is None else sources.rl
    directional = [sources.smc, cnn_vote, rl_vote, sources.ofi, sources.candle]
    direction = _direction_from(directional)
    if direction == "HOLD":
        return ConsensusResult(
            "REJECTED_NO_DIRECTION",
            reason="no majority among SMC/CNN/RL/OFI/CANDLE",
            detail={"directional_votes": directional},
        )
    confluence = sum(1 for v in directional if v == direction) + 2  # killzone_ok + news_clear (both required true above)
    if sources.po3 == direction:
        confluence += 1  # PO3 sweep+reclaim agreeing with direction is a bonus

    # 9: minimum agreement
    if confluence < min_agree:
        return ConsensusResult(
            "REJECTED_CONSENSUS",
            direction=direction,
            confluence=confluence,
            reason=f"only {confluence}/5 agree",
            detail={"directional_votes": directional},
        )

    # Fallback path (CNN failed): require Claude conf ≥70 + reduced risk.
    if fallback:
        if sources.smc == "HOLD":
            return ConsensusResult(
                "REJECTED_FALLBACK_NO_SMC",
                direction=direction, confluence=confluence, fallback=True,
                reason="CNN unavailable and SMC=HOLD",
            )

    # 10–13: Claude final filter
    if not enable_claude or claude_gate is None:
        return ConsensusResult(
            "EXECUTED",
            direction=direction, confluence=confluence,
            risk_pct=settings.FALLBACK_RISK_PCT if fallback else settings.RISK_PCT_PER_TRADE,
            fallback=fallback,
        )

    ctx = claude_context or {}
    try:
        c = claude_gate(ctx)
    except Exception as e:  # noqa: BLE001
        c = ClaudeResponse("SKIP", 0, f"gate raised: {e}", 1.0, ok=False, error=str(e))

    if not c.ok:
        return ConsensusResult(
            "REJECTED_CLAUDE_UNAVAILABLE",
            direction=direction, confluence=confluence, fallback=fallback,
            claude=c, reason=c.error or "claude error",
        )
    if c.decision == "SKIP":
        return ConsensusResult(
            "REJECTED_CLAUDE_SKIP",
            direction=direction, confluence=confluence, fallback=fallback,
            claude=c, reason="claude skip",
        )
    if c.decision != direction:
        return ConsensusResult(
            "REJECTED_CLAUDE_DISAGREE",
            direction=direction, confluence=confluence, fallback=fallback,
            claude=c, reason=f"claude={c.decision} ≠ direction={direction}",
        )
    if c.confidence < (70 if fallback else 50):
        return ConsensusResult(
            "REJECTED_CLAUDE_LOW_CONFIDENCE",
            direction=direction, confluence=confluence, fallback=fallback,
            claude=c, reason=f"confidence {c.confidence} below threshold",
        )

    # 14–16: clamp + execute
    K = max(0.5, min(1.5, c.risk_adjustment))
    base = settings.FALLBACK_RISK_PCT if fallback else settings.RISK_PCT_PER_TRADE
    return ConsensusResult(
        "EXECUTED",
        direction=direction, confluence=confluence,
        risk_pct=base * K, fallback=fallback, claude=c,
        detail={"K": K},
    )
