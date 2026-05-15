"""Anthropic Claude API gate — the final reasoning filter on every signal.

Returns a `ClaudeResponse` populated from a strict-JSON Claude reply. Retries
with exponential backoff on transient errors. The gate is intentionally tiny
and dependency-free outside the Anthropic SDK so it's easy to mock in tests.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from anthropic import Anthropic, APIError, APIStatusError
from loguru import logger

from engine.config import settings
from engine.strategy.consensus import ClaudeResponse

SYSTEM_PROMPT = (
    "You are the final decision gate for a professional forex trading engine. "
    "You receive a structured signal context JSON and must output ONLY valid JSON.\n\n"
    "Decision criteria — apply IN ORDER, first match wins:\n"
    "1. HARD REJECT if volatility_regime == EXTREME → SKIP, confidence 0, reasoning 'Extreme volatility — capital preservation', risk_adjustment 0.0\n"
    "2. HARD REJECT if spread_multiplier > 2.5 → SKIP with spread reasoning\n"
    "3. HARD REJECT if news_spike_detected == true → SKIP\n"
    "4. HARD REJECT if hours_to_next_news < 0.5 → SKIP (30 min pre-news window)\n"
    "5. HARD REJECT if ob_freshness == CONSUMED → SKIP\n"
    "6. HARD REJECT if h4_bias contradicts direction AND adx_h4 > 25 → SKIP (strong counter-trend)\n\n"
    "BOOST confidence by +15 if ALL of: silver_bullet_active AND fvg_ob_confluence AND ote_zone_active\n"
    "BOOST risk_adjustment to 1.2 if overlap_active AND confluence_score >= 4\n\n"
    "Output ONLY this JSON, no prose, no markdown, no code fences:\n"
    '{ "decision": "BUY"|"SELL"|"SKIP", '
    '"confidence": 0-100, '
    '"reasoning": "2-3 sentence institutional analysis ≤ 600 chars", '
    '"risk_adjustment": 0.5-1.5 }'
)


def hard_reject_check(ctx: dict[str, Any]) -> tuple[bool, str]:
    if ctx.get("volatility_regime") == "EXTREME":
        return True, "Extreme volatility — capital preservation"
    if float(ctx.get("spread_multiplier", 1.0)) > 2.5:
        return True, f"Spread {ctx.get('spread_multiplier'):.2f}× normal — rejecting"
    if bool(ctx.get("news_spike_detected", False)):
        return True, "Live news spike detected — sit out"
    hours = ctx.get("hours_to_next_news")
    if hours is not None and float(hours) < 0.5:
        return True, f"High-impact news in {hours*60:.0f}min — pre-news pause"
    if ctx.get("ob_freshness") == "CONSUMED":
        return True, "Order Block already consumed (2+ taps)"
    h4_bias = ctx.get("h4_bias")
    direction = ctx.get("direction")
    adx_h4 = float(ctx.get("adx_h4", 0))
    if direction == "BUY" and h4_bias == "BEARISH" and adx_h4 > 25:
        return True, f"H4 bearish trend (ADX {adx_h4:.0f}) — counter-trend BUY rejected"
    if direction == "SELL" and h4_bias == "BULLISH" and adx_h4 > 25:
        return True, f"H4 bullish trend (ADX {adx_h4:.0f}) — counter-trend SELL rejected"
    return False, ""


def build_rich_context(
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    confluence_score: int,
    premium_discount: str = "EQUILIBRIUM",
    ote_zone_active: bool = False,
    ob_freshness: str = "UNKNOWN",
    fvg_ob_confluence: bool = False,
    h4_bias: str = "RANGING",
    session: str = "DEAD",
    silver_bullet_active: bool = False,
    overlap_active: bool = False,
    spread_multiplier: float = 1.0,
    volatility_regime: str = "NORMAL",
    atr14_pct: float = 0.0,
    adx_h4: float = 0.0,
    last_5_trades_result: list[str] | None = None,
    account_drawdown_pct: float = 0.0,
    news_spike_detected: bool = False,
    hours_to_next_news: float = 999.0,
    ob_distance_pips: float = 0.0,
    liquidity_above_pips: float = 0.0,
    liquidity_below_pips: float = 0.0,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "confluence_score": confluence_score,
        "premium_discount": premium_discount,
        "ote_zone_active": ote_zone_active,
        "ob_freshness": ob_freshness,
        "fvg_ob_confluence": fvg_ob_confluence,
        "h4_bias": h4_bias,
        "session": session,
        "silver_bullet_active": silver_bullet_active,
        "overlap_active": overlap_active,
        "spread_multiplier": float(spread_multiplier),
        "volatility_regime": volatility_regime,
        "atr14_pct": float(atr14_pct),
        "adx_h4": float(adx_h4),
        "last_5_trades_result": last_5_trades_result or [],
        "account_drawdown_pct": float(account_drawdown_pct),
        "news_spike_detected": news_spike_detected,
        "hours_to_next_news": float(hours_to_next_news),
        "ob_distance_pips": float(ob_distance_pips),
        "liquidity_above_pips": float(liquidity_above_pips),
        "liquidity_below_pips": float(liquidity_below_pips),
    }


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Be lenient: accept fenced or partial responses, extract the first {…}."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


def _validate(payload: dict[str, Any]) -> ClaudeResponse:
    decision = str(payload["decision"]).upper()
    if decision not in ("BUY", "SELL", "SKIP"):
        raise ValueError(f"invalid decision: {decision!r}")
    conf = int(payload["confidence"])
    if not 0 <= conf <= 100:
        raise ValueError(f"confidence out of range: {conf}")
    reasoning = str(payload.get("reasoning", "")).strip()[:600]
    risk_adj = float(payload.get("risk_adjustment", 1.0))
    risk_adj = max(0.5, min(1.5, risk_adj))
    return ClaudeResponse(
        decision=decision, confidence=conf,
        reasoning=reasoning, risk_adjustment=risk_adj, ok=True,
    )


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not settings.have_anthropic():
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        _client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def decide(context: dict, *, model: str | None = None) -> ClaudeResponse:
    """Call Claude with the trade context, returning a validated ClaudeResponse.

    Persistent failure raises after `CLAUDE_RETRY_MAX` attempts; callers in
    consensus.py wrap the call in try/except and treat exceptions as
    `claude_unavailable`.
    """
    client = _get_client()
    user_payload = json.dumps(context, default=str, separators=(",", ":"))
    last_err: Exception | None = None
    for attempt in range(1, settings.CLAUDE_RETRY_MAX + 1):
        try:
            resp = client.messages.create(
                model=model or settings.CLAUDE_MODEL,
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_payload}],
                timeout=settings.CLAUDE_TIMEOUT_S,
            )
            text = "".join(
                getattr(block, "text", "") for block in resp.content
                if getattr(block, "type", "") == "text"
            )
            payload = _extract_json(text)
            return _validate(payload)
        except (APIError, APIStatusError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            backoff = min(8.0, 2.0 ** (attempt - 1))
            logger.warning("Claude call failed (attempt {}/{}): {}", attempt, settings.CLAUDE_RETRY_MAX, e)
            if attempt < settings.CLAUDE_RETRY_MAX:
                time.sleep(backoff)
    raise RuntimeError(f"Claude gate exhausted retries: {last_err}")


def gate_factory():
    """Return a `claude_gate` callable suitable for `consensus.evaluate(...)`."""
    def _call(context: dict) -> ClaudeResponse:
        return decide(context)
    return _call
