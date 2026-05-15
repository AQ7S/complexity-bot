"""Discord webhook embed builders + dispatcher.

One builder per event class from Appendix H. Each builder returns the full
JSON-serializable webhook body (`{username, embeds:[…]}`); `post()` ships
it to the configured webhook URL via httpx with a short timeout. Network
failures are logged and swallowed — notifications must never break trading.

Toggle via `NOTIFY_DISCORD_ENABLED` in the env. Errors route to the
secondary `DISCORD_ERROR_WEBHOOK_URL` if configured, else to the primary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from engine.config import settings

USERNAME = "Complexity Engine"
USERNAME_ERR = "Complexity Engine [ERROR]"
FOOTER = {"text": "Complexity Engine"}

COLOR_GREEN  = 3066993
COLOR_RED    = 15158332
COLOR_BLUE   = 3447003
COLOR_ORANGE = 15105570
COLOR_PURPLE = 10181046
COLOR_GOLD   = 15844367

POST_TIMEOUT_S = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _embed(title: str, color: int, fields: list[dict[str, Any]],
           *, description: str | None = None, footer: dict[str, str] | None = None,
           ts: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {
        "title": title, "color": color, "fields": fields,
        "footer": footer or FOOTER, "timestamp": ts or _now_iso(),
    }
    if description:
        e["description"] = description
    return e


# ---------------------------------------------------------------------------
# Embed builders — exact field shapes match Appendix H
# ---------------------------------------------------------------------------

def trade_opened(*, symbol: str, direction: str, entry: float, sl: float, tp: float,
                 lot: float, risk_pct: float, confluence: int,
                 claude_confidence: int | None, claude_note: str | None) -> dict:
    fields = [
        {"name": "Entry",       "value": f"{entry:g}",       "inline": True},
        {"name": "SL",          "value": f"{sl:g}",          "inline": True},
        {"name": "TP",          "value": f"{tp:g}",          "inline": True},
        {"name": "Lot",         "value": f"{lot:g}",         "inline": True},
        {"name": "Risk %",      "value": f"{risk_pct*100:.1f}%", "inline": True},
        {"name": "Confluence",  "value": f"{confluence}/5",  "inline": True},
    ]
    if claude_confidence is not None:
        fields.append({"name": "Claude Conf", "value": f"{claude_confidence}%", "inline": True})
    if claude_note:
        fields.append({"name": "Claude Note", "value": claude_note[:1000], "inline": False})
    return {
        "username": USERNAME,
        "embeds": [_embed(f"Trade Opened: {symbol} {direction}", COLOR_BLUE, fields,
                          description="Entry confirmed via consensus + Claude approval.")],
    }


def trade_closed(*, symbol: str, entry: float, exit_: float, pnl_usd: float,
                 rr_achieved: float | None, duration_s: float,
                 close_reason: str) -> dict:
    profit = pnl_usd >= 0
    title = f"{'Profit' if profit else 'Loss'} Closed: {symbol}"
    color = COLOR_GREEN if profit else COLOR_RED
    pnl_str = f"+${pnl_usd:.2f}" if profit else f"-${abs(pnl_usd):.2f}"
    mins, secs = divmod(int(duration_s), 60)
    fields = [
        {"name": "Entry",        "value": f"{entry:g}",                  "inline": True},
        {"name": "Exit",         "value": f"{exit_:g}",                  "inline": True},
        {"name": "P&L (USD)",    "value": pnl_str,                       "inline": True},
        {"name": "R:R Achieved", "value": f"{rr_achieved:.2f}" if rr_achieved is not None else "-", "inline": True},
        {"name": "Duration",     "value": f"{mins}m {secs}s",            "inline": True},
        {"name": "Close Reason", "value": close_reason,                  "inline": True},
    ]
    return {"username": USERNAME, "embeds": [_embed(title, color, fields)]}


def signal_detected(*, symbol: str, direction: str, confluence: int,
                    smc_zone: str, cnn_conf: int, rl_vote: str,
                    kill_zone_label: str, news_clear: bool,
                    claude_excerpt: str | None) -> dict:
    fields = [
        {"name": "Confluence", "value": f"{confluence}/5",             "inline": True},
        {"name": "SMC Zone",   "value": smc_zone,                      "inline": True},
        {"name": "CNN Conf",   "value": f"{cnn_conf}%",                "inline": True},
        {"name": "RL Vote",    "value": rl_vote,                       "inline": True},
        {"name": "Kill Zone",  "value": kill_zone_label,               "inline": True},
        {"name": "News Clear", "value": "Yes" if news_clear else "No", "inline": True},
    ]
    if claude_excerpt:
        fields.append({"name": "Claude Excerpt", "value": claude_excerpt[:1000], "inline": False})
    return {"username": USERNAME, "embeds": [
        _embed(f"Signal Detected: {symbol} {direction}", COLOR_PURPLE, fields)
    ]}


def kill_triggered(*, kind: str, drawdown_pct: float,
                   positions_closed: int, halted_until: str | None) -> dict:
    fields = [
        {"name": "Drawdown %",       "value": f"{drawdown_pct*100:.2f}%",   "inline": True},
        {"name": "Positions Closed", "value": str(positions_closed),         "inline": True},
        {"name": "Halted Until",     "value": halted_until or "-",           "inline": True},
    ]
    return {"username": USERNAME, "embeds": [
        _embed(f"KILL TRIGGERED — {kind}", COLOR_RED, fields,
               description="Account drawdown threshold breached. All positions closed.")
    ]}


def news_warning(*, event_name: str, currency: str, impact: str,
                 minutes_until: int, affected_symbols: list[str]) -> dict:
    fields = [
        {"name": "Currency",         "value": currency,                  "inline": True},
        {"name": "Impact",           "value": impact,                    "inline": True},
        {"name": "Time Until",       "value": f"{minutes_until} minutes","inline": True},
        {"name": "Affected Symbols", "value": ", ".join(affected_symbols) or "-", "inline": False},
        {"name": "Action",           "value": f"Affected positions will be closed in ~{minutes_until} min.", "inline": False},
    ]
    return {"username": USERNAME, "embeds": [
        _embed(f"News Warning: {event_name}", COLOR_ORANGE, fields)
    ]}


def engine_error(*, component: str, error_type: str,
                 stack_excerpt: str, action_taken: str) -> dict:
    fields = [
        {"name": "Component",    "value": component,    "inline": True},
        {"name": "Error Type",   "value": error_type,   "inline": True},
        {"name": "Action Taken", "value": action_taken, "inline": True},
        {"name": "Stack Excerpt","value": f"```py\n{stack_excerpt[:900]}\n```", "inline": False},
    ]
    return {"username": USERNAME_ERR, "embeds": [_embed("Engine Error", COLOR_RED, fields)]}


def training_complete(*, model_name: str, version: str,
                      accuracy_delta: float, loss_delta: float,
                      trades_trained_on: int) -> dict:
    fields = [
        {"name": "New Version",       "value": version,                       "inline": True},
        {"name": "Accuracy Δ",        "value": f"{accuracy_delta:+.4f}",      "inline": True},
        {"name": "Loss Δ",            "value": f"{loss_delta:+.4f}",          "inline": True},
        {"name": "Trades Trained On", "value": str(trades_trained_on),        "inline": True},
    ]
    return {"username": USERNAME, "embeds": [
        _embed(f"Model Retrained: {model_name}", COLOR_GOLD, fields)
    ]}


def daily_summary(*, date_str: str, trades: int, wins: int, losses: int,
                  net_pnl: float, equity: float,
                  best_trade: str, worst_trade: str,
                  drawdown_max_pct: float) -> dict:
    win_rate = (wins / trades * 100) if trades else 0.0
    fields = [
        {"name": "Trades",       "value": str(trades),                "inline": True},
        {"name": "Wins",         "value": str(wins),                  "inline": True},
        {"name": "Losses",       "value": str(losses),                "inline": True},
        {"name": "Win Rate",     "value": f"{win_rate:.1f}%",         "inline": True},
        {"name": "Net P&L",      "value": f"${net_pnl:+,.2f}",        "inline": True},
        {"name": "Equity",       "value": f"${equity:,.2f}",          "inline": True},
        {"name": "Best Trade",   "value": best_trade,                 "inline": True},
        {"name": "Worst Trade",  "value": worst_trade,                "inline": True},
        {"name": "Drawdown Max", "value": f"{drawdown_max_pct*100:.2f}%", "inline": True},
    ]
    return {"username": USERNAME, "embeds": [
        _embed(f"Daily Summary — {date_str}", COLOR_GOLD, fields)
    ]}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def post(payload: dict, *, error_channel: bool = False, url: str | None = None) -> bool:
    """Send a webhook payload. Returns True on 2xx. Honors NOTIFY_DISCORD_ENABLED."""
    if not settings.NOTIFY_DISCORD_ENABLED:
        return False
    target = url or (settings.DISCORD_ERROR_WEBHOOK_URL if error_channel else settings.DISCORD_WEBHOOK_URL)
    if not target:
        logger.debug("discord webhook URL missing; skipping post")
        return False
    try:
        r = httpx.post(target, json=payload, timeout=POST_TIMEOUT_S)
        if r.status_code >= 300:
            logger.warning("discord webhook {} returned {}: {}", target, r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("discord post failed: {}", e)
        return False
