"""Pydantic models for the IPC wire protocol — Appendix D.

Every message on the WebSocket is `{type, ts, data}`. `ts` is unix epoch ms
in UTC. `data` is one of the typed payload models below; the discriminator is
the parent `type` field. `dump_schema()` writes the canonical JSON Schema to
`shared/ipc-schema.json` for the Electron side to consume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Engine → UI payloads
# ---------------------------------------------------------------------------

class EngineStatus(BaseModel):
    status: Literal["LIVE", "PAUSED", "TRAINING", "ERROR", "STARTING"]
    uptime_s: int = 0
    mt5_connected: bool = False
    version: str = "1.0.0"


class TickUpdate(BaseModel):
    symbol: str
    bid: float
    ask: float
    spread: float
    volume: float = 0.0


class BarUpdate(BaseModel):
    symbol: str
    timeframe: Literal["M1", "M5", "M15", "H1", "H4", "D1"]
    o: float
    h: float
    l: float
    c: float
    v: float
    ts_bar: int


class SignalSources(BaseModel):
    smc: Literal["BUY", "SELL", "HOLD"]
    cnn: Literal["BUY", "SELL", "HOLD"]
    rl: Literal["BUY", "SELL", "HOLD"]
    killzone: bool
    news_clear: bool


class ClaudeBlock(BaseModel):
    decision: Literal["BUY", "SELL", "SKIP"]
    confidence: int = Field(ge=0, le=100)
    reasoning: str = Field(max_length=600)
    risk_adjustment: float = Field(ge=0.5, le=1.5)


class SignalDetected(BaseModel):
    signal_id: str
    symbol: str
    timeframe: str
    direction: Literal["BUY", "SELL", "HOLD"]
    confluence: int = Field(ge=0, le=5)
    sources: SignalSources
    claude: ClaudeBlock | None = None


class TradeOpened(BaseModel):
    ticket: int
    symbol: str
    direction: Literal["BUY", "SELL"]
    entry: float
    sl: float
    tp: float
    lot: float
    signal_id: str | None = None


class TradeUpdated(BaseModel):
    ticket: int
    current_price: float
    pnl: float
    rr_current: float | None = None


class TradeClosed(BaseModel):
    ticket: int
    exit: float
    pnl: float
    rr_achieved: float | None = None
    close_reason: Literal["TP", "SL", "TRAIL", "MANUAL", "KILL", "NEWS"]


class AccountUpdate(BaseModel):
    equity: float
    balance: float
    free_margin: float
    drawdown_pct: float
    open_positions: int


class ModelUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: Literal["cnn_lstm", "rl_dqn"]
    version: str
    accuracy: float | None = None
    loss: float | None = None


class CorrelationUpdate(BaseModel):
    symbols: list[str]
    matrix: list[list[float]]


class RegimeChange(BaseModel):
    symbol: str
    regime: Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY"]
    adx: float | None = None
    atr_pct: float | None = None


class NewsWarning(BaseModel):
    event_name: str
    currency: str = Field(min_length=3, max_length=3)
    impact: Literal["LOW", "MEDIUM", "HIGH"]
    time_until_minutes: int
    affected_symbols: list[str] = Field(default_factory=list)


class KillTriggered(BaseModel):
    kind: Literal["INTRADAY", "WEEKLY", "MANUAL", "NEWS"]
    drawdown_pct: float
    positions_closed: int = 0
    halted_until: str | None = None


class PriceAlert(BaseModel):
    alert_id: int
    symbol: str
    direction: Literal["ABOVE", "BELOW"]
    threshold: float
    current_price: float


class ClaudeFeed(BaseModel):
    trade_id: int | None = None
    symbol: str
    decision: Literal["BUY", "SELL", "SKIP"]
    confidence: int
    reasoning_excerpt: str = Field(max_length=240)


class Notification(BaseModel):
    event: Literal[
        "TRADE_OPENED", "TRADE_CLOSED_PROFIT", "TRADE_CLOSED_LOSS",
        "SIGNAL_DETECTED", "KILL_TRIGGERED", "NEWS_WARNING",
        "ENGINE_ERROR", "TRAINING_COMPLETE",
    ]
    title: str
    body: str
    sound: str


# ---------------------------------------------------------------------------
# UI → Engine commands
# ---------------------------------------------------------------------------

class CmdEmergencyClose(BaseModel):
    pass


class CmdPause(BaseModel):
    paused: bool


class CmdManualRetrain(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: Literal["cnn_lstm", "rl_dqn"]


class CmdSettingsUpdate(BaseModel):
    partial: dict[str, Any]


class CmdRunBacktest(BaseModel):
    symbol: str
    from_: str = Field(alias="from")
    to: str
    strategy_config: dict[str, Any] = Field(default_factory=dict)


class CmdSetAlert(BaseModel):
    symbol: str
    direction: Literal["ABOVE", "BELOW"]
    threshold: float


class CmdGetTrades(BaseModel):
    limit: int = 200


class CmdGetSettings(BaseModel):
    pass


class TradesSnapshot(BaseModel):
    trades: list[dict[str, Any]]


class SettingsSnapshot(BaseModel):
    values: dict[str, Any]


class Ack(BaseModel):
    ref_type: str
    ok: bool
    error: str | None = None


class MacroSnapshot(BaseModel):
    yield_curve_bias: Literal["USD_BULLISH", "USD_BEARISH", "NEUTRAL"]
    crypto_fear_greed: Literal["EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"]
    fear_greed_value: int | None = None
    spread_us10y_us2y: float | None = None


class ShadowStatus(BaseModel):
    active: bool
    total: int
    open_count: int
    closed_count: int
    wins: int
    losses: int
    time_exits: int
    win_rate: float
    avg_r: float
    sharpe: float
    cumulative_pnl_r: float


class ModelPromotionReady(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    current_model_sharpe: float | None = None
    shadow_sharpe: float
    shadow_win_rate: float
    shadow_trades: int
    avg_r: float


class CalibrationUpdate(BaseModel):
    ece_score: float
    n_trades: int
    bins: list[dict[str, float]]
    overconfident: bool


class StrategyHealthFrame(BaseModel):
    name: str
    style: str
    state: Literal["ACTIVE", "PAUSED", "SHADOW", "DISABLED"]
    weight: float = Field(ge=0.0, le=1.0)
    rolling_sharpe: float
    consecutive_losses: int
    trades_today: int
    pnl_today_usd: float
    paused_until_ts: int = 0
    shadow_only_until_ts: int = 0


class StrategyStatus(BaseModel):
    total_risk_pct: float
    strategies: list[StrategyHealthFrame]


class CmdStrategyToggle(BaseModel):
    """Set a strategy's operating mode: ON, SHADOW (signal-only), or OFF."""
    name: str
    mode: Literal["ON", "SHADOW", "OFF"]


class BacktestResult(BaseModel):
    symbol: str
    timeframe: str
    from_date: str
    to_date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl_usd: float
    avg_r_multiple: float
    sharpe: float
    profit_factor: float
    max_drawdown_pct: float
    spread_pips_used: float
    slippage_pips_used: float
    swap_long_pips_used: float
    swap_short_pips_used: float
    starting_equity: float
    ending_equity: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Type registry — single source of truth for (type_string → model class)
# ---------------------------------------------------------------------------

PAYLOAD_TYPES: dict[str, type[BaseModel]] = {
    "engine_status":        EngineStatus,
    "tick_update":          TickUpdate,
    "bar_update":           BarUpdate,
    "signal_detected":      SignalDetected,
    "trade_opened":         TradeOpened,
    "trade_updated":        TradeUpdated,
    "trade_closed":         TradeClosed,
    "account_update":       AccountUpdate,
    "model_update":         ModelUpdate,
    "correlation_update":   CorrelationUpdate,
    "regime_change":        RegimeChange,
    "news_warning":         NewsWarning,
    "kill_triggered":       KillTriggered,
    "price_alert":          PriceAlert,
    "claude_feed":          ClaudeFeed,
    "notification":         Notification,
    "cmd_emergency_close":  CmdEmergencyClose,
    "cmd_pause":            CmdPause,
    "cmd_manual_retrain":   CmdManualRetrain,
    "cmd_settings_update":  CmdSettingsUpdate,
    "cmd_run_backtest":     CmdRunBacktest,
    "cmd_set_alert":        CmdSetAlert,
    "cmd_get_trades":       CmdGetTrades,
    "cmd_get_settings":     CmdGetSettings,
    "trades_snapshot":      TradesSnapshot,
    "settings_snapshot":    SettingsSnapshot,
    "macro_snapshot":       MacroSnapshot,
    "shadow_status":        ShadowStatus,
    "model_promotion_ready": ModelPromotionReady,
    "calibration_update":   CalibrationUpdate,
    "backtest_result":      BacktestResult,
    "strategy_status":      StrategyStatus,
    "cmd_strategy_toggle":  CmdStrategyToggle,
    "ack":                  Ack,
}

COMMAND_TYPES = frozenset({
    "cmd_emergency_close", "cmd_pause", "cmd_manual_retrain",
    "cmd_settings_update", "cmd_run_backtest", "cmd_set_alert",
    "cmd_get_trades", "cmd_get_settings", "cmd_strategy_toggle",
})


def envelope(type_: str, payload: BaseModel | dict | None) -> dict:
    """Wrap a typed payload in the standard `{type, ts, data}` envelope."""
    if isinstance(payload, BaseModel):
        data = payload.model_dump(by_alias=True)
    else:
        data = payload or {}
    return {"type": type_, "ts": now_ms(), "data": data}


def parse(raw: str | bytes | dict) -> tuple[str, BaseModel]:
    """Validate an inbound frame; return (type, model). Raises ValueError."""
    obj = json.loads(raw) if not isinstance(raw, dict) else raw
    if not isinstance(obj, dict) or "type" not in obj:
        raise ValueError("frame missing 'type'")
    t = obj["type"]
    cls = PAYLOAD_TYPES.get(t)
    if cls is None:
        raise ValueError(f"unknown type {t!r}")
    data = obj.get("data") or {}
    return t, cls.model_validate(data)


def dump_schema(out_path: Path | None = None) -> Path:
    """Write a coarse JSON Schema (one entry per payload) to disk."""
    out_path = out_path or (REPO_ROOT / "shared" / "ipc-schema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://complexity.engine/ipc-schema.json",
        "title": "Complexity Engine IPC Messages",
        "envelope": {
            "type": "object",
            "required": ["type", "ts", "data"],
            "properties": {
                "type": {"type": "string"},
                "ts":   {"type": "integer", "minimum": 0},
                "data": {"type": "object"},
            },
        },
        "payloads": {t: cls.model_json_schema() for t, cls in PAYLOAD_TYPES.items()},
    }
    out_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = dump_schema()
    print(f"wrote {p}")
