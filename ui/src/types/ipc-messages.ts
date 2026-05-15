/**
 * Mirror of `engine/ipc/messages.py` — every wire frame is `{type, ts, data}`.
 * Hand-maintained for now; Phase 15 will codegen this from
 * `shared/ipc-schema.json` to eliminate drift.
 */

export type Frame<T = unknown> = { type: string; ts: number; data: T };

export type EngineStatus = {
  status: 'LIVE' | 'PAUSED' | 'TRAINING' | 'ERROR' | 'STARTING';
  uptime_s: number;
  mt5_connected: boolean;
  version?: string;
};

export type AccountUpdate = {
  equity: number;
  balance: number;
  free_margin: number;
  drawdown_pct: number;
  open_positions: number;
};

export type TickUpdate = {
  symbol: string; bid: number; ask: number; spread: number; volume?: number;
};

export type BarUpdate = {
  symbol: string;
  timeframe: 'M1' | 'M5' | 'M15' | 'H1' | 'H4' | 'D1';
  o: number; h: number; l: number; c: number; v: number;
  ts_bar: number;
};

export type Direction = 'BUY' | 'SELL';
export type SignalDir = 'BUY' | 'SELL' | 'HOLD';

export type SignalDetected = {
  signal_id: string;
  symbol: string;
  timeframe: string;
  direction: SignalDir;
  confluence: number;
  sources: {
    smc: SignalDir; cnn: SignalDir; rl: SignalDir;
    killzone: boolean; news_clear: boolean;
  };
  claude: {
    decision: 'BUY' | 'SELL' | 'SKIP';
    confidence: number; reasoning: string; risk_adjustment: number;
  } | null;
};

export type TradeOpened = {
  ticket: number; symbol: string; direction: Direction;
  entry: number; sl: number; tp: number; lot: number;
  signal_id?: string;
};
export type TradeUpdated = {
  ticket: number; current_price: number; pnl: number; rr_current?: number;
};
export type TradeClosed = {
  ticket: number; exit: number; pnl: number; rr_achieved?: number;
  close_reason: 'TP' | 'SL' | 'TRAIL' | 'MANUAL' | 'KILL' | 'NEWS';
};

export type ModelUpdate = {
  model_name: 'cnn_lstm' | 'rl_dqn';
  version: string;
  accuracy?: number | null;
  loss?: number | null;
};

export type RegimeChange = {
  symbol: string;
  regime: 'TRENDING_UP' | 'TRENDING_DOWN' | 'RANGING' | 'HIGH_VOLATILITY';
  adx?: number;
  atr_pct?: number;
};

export type CorrelationUpdate = {
  symbols: string[];
  matrix: number[][];
};

export type ClaudeFeed = {
  trade_id: number | null;
  symbol: string;
  decision: 'BUY' | 'SELL' | 'SKIP';
  confidence: number;
  reasoning_excerpt: string;
};

export type Notification = {
  event:
    | 'TRADE_OPENED' | 'TRADE_CLOSED_PROFIT' | 'TRADE_CLOSED_LOSS'
    | 'SIGNAL_DETECTED' | 'KILL_TRIGGERED' | 'NEWS_WARNING'
    | 'ENGINE_ERROR' | 'TRAINING_COMPLETE';
  title: string; body: string; sound: string;
};

export type WSStatus = { connected: boolean };

export type TradeRow = {
  id: number;
  mt5_ticket: number;
  symbol: string;
  direction: 'BUY' | 'SELL';
  entry_price: number;
  exit_price: number | null;
  lot_size: number;
  sl: number; tp: number;
  pnl: number | null;
  rr_achieved: number | null;
  open_time: string;
  close_time: string | null;
  close_reason: string | null;
  signal_confluence: number | null;
  claude_decision: string | null;
  claude_confidence: number | null;
  claude_reasoning: string | null;
};

export type TradesSnapshot = { trades: TradeRow[] };
export type SettingsSnapshot = { values: Record<string, string> };

