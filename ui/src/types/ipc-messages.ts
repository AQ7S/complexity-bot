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
  ts: number;
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

export type NewsWarning = {
  event_name: string;
  currency: string;
  impact: 'LOW' | 'MEDIUM' | 'HIGH';
  time_until_minutes: number;
  affected_symbols?: string[];
};

export type WeeklyDebrief = {
  week_start: string;
  markdown: string;
  param_recommendations?: Record<string, unknown> | null;
  trades_count?: number | null;
  net_pnl?: number | null;
  win_rate?: number | null;
};

export type MacroSnapshot = {
  yield_curve_bias: 'USD_BULLISH' | 'USD_BEARISH' | 'NEUTRAL';
  crypto_fear_greed: 'EXTREME_FEAR' | 'FEAR' | 'NEUTRAL' | 'GREED' | 'EXTREME_GREED';
  fear_greed_value?: number | null;
  spread_us10y_us2y?: number | null;
};

export type ShadowStatus = {
  active: boolean;
  total: number;
  open_count: number;
  closed_count: number;
  wins: number;
  losses: number;
  time_exits: number;
  win_rate: number;
  avg_r: number;
  sharpe: number;
  cumulative_pnl_r: number;
};

export type ModelPromotionReady = {
  current_model_sharpe?: number | null;
  shadow_sharpe: number;
  shadow_win_rate: number;
  shadow_trades: number;
  avg_r: number;
};

export type CalibrationBin = {
  bin_start: number;
  bin_end: number;
  n: number;
  avg_confidence: number;
  win_rate: number;
};

export type CalibrationUpdate = {
  ece_score: number;
  n_trades: number;
  bins: CalibrationBin[];
  overconfident: boolean;
};

export type BacktestResult = {
  symbol: string;
  timeframe: string;
  from_date: string;
  to_date: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  net_pnl_usd: number;
  avg_r_multiple: number;
  sharpe: number;
  profit_factor: number;
  max_drawdown_pct: number;
  spread_pips_used: number;
  slippage_pips_used: number;
  swap_long_pips_used: number;
  swap_short_pips_used: number;
  starting_equity: number;
  ending_equity: number;
  error?: string | null;
};

export type Ack = {
  ref_type: string;
  ok: boolean;
  error?: string | null;
};

export type StrategyMode = 'ON' | 'SHADOW' | 'OFF';
export type StrategyState = 'ACTIVE' | 'PAUSED' | 'SHADOW' | 'DISABLED';

export type StrategyHealthFrame = {
  name: string;
  style: string;
  state: StrategyState;
  weight: number;
  rolling_sharpe: number;
  consecutive_losses: number;
  trades_today: number;
  pnl_today_usd: number;
  paused_until_ts: number;
  shadow_only_until_ts: number;
};

export type StrategyStatus = {
  total_risk_pct: number;
  strategies: StrategyHealthFrame[];
};
