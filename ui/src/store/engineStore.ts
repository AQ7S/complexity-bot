import { create } from 'zustand';
import type {
  AccountUpdate, ClaudeFeed, CorrelationUpdate, EngineStatus, ModelUpdate,
  Notification, RegimeChange, SignalDetected, TickUpdate, TradeClosed, TradeOpened,
  TradeRow, TradeUpdated, WSStatus,
} from '@/types/ipc-messages';

export type OpenPosition = TradeOpened & {
  current_price?: number;
  pnl?: number;
  rr_current?: number;
};

export type NotificationLogEntry = Notification & { id: number; ts: number; read: boolean };
export type TickSample = { ts: number; mid: number; spread: number };
export type ClosedTrade = TradeClosed & { ts: number; symbol?: string };

const CLAUDE_FEED_MAX = 20;
const SIGNAL_FEED_MAX = 40;
const NOTIFY_MAX = 60;
const TICK_HISTORY_MAX = 240;
const SPREAD_HISTORY_MAX = 60;
const CLAUDE_DECISION_MAX = 100;
const CLOSED_TRADES_MAX = 50;

type ClaudeDecisionStats = {
  total: number;
  buys: number;
  sells: number;
  skips: number;
  lastTs: number | null;
};

type State = {
  wsConnected: boolean;
  engineStatus: EngineStatus | null;
  account: AccountUpdate | null;
  weeklyPnl: number;
  todayPnl: number;
  sessionPnl: number;
  sessionStartEquity: number | null;
  sessionStartTs: number;
  ticks: Record<string, TickUpdate>;
  tickHistory: Record<string, TickSample[]>;
  lastTickTs: Record<string, number>;
  dayOpenMid: Record<string, number>;
  positions: Record<number, OpenPosition>;
  closedTrades: ClosedTrade[];
  signals: SignalDetected[];
  signalsBySymbol: Record<string, SignalDetected | undefined>;
  claudeFeed: ClaudeFeed[];
  claudeStats: ClaudeDecisionStats;
  regimes: Record<string, RegimeChange>;
  correlation: CorrelationUpdate | null;
  correlationFirstAt: number | null;
  tradesHistory: TradeRow[];
  settingsKv: Record<string, string>;
  modelUpdates: Record<'cnn_lstm' | 'rl_dqn', ModelUpdate | null>;
  notifications: NotificationLogEntry[];
  notificationsUnread: number;

  setWS: (s: WSStatus) => void;
  setEngineStatus: (s: EngineStatus) => void;
  setAccount: (a: AccountUpdate) => void;
  setTick: (t: TickUpdate) => void;
  upsertPositionOpened: (t: TradeOpened) => void;
  applyTradeUpdate: (u: TradeUpdated) => void;
  closePosition: (c: TradeClosed & { symbol?: string }) => void;
  pushSignal: (s: SignalDetected) => void;
  pushClaude: (c: ClaudeFeed) => void;
  setRegime: (r: RegimeChange) => void;
  setCorrelation: (c: CorrelationUpdate) => void;
  setTradesHistory: (t: TradeRow[]) => void;
  setSettingsKv: (v: Record<string, string>) => void;
  setModelUpdate: (m: ModelUpdate) => void;
  pushNotification: (n: Notification) => void;
  markNotificationsRead: () => void;
  clearNotifications: () => void;
};

function todayKeyEST(ts: number): string {
  const d = new Date(ts - 5 * 3600 * 1000);
  return d.toISOString().slice(0, 10);
}

export const useEngineStore = create<State>((set) => ({
  wsConnected: false,
  engineStatus: null,
  account: null,
  weeklyPnl: 0,
  todayPnl: 0,
  sessionPnl: 0,
  sessionStartEquity: null,
  sessionStartTs: Date.now(),
  ticks: {},
  tickHistory: {},
  lastTickTs: {},
  dayOpenMid: {},
  positions: {},
  closedTrades: [],
  signals: [],
  signalsBySymbol: {},
  claudeFeed: [],
  claudeStats: { total: 0, buys: 0, sells: 0, skips: 0, lastTs: null },
  regimes: {},
  correlation: null,
  correlationFirstAt: null,
  tradesHistory: [],
  settingsKv: {},
  modelUpdates: { cnn_lstm: null, rl_dqn: null },
  notifications: [],
  notificationsUnread: 0,

  setWS: (s) => set({ wsConnected: s.connected }),
  setEngineStatus: (s) => set({ engineStatus: s }),
  setAccount: (a) => set((prev) => ({
    account: a,
    sessionStartEquity: prev.sessionStartEquity ?? a.equity,
  })),
  setTick: (t) => set((prev) => {
    const now = Date.now();
    const mid = (t.bid + t.ask) / 2;
    const hist = prev.tickHistory[t.symbol] ?? [];
    const next = hist.length >= TICK_HISTORY_MAX
      ? [...hist.slice(1), { ts: now, mid, spread: t.spread }]
      : [...hist, { ts: now, mid, spread: t.spread }];
    const dayKey = todayKeyEST(now);
    const prevOpen = prev.dayOpenMid[t.symbol];
    const sessionKey = `${t.symbol}|${dayKey}`;
    const knownKey = (prev as any)._sessionKeys?.[t.symbol] as string | undefined;
    let dayOpen = prevOpen;
    let nextSessionKeys = (prev as any)._sessionKeys ?? {};
    if (knownKey !== sessionKey) {
      dayOpen = mid;
      nextSessionKeys = { ...nextSessionKeys, [t.symbol]: sessionKey };
    }
    return {
      ticks: { ...prev.ticks, [t.symbol]: t },
      tickHistory: { ...prev.tickHistory, [t.symbol]: next },
      lastTickTs: { ...prev.lastTickTs, [t.symbol]: now },
      dayOpenMid: { ...prev.dayOpenMid, [t.symbol]: dayOpen ?? mid },
      _sessionKeys: nextSessionKeys,
    } as any;
  }),

  upsertPositionOpened: (t) => set((prev) => ({
    positions: { ...prev.positions, [t.ticket]: { ...t } },
  })),
  applyTradeUpdate: (u) => set((prev) => {
    const existing = prev.positions[u.ticket];
    if (!existing) return {};
    return {
      positions: {
        ...prev.positions,
        [u.ticket]: {
          ...existing,
          current_price: u.current_price,
          pnl: u.pnl,
          rr_current: u.rr_current,
        },
      },
    };
  }),
  closePosition: (c) => set((prev) => {
    const next = { ...prev.positions };
    const closedSymbol = c.symbol ?? prev.positions[c.ticket]?.symbol;
    delete next[c.ticket];
    const entry: ClosedTrade = { ...c, ts: Date.now(), symbol: closedSymbol };
    const closed = [entry, ...prev.closedTrades].slice(0, CLOSED_TRADES_MAX);
    return {
      positions: next,
      weeklyPnl: prev.weeklyPnl + c.pnl,
      todayPnl: prev.todayPnl + c.pnl,
      sessionPnl: prev.sessionPnl + c.pnl,
      closedTrades: closed,
    };
  }),

  pushSignal: (s) => set((prev) => ({
    signals: [s, ...prev.signals].slice(0, SIGNAL_FEED_MAX),
    signalsBySymbol: { ...prev.signalsBySymbol, [s.symbol]: s },
  })),
  pushClaude: (c) => set((prev) => {
    const stats = { ...prev.claudeStats, total: prev.claudeStats.total + 1, lastTs: Date.now() };
    if (c.decision === 'BUY') stats.buys += 1;
    else if (c.decision === 'SELL') stats.sells += 1;
    else stats.skips += 1;
    return {
      claudeFeed: [c, ...prev.claudeFeed].slice(0, CLAUDE_FEED_MAX),
      claudeStats: stats,
    };
  }),
  setRegime: (r) => set((prev) => ({ regimes: { ...prev.regimes, [r.symbol]: r } })),
  setCorrelation: (c) => set((prev) => ({
    correlation: c,
    correlationFirstAt: prev.correlationFirstAt ?? Date.now(),
  })),
  setTradesHistory: (t) => set({ tradesHistory: t }),
  setSettingsKv: (v) => set({ settingsKv: v }),
  setModelUpdate: (m) => set((prev) => ({
    modelUpdates: { ...prev.modelUpdates, [m.model_name]: m },
  })),

  pushNotification: (n) => set((prev) => {
    const entry: NotificationLogEntry = { ...n, id: Date.now() + Math.random(), ts: Date.now(), read: false };
    return {
      notifications: [entry, ...prev.notifications].slice(0, NOTIFY_MAX),
      notificationsUnread: prev.notificationsUnread + 1,
    };
  }),
  markNotificationsRead: () => set((prev) => ({
    notifications: prev.notifications.map((n) => ({ ...n, read: true })),
    notificationsUnread: 0,
  })),
  clearNotifications: () => set({ notifications: [], notificationsUnread: 0 }),
}));

void SPREAD_HISTORY_MAX;
void CLAUDE_DECISION_MAX;
