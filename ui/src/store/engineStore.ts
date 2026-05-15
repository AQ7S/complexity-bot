import { create } from 'zustand';
import type {
  AccountUpdate, ClaudeFeed, CorrelationUpdate, EngineStatus, ModelUpdate,
  RegimeChange, SignalDetected, TickUpdate, TradeClosed, TradeOpened,
  TradeRow, TradeUpdated, WSStatus,
} from '@/types/ipc-messages';

export type OpenPosition = TradeOpened & {
  current_price?: number;
  pnl?: number;
  rr_current?: number;
};

const CLAUDE_FEED_MAX = 20;
const SIGNAL_FEED_MAX = 20;

type State = {
  wsConnected: boolean;
  engineStatus: EngineStatus | null;
  account: AccountUpdate | null;
  weeklyPnl: number;          // running sum of trade_closed pnl this session
  ticks: Record<string, TickUpdate>;
  positions: Record<number, OpenPosition>;
  signals: SignalDetected[];   // newest first
  claudeFeed: ClaudeFeed[];    // newest first
  regimes: Record<string, RegimeChange>;
  correlation: CorrelationUpdate | null;
  tradesHistory: TradeRow[];
  settingsKv: Record<string, string>;
  modelUpdates: Record<'cnn_lstm' | 'rl_dqn', ModelUpdate | null>;

  setWS: (s: WSStatus) => void;
  setEngineStatus: (s: EngineStatus) => void;
  setAccount: (a: AccountUpdate) => void;
  setTick: (t: TickUpdate) => void;
  upsertPositionOpened: (t: TradeOpened) => void;
  applyTradeUpdate: (u: TradeUpdated) => void;
  closePosition: (c: TradeClosed) => void;
  pushSignal: (s: SignalDetected) => void;
  pushClaude: (c: ClaudeFeed) => void;
  setRegime: (r: RegimeChange) => void;
  setCorrelation: (c: CorrelationUpdate) => void;
  setTradesHistory: (t: TradeRow[]) => void;
  setSettingsKv: (v: Record<string, string>) => void;
  setModelUpdate: (m: ModelUpdate) => void;
};

export const useEngineStore = create<State>((set) => ({
  wsConnected: false,
  engineStatus: null,
  account: null,
  weeklyPnl: 0,
  ticks: {},
  positions: {},
  signals: [],
  claudeFeed: [],
  regimes: {},
  correlation: null,
  tradesHistory: [],
  settingsKv: {},
  modelUpdates: { cnn_lstm: null, rl_dqn: null },

  setWS: (s) => set({ wsConnected: s.connected }),
  setEngineStatus: (s) => set({ engineStatus: s }),
  setAccount: (a) => set({ account: a }),
  setTick: (t) => set((prev) => ({ ticks: { ...prev.ticks, [t.symbol]: t } })),

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
    delete next[c.ticket];
    return { positions: next, weeklyPnl: prev.weeklyPnl + c.pnl };
  }),

  pushSignal: (s) => set((prev) => ({
    signals: [s, ...prev.signals].slice(0, SIGNAL_FEED_MAX),
  })),
  pushClaude: (c) => set((prev) => ({
    claudeFeed: [c, ...prev.claudeFeed].slice(0, CLAUDE_FEED_MAX),
  })),
  setRegime: (r) => set((prev) => ({ regimes: { ...prev.regimes, [r.symbol]: r } })),
  setCorrelation: (c) => set({ correlation: c }),
  setTradesHistory: (t) => set({ tradesHistory: t }),
  setSettingsKv: (v) => set({ settingsKv: v }),
  setModelUpdate: (m) => set((prev) => ({
    modelUpdates: { ...prev.modelUpdates, [m.model_name]: m },
  })),
}));
