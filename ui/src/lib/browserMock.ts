/**
 * Browser mock for window.engineBridge.
 * Active when the React app is loaded outside Electron (vite dev server / Chrome).
 * Emits realistic fake data so every UI panel has something to render.
 */

import { SYMBOLS_13 } from './constants';

type EventHandler = (frame: { type: string; ts: number; data: unknown }) => void;

// ─── Realistic base mid-prices ──────────────────────────────────────────────
const BASE: Record<string, number> = {
  'EURUSD#': 1.08542,
  'USDJPY#': 154.321,
  'GBPUSD#': 1.26845,
  'USDCHF#': 0.89213,
  'GOLD#': 2348.50,
  'BTCUSD#': 63_241.0,
  'ETHUSD#': 3_187.5,
  'AI_INDX#': 1_842.3,
  'Crypto_10#': 4_210.7,
  'TrumpWinners#': 0.6512,
  'HarrisWinners#': 0.3491,
  'EURJPY#': 167.812,
  'AUDUSD#': 0.65421,
};

function spread(name: string): number {
  if (name === 'GOLD#') return 0.30;
  if (name.includes('BTC') || name.includes('ETH')) return 12.0;
  if (name.includes('JPY')) return 0.012;
  if (name.includes('Winners') || name.includes('Crypto') || name.includes('AI')) return 0.002;
  return 0.00012;
}

function digits(name: string): number {
  if (name === 'GOLD#') return 2;
  if (name.includes('JPY')) return 3;
  if (name.includes('BTC')) return 1;
  if (name.includes('ETH') || name.includes('AI_INDX') || name.includes('Crypto')) return 2;
  return 5;
}

function noise(base: number, bps = 5): number {
  return base * (1 + (Math.random() - 0.5) * bps * 0.0001);
}

// ─── Shared mutable state ────────────────────────────────────────────────────
const mids: Record<string, number> = { ...BASE };

// ─── Mock broker ─────────────────────────────────────────────────────────────
export function installBrowserMock(): void {
  if (typeof window === 'undefined') return;
  if ((window as any).engineBridge) return; // real bridge already present

  const handlers = new Set<EventHandler>();
  let equity = 10_412.88;
  let balance = 10_412.88;
  let uptime = 0;
  let notifId = 0;

  function emit(type: string, data: unknown): void {
    const frame = { type, ts: Date.now(), data };
    handlers.forEach((h) => h(frame));
  }

  // ── Engine status ──────────────────────────────────────────────────────────
  emit('engine_status', { status: 'LIVE', uptime_s: 0, mt5_connected: true, version: '1.0.6' });
  emit('ui:ws_status', { connected: true });

  const intervals: ReturnType<typeof setInterval>[] = [];

  // ── Account ────────────────────────────────────────────────────────────────
  intervals.push(setInterval(() => {
    uptime += 5;
    equity += (Math.random() - 0.48) * 0.8;
    const drawdown_pct = Math.max(0, (balance - equity) / balance);
    emit('engine_status', { status: 'LIVE', uptime_s: uptime, mt5_connected: true, version: '1.0.6' });
    emit('account_update', {
      equity: +equity.toFixed(2),
      balance: +balance.toFixed(2),
      free_margin: +(equity * 0.92).toFixed(2),
      drawdown_pct: +drawdown_pct.toFixed(4),
      open_positions: 2,
    });
  }, 5_000));

  // ── Initial account ────────────────────────────────────────────────────────
  setTimeout(() => {
    emit('account_update', {
      equity: 10_412.88,
      balance: 10_412.88,
      free_margin: 9_580.24,
      drawdown_pct: 0,
      open_positions: 2,
    });
  }, 300);

  // ── Ticks ──────────────────────────────────────────────────────────────────
  intervals.push(setInterval(() => {
    for (const { name } of SYMBOLS_13) {
      mids[name] = noise(mids[name] ?? BASE[name]);
      const sp = spread(name);
      const bid = +(mids[name] - sp / 2).toFixed(digits(name));
      const ask = +(mids[name] + sp / 2).toFixed(digits(name));
      emit('tick_update', { symbol: name, bid, ask, spread: sp, volume: +(Math.random() * 3).toFixed(2) });
    }
  }, 800));

  // ── Open positions ─────────────────────────────────────────────────────────
  const pos1Ts = Date.now() - 22 * 60 * 1000;
  const pos2Ts = Date.now() - 7 * 60 * 1000;
  setTimeout(() => {
    emit('trade_opened', {
      ticket: 10_000_001,
      symbol: 'EURUSD#',
      direction: 'BUY',
      entry: 1.08420,
      sl: 1.08270,
      tp: 1.08720,
      lot: 0.45,
      signal_id: 'mock-sig-1',
      ts: pos1Ts,
    });
    emit('trade_opened', {
      ticket: 10_000_002,
      symbol: 'GOLD#',
      direction: 'SELL',
      entry: 2352.10,
      sl: 2358.50,
      tp: 2339.20,
      lot: 0.10,
      signal_id: 'mock-sig-2',
      ts: pos2Ts,
    });
  }, 400);

  intervals.push(setInterval(() => {
    const eurMid = mids['EURUSD#'] ?? 1.08542;
    const goldMid = mids['GOLD#'] ?? 2348.5;
    const pnl1 = +((eurMid - 1.08420) * 10000 * 4.5).toFixed(2);
    const pnl2 = +((2352.10 - goldMid) * 0.10 * 100).toFixed(2);
    const rr1 = +((eurMid - 1.08420) / (1.08420 - 1.08270)).toFixed(2);
    const rr2 = +((2352.10 - goldMid) / (2358.50 - 2352.10)).toFixed(2);
    emit('trade_updated', { ticket: 10_000_001, current_price: eurMid, pnl: pnl1, rr_current: rr1 });
    emit('trade_updated', { ticket: 10_000_002, current_price: goldMid, pnl: pnl2, rr_current: rr2 });
  }, 1_500));

  // ── Regimes ────────────────────────────────────────────────────────────────
  const REGIMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOLATILITY'] as const;
  for (const { name } of SYMBOLS_13) {
    emit('regime_change', {
      symbol: name,
      regime: REGIMES[Math.floor(Math.random() * REGIMES.length)],
      adx: +(18 + Math.random() * 25).toFixed(1),
      atr_pct: +(0.04 + Math.random() * 0.08).toFixed(4),
    });
  }

  // ── Correlation ────────────────────────────────────────────────────────────
  setTimeout(() => {
    const syms = SYMBOLS_13.map((s) => s.name);
    const n = syms.length;
    const matrix: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (__, j) => {
        if (i === j) return 1;
        const v = +(Math.random() * 1.6 - 0.8).toFixed(3);
        return Math.max(-1, Math.min(1, v));
      }),
    );
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) matrix[j][i] = matrix[i][j];
    emit('correlation_update', { symbols: syms, matrix });
  }, 600);

  // ── Signals ────────────────────────────────────────────────────────────────
  const DIRS = ['BUY', 'SELL'] as const;
  const VOTES = ['BUY', 'SELL', 'HOLD'] as const;
  const symsArr = SYMBOLS_13.map((s) => s.name);
  let sigIdx = 0;

  function emitSignal() {
    const sym = symsArr[Math.floor(Math.random() * symsArr.length)];
    const dir = DIRS[Math.floor(Math.random() * DIRS.length)];
    const conf = 30 + Math.floor(Math.random() * 60);
    const usesClaude = conf > 50;
    const claudeDecision = usesClaude
      ? (Math.random() > 0.25 ? dir : 'SKIP')
      : null;
    emit('signal_detected', {
      signal_id: `mock-${++sigIdx}`,
      ts: Date.now(),
      symbol: sym,
      timeframe: 'M5',
      direction: dir,
      confluence: 2 + Math.floor(Math.random() * 4),
      sources: {
        smc:  VOTES[Math.floor(Math.random() * 3)],
        cnn:  dir,
        rl:   VOTES[Math.floor(Math.random() * 3)],
        killzone:  Math.random() > 0.3,
        news_clear: Math.random() > 0.15,
      },
      claude: usesClaude ? {
        decision: claudeDecision,
        confidence: conf,
        reasoning: `Mock reasoning for ${sym} ${dir}: confluence aligned with HTF bias, RSI oversold recovery, no high-impact news in next 4h.`,
        risk_adjustment: +(0.8 + Math.random() * 0.5).toFixed(2),
      } : null,
    });
  }

  // ── Claude feed ────────────────────────────────────────────────────────────
  function emitClaude() {
    const sym = symsArr[Math.floor(Math.random() * symsArr.length)];
    const dec = Math.random() > 0.5 ? 'BUY' : Math.random() > 0.5 ? 'SELL' : 'SKIP';
    emit('claude_feed', {
      trade_id: Math.floor(Math.random() * 50) + 1,
      symbol: sym,
      decision: dec,
      confidence: 40 + Math.floor(Math.random() * 55),
      reasoning_excerpt: `${sym}: ${dec === 'SKIP' ? 'Conflicting signals. Sitting out.' : `Confirmed ${dec} — OB intact, momentum aligned.`}`,
    });
  }

  // Seed initial signals/claude
  setTimeout(() => {
    for (let i = 0; i < 5; i++) emitSignal();
    for (let i = 0; i < 4; i++) emitClaude();
  }, 700);

  intervals.push(setInterval(emitSignal, 12_000));
  intervals.push(setInterval(emitClaude, 8_000));

  // ── Model updates ──────────────────────────────────────────────────────────
  setTimeout(() => {
    emit('model_update', { model_name: 'cnn_lstm', version: 'v12_2026-05-14', accuracy: 0.5681, loss: 0.7421 });
    emit('model_update', { model_name: 'rl_dqn',   version: 'v8_2026-05-10',  accuracy: 0.5220, loss: 0.8813 });
  }, 1_000);

  // ── Historical trades (for journal / heatmap) ──────────────────────────────
  setTimeout(() => {
    const hist = Array.from({ length: 28 }, (_, i) => {
      const sym = symsArr[i % symsArr.length];
      const dir = i % 3 === 0 ? 'SELL' : 'BUY';
      const entry = +(mids[sym] * (1 + (Math.random() - 0.5) * 0.002)).toFixed(digits(sym));
      const pnl = +((Math.random() - 0.38) * 120).toFixed(2);
      const daysAgo = i * 1.4;
      const openTime = new Date(Date.now() - daysAgo * 86_400_000).toISOString();
      const closeTime = new Date(Date.now() - daysAgo * 86_400_000 + 3_600_000).toISOString();
      return {
        id: 1000 + i,
        mt5_ticket: 10_000_100 + i,
        symbol: sym,
        direction: dir,
        entry_price: entry,
        exit_price: +(entry * (1 + (pnl > 0 ? 0.001 : -0.001))).toFixed(digits(sym)),
        lot_size: 0.10 + (i % 4) * 0.05,
        sl: +(entry * 0.999).toFixed(digits(sym)),
        tp: +(entry * 1.003).toFixed(digits(sym)),
        pnl,
        rr_achieved: +(pnl / 50).toFixed(2),
        open_time: openTime,
        close_time: closeTime,
        close_reason: pnl > 0 ? 'TP' : 'SL',
        signal_confluence: 3 + (i % 3),
        claude_decision: ['BUY', 'SELL', 'SKIP'][i % 3],
        claude_confidence: 55 + (i % 35),
        claude_reasoning: 'Mock Claude reasoning for historical trade.',
        synced_supabase: 1,
      };
    });
    emit('trades_snapshot', { trades: hist });

    // also push closed trade events so heatmap/sessionPnl work
    for (const t of hist.slice(0, 8)) {
      emit('trade_closed', {
        ticket: t.mt5_ticket,
        symbol: t.symbol,
        exit: t.exit_price,
        pnl: t.pnl,
        rr_achieved: t.rr_achieved,
        close_reason: t.close_reason,
      } as any);
    }
  }, 800);

  // ── Weekly debrief (once after 8s for browser testing) ───────────────────
  setTimeout(() => {
    emit('weekly_debrief', {
      week_start: '2026-05-11',
      markdown: `## Top 3 Mistakes\n- **Over-trading USDJPY#** during Asian session ranging conditions (ADX < 18). 3 of 4 losses came from this pair.\n- **Ignoring news window** — 2 trades entered within 25 min of USD NFP; both hit SL.\n- **Low-confluence entries** — 2 trades with confluence 3/5 where CNN and RL disagreed; should have waited.\n\n## Top 3 Successful Patterns\n- **GOLD# SELL during London open** with OB rejection + CNN 78% + RL BUY-hold: 2 wins, avg R:R 2.1.\n- **EURUSD# BUY on NY open pullbacks** to EMA21 with bullish OB intact: 3 wins, avg R:R 1.8.\n- **5/5 confluence entries** always profitable this week (3/3); consider raising risk_adjustment cap to 1.4.\n\n## Recommended Parameter Changes\n\`\`\`json\n{"NEWS_PAUSE_MINUTES_BEFORE": 35, "CONSENSUS_MIN_AGREE": 3, "ATR_SL_MULTIPLIER": 1.6}\n\`\`\`\n\n## Market Conditions Summary\nWeek characterized by USD strength post-NFP and range-bound Gold. EUR showed strong trend continuation during NY open sessions. Risk-on sentiment supported EURUSD longs. Avoid counter-trend entries on USDJPY until ADX > 25.`,
      trades_count: 11,
      net_pnl: 214.50,
      win_rate: 0.636,
      param_recommendations: { NEWS_PAUSE_MINUTES_BEFORE: 35, ATR_SL_MULTIPLIER: 1.6 },
    });
  }, 8_000);

  // ── News warning (once after 15s, then every 90s) ──────────────────────────
  const newsWarning = () =>
    emit('news_warning', {
      event_name: 'US Non-Farm Payrolls',
      currency: 'USD',
      impact: 'HIGH',
      time_until_minutes: 28 + Math.floor(Math.random() * 5),
      affected_symbols: ['EURUSD#', 'USDJPY#', 'GBPUSD#', 'GOLD#'],
    });
  setTimeout(newsWarning, 15_000);
  intervals.push(setInterval(newsWarning, 90_000));

  // ── Notification samples ───────────────────────────────────────────────────
  setTimeout(() => {
    const samples = [
      { event: 'TRADE_OPENED',  title: 'EURUSD# BUY 0.45', body: 'Entry 1.08420 · SL 1.08270 · TP 1.08720', sound: 'trading_open.wav' },
      { event: 'SIGNAL_DETECTED', title: 'GOLD# SELL signal', body: '4/5 confluence · Claude 72%', sound: 'signal.wav' },
    ];
    for (const n of samples) {
      emit('notification', { ...n, id: ++notifId });
    }
  }, 1_200);

  intervals.push(setInterval(() => {
    emit('notification', {
      event: 'SIGNAL_DETECTED',
      title: `${symsArr[Math.floor(Math.random() * symsArr.length)]} signal`,
      body: `Confluence ${3 + Math.floor(Math.random() * 2)}/5 · Claude ${50 + Math.floor(Math.random() * 40)}%`,
      sound: 'signal.wav',
      id: ++notifId,
    });
  }, 25_000));

  // ── Settings snapshot ──────────────────────────────────────────────────────
  setTimeout(() => {
    emit('settings_snapshot', {
      values: {
        notify_sound_enabled: 'true',
        notify_toast_enabled: 'true',
        notify_discord_enabled: 'true',
        enable_claude_gate: 'true',
        enable_rl: 'true',
        risk_pct: '2.0',
      },
    });
  }, 500);

  // ── Bridge API ─────────────────────────────────────────────────────────────
  (window as any).engineBridge = {
    onEvent(handler: EventHandler) {
      handlers.add(handler);
      // Re-send status so late subscribers get it
      handler({ type: 'ui:ws_status', ts: Date.now(), data: { connected: true } });
      return () => handlers.delete(handler);
    },
    send(_frame: unknown): Promise<boolean> {
      // Simulate engine ACK after short delay
      return new Promise((resolve) => setTimeout(() => resolve(true), 80));
    },
  };

  // cleanup on HMR
  const hot = (import.meta as any).hot;
  if (hot) {
    hot.dispose(() => {
      intervals.forEach(clearInterval);
      delete (window as any).engineBridge;
    });
  }
}
