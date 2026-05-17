import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';
import type { SignalDetected } from '@/types/ipc-messages';

const VOTE_COLOR = {
  BUY:  'bg-accent-green/20 text-accent-green border-accent-green/40',
  SELL: 'bg-accent-red/20 text-accent-red border-accent-red/40',
  HOLD: 'bg-white/5  text-white/50  border-white/10',
  SKIP: 'bg-white/5  text-white/50  border-white/10',
} as const;

const FLAG_COLOR = {
  true:  'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/40',
  false: 'bg-accent-red/10  text-accent-red/70 border-accent-red/30',
} as const;

function VoteChip({ label, value }: { label: string; value: 'BUY' | 'SELL' | 'HOLD' }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[9px] uppercase tracking-wider text-white/40">{label}</span>
      <span className={`mt-0.5 inline-block rounded border px-2 py-0.5 text-[10px] font-mono ${VOTE_COLOR[value]}`}>
        {value}
      </span>
    </div>
  );
}

function FlagChip({ label, value }: { label: string; value: boolean }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[9px] uppercase tracking-wider text-white/40">{label}</span>
      <span className={`mt-0.5 inline-block rounded border px-2 py-0.5 text-[10px] font-mono ${FLAG_COLOR[String(value) as 'true' | 'false']}`}>
        {value ? '✓ OK' : '✗ BLOCKED'}
      </span>
    </div>
  );
}

function qualityScore(sig: SignalDetected): { score: number; max: number; tone: 'high' | 'mid' | 'low' } {
  let score = 0;
  const max = 8;
  if (sig.sources.smc !== 'HOLD' && sig.sources.smc === sig.direction) score++;
  if (sig.sources.cnn !== 'HOLD' && sig.sources.cnn === sig.direction) score++;
  if (sig.sources.rl  !== 'HOLD' && sig.sources.rl  === sig.direction) score++;
  if (sig.sources.killzone) score++;
  if (sig.sources.news_clear) score++;
  if (sig.confluence >= 4) score++;
  if (sig.claude?.decision === sig.direction) score++;
  if (sig.claude && sig.claude.confidence >= 70) score++;
  const tone: 'high' | 'mid' | 'low' = score >= 7 ? 'high' : score >= 5 ? 'mid' : 'low';
  return { score, max, tone };
}

function SignalCard({ sig }: { sig: SignalDetected }) {
  const [open, setOpen] = useState(false);
  const dirColor = sig.direction === 'BUY' ? 'text-accent-green'
                 : sig.direction === 'SELL' ? 'text-accent-red'
                 : 'text-white/40';
  const claude = sig.claude;
  const q = qualityScore(sig);
  const qColor = q.tone === 'high' ? 'text-accent-cyan ring-accent-cyan/40 shadow-[0_0_18px_-4px_rgba(0,212,255,0.6)]'
              : q.tone === 'mid'   ? 'text-accent-gold ring-accent-gold/40'
              :                      'text-accent-red  ring-accent-red/30';

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`rounded-lg border border-white/5 bg-bg-secondary p-4 transition-shadow ${
        q.tone === 'high' ? 'ring-1 ring-accent-cyan/30 shadow-[0_0_24px_-8px_rgba(0,212,255,0.5)]' : ''
      }`}
      data-testid="decision-card"
    >
      <header className="flex flex-wrap items-baseline gap-3">
        <span className="font-mono text-lg font-bold text-white">{sig.symbol}</span>
        <span className="text-xs text-white/40">{sig.timeframe}</span>
        <span className={`font-mono text-lg font-bold ${dirColor}`}>{sig.direction}</span>
        <span className={`ml-auto inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono text-xs ring-1 ${qColor}`}>
          QUALITY {q.score}/{q.max}
        </span>
        <span className="rounded bg-bg-tertiary px-2 py-0.5 font-mono text-xs text-white/70">
          {sig.confluence}/5 confluence
        </span>
      </header>

      <section className="mt-3 flex flex-wrap gap-3 rounded bg-bg-tertiary/50 p-3">
        <VoteChip label="SMC"  value={sig.sources.smc} />
        <VoteChip label="CNN"  value={sig.sources.cnn} />
        <VoteChip label="RL"   value={sig.sources.rl} />
        <FlagChip label="Kill Zone"  value={sig.sources.killzone} />
        <FlagChip label="News Clear" value={sig.sources.news_clear} />
      </section>

      {claude ? (
        <section className="mt-3 rounded border border-accent-purple/30 bg-accent-purple/5 p-3">
          <header className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-accent-purple">
              Claude Gate
            </span>
            <div className="flex items-center gap-3 font-mono text-xs">
              <span className={`font-bold ${VOTE_COLOR[claude.decision].split(' ')[1]}`}>
                {claude.decision}
              </span>
              <span className="text-white/60">conf {claude.confidence}%</span>
              <span className="text-white/60">risk×{claude.risk_adjustment.toFixed(2)}</span>
              <button
                type="button"
                onClick={() => setOpen((o) => !o)}
                className="text-white/30 hover:text-white"
                data-testid="toggle-reasoning"
              >
                {open ? '▲' : '▼'}
              </button>
            </div>
          </header>
          <p className={`mt-2 text-xs text-white/70 ${open ? '' : 'line-clamp-2'}`}>
            {claude.reasoning}
          </p>
        </section>
      ) : (
        <section className="mt-3 rounded border border-white/5 bg-bg-tertiary/30 p-3 text-xs text-white/40">
          No Claude evaluation (pre-gate reject or gate disabled).
        </section>
      )}
    </motion.article>
  );
}

const DEMO_SIGNALS: SignalDetected[] = [
  {
    signal_id: 'demo-1', ts: Date.now(), symbol: 'EURUSD#', timeframe: 'M5', direction: 'BUY', confluence: 4,
    sources: { smc: 'BUY', cnn: 'BUY', rl: 'HOLD', killzone: true, news_clear: true },
    claude: {
      decision: 'BUY', confidence: 78, risk_adjustment: 1.1,
      reasoning: "Bullish OB confluence with NY open session active. RSI rising from oversold (32 -> 41) with positive divergence on M15. CNN-LSTM strong BUY (72% conf). RL holding due to range-bound state, but SMC + CNN agree with M15 bias. No high-impact news in next 4h. Recommend BUY with 1.1x risk adjustment given clean confluence."
    }
  },
  {
    signal_id: 'demo-2', ts: Date.now(), symbol: 'GOLD#', timeframe: 'M5', direction: 'SELL', confluence: 3,
    sources: { smc: 'SELL', cnn: 'SELL', rl: 'HOLD', killzone: true, news_clear: true },
    claude: {
      decision: 'SKIP', confidence: 42, risk_adjustment: 1.0,
      reasoning: "SMC and CNN agree SELL but ADX low (18) suggests ranging market on H1. Last 3 GOLD trades all losers. RSI neutral. Despite 3/5 consensus, the trend signal is weak and recent track record poor. Sit out and wait for cleaner structure break."
    }
  },
  {
    signal_id: 'demo-3', ts: Date.now(), symbol: 'GBPUSD#', timeframe: 'M5', direction: 'HOLD', confluence: 1,
    sources: { smc: 'HOLD', cnn: 'BUY', rl: 'SELL', killzone: false, news_clear: true },
    claude: null
  },
];

export default function DecisionTrace() {
  const signals = useEngineStore((s) => s.signals);
  const pushSignal = useEngineStore((s) => s.pushSignal);
  const [showDemo, setShowDemo] = useState(false);
  const [filterSymbol, setFilterSymbol]    = useState('');
  const [filterDir, setFilterDir]          = useState('');
  const [filterMinConf, setFilterMinConf]  = useState(0);

  useEffect(() => {
    if (signals.length > 0) { setShowDemo(false); return; }
    const t = setTimeout(() => setShowDemo(true), 4000);
    return () => clearTimeout(t);
  }, [signals.length]);

  const baseItems = signals.length > 0 ? signals : (showDemo ? DEMO_SIGNALS : []);

  const items = useMemo(() => baseItems.filter((s) => {
    if (filterSymbol && s.symbol !== filterSymbol) return false;
    if (filterDir && s.direction !== filterDir) return false;
    if (filterMinConf > 0 && s.confluence < filterMinConf) return false;
    return true;
  }), [baseItems, filterSymbol, filterDir, filterMinConf]);

  return (
    <section data-testid="page-decision-trace" className="space-y-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="font-hero text-2xl text-accent-cyan">Decision Trace</h1>
        <p className="text-xs text-white/40">
          Live 5-source consensus pipeline. Each signal shows SMC + CNN-LSTM + RL votes,
          kill-zone & news flags, and Claude's full reasoning.
        </p>
        <span className="ml-auto rounded bg-bg-tertiary px-2 py-1 font-mono text-xs text-white/60">
          {items.length}/{baseItems.length}
        </span>
        {signals.length === 0 && showDemo && (
          <button
            type="button"
            onClick={() => DEMO_SIGNALS.forEach((s) => pushSignal(s))}
            className="rounded bg-accent-purple/20 px-3 py-1 text-xs text-accent-purple hover:bg-accent-purple/30"
            data-testid="load-demo"
          >
            Pin demo signals
          </button>
        )}
      </header>

      {/* Filter bar */}
      <div className="flex flex-wrap gap-2 rounded-lg border border-white/5 bg-bg-secondary p-3">
        <select
          value={filterSymbol}
          onChange={(e) => setFilterSymbol(e.target.value)}
          className="rounded bg-bg-tertiary px-2 py-1 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="filter-symbol"
        >
          <option value="">All symbols</option>
          {SYMBOLS_13.map(({ name }) => <option key={name} value={name}>{name}</option>)}
        </select>

        <select
          value={filterDir}
          onChange={(e) => setFilterDir(e.target.value)}
          className="rounded bg-bg-tertiary px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="filter-direction"
        >
          <option value="">All directions</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
          <option value="HOLD">HOLD</option>
        </select>

        <select
          value={filterMinConf}
          onChange={(e) => setFilterMinConf(Number(e.target.value))}
          className="rounded bg-bg-tertiary px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="filter-confluence"
        >
          <option value={0}>Any confluence</option>
          <option value={3}>≥ 3/5</option>
          <option value={4}>≥ 4/5</option>
          <option value={5}>5/5 only</option>
        </select>

        {(filterSymbol || filterDir || filterMinConf > 0) && (
          <button
            type="button"
            onClick={() => { setFilterSymbol(''); setFilterDir(''); setFilterMinConf(0); }}
            className="rounded bg-white/5 px-2 py-1 text-[10px] text-white/50 hover:bg-white/10 hover:text-white"
          >
            Clear filters
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="rounded-lg border border-white/5 bg-bg-secondary p-8 text-center text-sm text-white/40">
          <div className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-accent-cyan border-t-transparent" />
          <p className="mt-3">Listening for live signals…</p>
          <p className="mt-2 text-xs text-white/30">
            Engine evaluates every M5 bar across 13 symbols. Sample signals will preview here if none arrive within 4 s.
          </p>
        </div>
      ) : (
        <>
          {signals.length === 0 && showDemo && (
            <div className="rounded border border-accent-purple/30 bg-accent-purple/5 px-3 py-2 text-xs text-accent-purple/80">
              Showing sample signals (engine has not emitted a live signal yet). Real signals will replace these automatically.
            </div>
          )}
          <AnimatePresence initial={false}>
            <div className="space-y-3">
              {items.map((s, i) => (
                <SignalCard key={`${s.signal_id}-${i}`} sig={s} />
              ))}
            </div>
          </AnimatePresence>
        </>
      )}
    </section>
  );
}
