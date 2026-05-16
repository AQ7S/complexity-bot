import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useEngineStore } from '@/store/engineStore';
import { fmtPrice } from '@/lib/format';
import RegimeBadge from './RegimeBadge';
import { ALWAYS_ON } from '@/lib/constants';

const STALE_MS = 5 * 60_000;

function digitsFor(symbol: string): number {
  if (symbol.includes('JPY')) return 3;
  if (symbol.startsWith('GOLD') || symbol.startsWith('XAU')) return 2;
  if (symbol.includes('BTC') || symbol.includes('ETH') || symbol.includes('INDX') || symbol.includes('Crypto')) return 2;
  return 5;
}

function Sparkline({ values, up }: { values: number[]; up: boolean | null }) {
  if (values.length < 2) {
    return <div className="h-10 w-full opacity-30 text-[9px] text-white/30 flex items-center justify-center">no data</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = 100;
  const h = 36;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const stroke = up === null ? '#94a3b8' : up ? '#00ff88' : '#ff3b6b';
  const fill = up === null ? 'rgba(148,163,184,0.08)' : up ? 'rgba(0,255,136,0.10)' : 'rgba(255,59,107,0.10)';
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-10 w-full">
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill={fill} />
      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="1.2" />
    </svg>
  );
}

function VolumeBar({ pct }: { pct: number }) {
  const clamped = Math.max(0, Math.min(1, pct));
  const colour =
    clamped < 0.25 ? 'bg-white/30' :
    clamped > 0.85 ? 'bg-accent-gold' :
    'bg-accent-cyan';
  return (
    <div className="h-1 w-full rounded-full bg-bg-tertiary overflow-hidden">
      <div className={`h-full ${colour}`} style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

const SIGNAL_BADGE: Record<'BUY' | 'SELL' | 'HOLD' | 'SCAN', string> = {
  BUY:  'bg-accent-green/20 text-accent-green border-accent-green/40',
  SELL: 'bg-accent-red/20 text-accent-red border-accent-red/40',
  HOLD: 'bg-white/5 text-white/50 border-white/10',
  SCAN: 'bg-accent-cyan/10 text-accent-cyan/70 border-accent-cyan/30',
};

export default function SymbolCard({ symbol, kind }: { symbol: string; kind: string }) {
  const tick = useEngineStore((s) => s.ticks[symbol]);
  const regime = useEngineStore((s) => s.regimes[symbol]?.regime);
  const history = useEngineStore((s) => s.tickHistory[symbol]);
  const lastTickAt = useEngineStore((s) => s.lastTickTs[symbol]);
  const dayOpen = useEngineStore((s) => s.dayOpenMid[symbol]);
  const signal = useEngineStore((s) => s.signalsBySymbol[symbol]);
  const navigate = useNavigate();
  const digits = digitsFor(symbol);

  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 5_000);
    return () => clearInterval(id);
  }, []);

  const now = Date.now();
  const age = lastTickAt ? now - lastTickAt : Infinity;
  const stale = !tick || age > STALE_MS;
  const alwaysOn = ALWAYS_ON.has(symbol);
  const inDeadZone = stale && !alwaysOn;

  const sparkValues = useMemo(() => (history ?? []).map((h) => h.mid), [history]);
  const last = tick ? (tick.bid + tick.ask) / 2 : (sparkValues.at(-1) ?? null);
  const changePct = last != null && dayOpen ? ((last - dayOpen) / dayOpen) * 100 : null;
  const trendingUp =
    sparkValues.length < 2 ? null :
    sparkValues.at(-1)! > sparkValues[0];

  const volumePct = useMemo(() => {
    const hist = history ?? [];
    if (hist.length < 10) return 0;
    const recent = hist.slice(-10).reduce((acc, t, i, arr) => i ? acc + Math.abs(t.mid - arr[i - 1].mid) : acc, 0);
    const full = hist.reduce((acc, t, i, arr) => i ? acc + Math.abs(t.mid - arr[i - 1].mid) : acc, 0);
    const avg = full / Math.max(1, hist.length - 1);
    const recentAvg = recent / 9;
    if (avg <= 0) return 0;
    return Math.min(1.2, recentAvg / avg) / 1.2;
  }, [history]);

  const signalLabel: 'BUY' | 'SELL' | 'HOLD' | 'SCAN' =
    signal ? (signal.direction === 'BUY' ? 'BUY' : signal.direction === 'SELL' ? 'SELL' : 'HOLD') : 'SCAN';
  const signalConf = signal?.confluence;

  const onClick = () => navigate(`/charts?symbol=${encodeURIComponent(symbol)}`);

  const lastPrice = tick ? fmtPrice((tick.bid + tick.ask) / 2, digits) : (sparkValues.at(-1) ? fmtPrice(sparkValues.at(-1)!, digits) : '—');
  const stateLabel = inDeadZone ? (alwaysOn ? 'IDLE' : 'MARKET CLOSED') : 'LIVE';

  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={`symbol-card-${symbol}`}
      className={`group rounded-lg border bg-bg-secondary p-3 text-left transition-all hover:border-accent-cyan/40 hover:bg-bg-secondary/80 ${
        inDeadZone ? 'border-white/5 opacity-60' : 'border-white/5'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold text-white">{symbol}</span>
          <span className="rounded bg-white/5 px-1 text-[9px] uppercase text-white/40">
            {kind}
          </span>
          {alwaysOn && (
            <span className="text-[9px] text-accent-cyan">24/7</span>
          )}
        </div>
        <RegimeBadge regime={regime} />
      </div>

      {inDeadZone ? (
        <div className="mt-2 rounded bg-bg-tertiary/60 px-2 py-1 text-center">
          <div className="text-[9px] uppercase tracking-wider text-white/40">{stateLabel}</div>
          <div className="font-mono text-base text-white/40">{lastPrice}</div>
          {lastTickAt && (
            <div className="mt-0.5 text-[9px] text-white/30">
              last tick {Math.round(age / 60_000)}m ago
            </div>
          )}
        </div>
      ) : (
        <div className="mt-2 flex items-baseline justify-between font-mono">
          <div>
            <p className="text-[10px] uppercase text-white/40">Bid</p>
            <p className="text-base text-accent-red">{tick ? fmtPrice(tick.bid, digits) : '—'}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] uppercase text-white/40">Ask</p>
            <p className="text-base text-accent-green">{tick ? fmtPrice(tick.ask, digits) : '—'}</p>
          </div>
        </div>
      )}

      <div className="mt-1 flex items-center justify-between text-[10px] text-white/40">
        <span>Spread</span>
        <span className="font-mono">{tick ? fmtPrice(tick.spread, digits) : '—'}</span>
      </div>

      <div className="mt-1 flex items-center justify-between text-[10px]">
        <span className="uppercase text-white/40">24h</span>
        {changePct != null ? (
          <span className={`font-mono ${changePct >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
          </span>
        ) : (
          <span className="font-mono text-white/30">—</span>
        )}
      </div>

      <div className="mt-2">
        <Sparkline values={sparkValues} up={trendingUp} />
      </div>

      <div className="mt-2 flex items-center justify-between gap-2">
        <span
          className={`rounded border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${SIGNAL_BADGE[signalLabel]}`}
          data-testid={`signal-badge-${symbol}`}
        >
          {signalLabel}{signalConf != null ? ` ${signalConf}/5` : ''}
        </span>
        <div className="flex-1">
          <VolumeBar pct={volumePct} />
        </div>
      </div>
    </button>
  );
}
