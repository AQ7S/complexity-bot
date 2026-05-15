import { useEffect, useMemo, useRef, useState } from 'react';
import CandlestickChart, { type Candle } from '@/components/charts/CandlestickChart';
import { SYMBOLS_13, TIMEFRAMES, type Timeframe } from '@/lib/constants';
import { useEngineStore } from '@/store/engineStore';

const TF_SECONDS: Record<Timeframe, number> = {
  M1: 60, M5: 300, M15: 900, H1: 3600, H4: 14400, D1: 86400,
};

function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0] ?? 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    prev = i === 0 ? v : v * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

function syntheticHistory(basePrice: number, tf: Timeframe, n: number): Candle[] {
  const tfSec = TF_SECONDS[tf];
  const now = Math.floor(Date.now() / 1000);
  const bucketStart = now - (now % tfSec);
  const out: Candle[] = [];
  let p = basePrice * (1 - 0.002);
  for (let i = n - 1; i > 0; i--) {
    const drift = Math.sin(i / 13) * basePrice * 0.0004;
    const noise = (Math.random() - 0.5) * basePrice * 0.0006;
    const o = p;
    const c = p + drift + noise;
    const h = Math.max(o, c) + Math.abs(noise) * 0.6;
    const l = Math.min(o, c) - Math.abs(noise) * 0.6;
    out.push({ time: bucketStart - i * tfSec, open: o, high: h, low: l, close: c });
    p = c;
  }
  return out;
}

export default function Charts() {
  const [symbol, setSymbol] = useState('EURUSD#');
  const [tf, setTf] = useState<Timeframe>('M5');
  const tick = useEngineStore((s) => s.ticks[symbol]);
  const liveMid = tick ? (tick.bid + tick.ask) / 2 : null;

  const [history, setHistory] = useState<Candle[]>([]);
  const baseRef = useRef<number | null>(null);
  const tfSec = TF_SECONDS[tf];

  useEffect(() => {
    const base = liveMid ?? (symbol.includes('JPY') ? 150 : symbol.startsWith('GOLD') ? 2350 : symbol.startsWith('BTC') ? 75000 : 1.07);
    baseRef.current = base;
    setHistory(syntheticHistory(base, tf, 200));
  }, [symbol, tf]);

  useEffect(() => {
    if (liveMid == null) return;
    const now = Math.floor(Date.now() / 1000);
    const bucket = now - (now % tfSec);
    setHistory((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.time === bucket) {
        const updated: Candle = {
          time: bucket,
          open: last.open,
          high: Math.max(last.high, liveMid),
          low: Math.min(last.low, liveMid),
          close: liveMid,
        };
        return [...prev.slice(0, -1), updated];
      }
      const newBar: Candle = { time: bucket, open: liveMid, high: liveMid, low: liveMid, close: liveMid };
      return [...prev.slice(-199), newBar];
    });
  }, [liveMid, tfSec]);

  const closes = useMemo(() => history.map((c) => c.close), [history]);
  const emas = useMemo(() => [9, 21, 50].map((p) => ({
    period: p,
    data: ema(closes, p).map((v, i) => ({ time: history[i].time, value: v })),
  })), [closes, history]);
  const last = history.at(-1);
  const isLive = liveMid != null;

  return (
    <section data-testid="page-charts" className="flex h-full flex-col p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <select
            data-testid="chart-symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="rounded bg-bg-secondary px-2 py-1 text-sm font-mono text-white"
          >
            {SYMBOLS_13.map(({ name }) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <div className="flex rounded border border-white/10">
            {TIMEFRAMES.map((t) => (
              <button
                key={t}
                data-testid={`tf-${t}`}
                onClick={() => setTf(t)}
                className={`px-2 py-1 text-xs font-mono ${
                  tf === t ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-white/60 hover:text-white'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <span className={`ml-2 flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            isLive ? 'bg-accent-green/20 text-accent-green' : 'bg-white/5 text-white/40'
          }`}>
            <span className={`inline-block h-1.5 w-1.5 rounded-full ${isLive ? 'animate-pulse bg-accent-green' : 'bg-white/30'}`} />
            {isLive ? 'LIVE' : 'WAITING'}
          </span>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs">
          <span className="text-white/40">Last</span>
          <span className="text-white">{last?.close.toFixed(5)}</span>
          {emas.map((e, i) => (
            <span key={e.period} className="text-white/60">
              EMA{e.period}{' '}
              <span className={i === 0 ? 'text-accent-cyan' : i === 1 ? 'text-accent-gold' : 'text-accent-purple'}>
                {e.data.at(-1)?.value.toFixed(5)}
              </span>
            </span>
          ))}
        </div>
      </div>
      <div className="flex-1">
        <CandlestickChart candles={history} emas={emas} />
      </div>
    </section>
  );
}
