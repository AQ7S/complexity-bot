import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import CandlestickChart, {
  type Candle, type EmaSeries, type Marker, type VwapSeries, type Zone,
  type ShadedRange, type CrosshairInfo,
} from '@/components/charts/CandlestickChart';
import { SYMBOLS_13, TIMEFRAMES, KILL_ZONES, type Timeframe } from '@/lib/constants';
import { useEngineStore } from '@/store/engineStore';

const TF_SECONDS: Record<Timeframe, number> = {
  M1: 60, M5: 300, M15: 900, H1: 3600, H4: 14400, D1: 86400,
};

// ── Technical Indicators ────────────────────────────────────────────────────

function calcEma(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0] ?? 0;
  for (let i = 0; i < values.length; i++) {
    prev = i === 0 ? values[i] : values[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

function calcRsi(closes: number[], period = 14): number[] {
  const out = new Array(closes.length).fill(NaN);
  if (closes.length <= period) return out;
  let avgGain = 0; let avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    avgGain += d > 0 ? d : 0;
    avgLoss += d < 0 ? -d : 0;
  }
  avgGain /= period; avgLoss /= period;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    avgGain = (avgGain * (period - 1) + (d > 0 ? d : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (d < 0 ? -d : 0)) / period;
    out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

function calcAtr(candles: Candle[], period = 14): number[] {
  const tr: number[] = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1].close;
    return Math.max(c.high - c.low, Math.abs(c.high - prev), Math.abs(c.low - prev));
  });
  const out: number[] = [];
  let atr = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < candles.length; i++) {
    if (i < period - 1) { out.push(NaN); continue; }
    if (i === period - 1) { out.push(atr); continue; }
    atr = (atr * (period - 1) + tr[i]) / period;
    out.push(atr);
  }
  return out;
}

function calcAdx(candles: Candle[], period = 14): number {
  if (candles.length < period * 2) return NaN;
  let plusDm = 0; let minusDm = 0; let trSum = 0;
  for (let i = 1; i <= period; i++) {
    const c = candles[candles.length - i];
    const p = candles[candles.length - i - 1];
    const upMove = c.high - p.high;
    const downMove = p.low - c.low;
    if (upMove > downMove && upMove > 0) plusDm += upMove;
    if (downMove > upMove && downMove > 0) minusDm += downMove;
    const tr = Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close));
    trSum += tr;
  }
  if (trSum === 0) return NaN;
  const di_plus = (plusDm / trSum) * 100;
  const di_minus = (minusDm / trSum) * 100;
  const dif = Math.abs(di_plus - di_minus);
  const sum = di_plus + di_minus;
  return sum === 0 ? 0 : (dif / sum) * 100;
}

function calcMacd(closes: number[]): { macd: number; signal: number; hist: number } {
  if (closes.length < 26) return { macd: 0, signal: 0, hist: 0 };
  const fast = calcEma(closes, 12);
  const slow = calcEma(closes, 26);
  const macdLine = fast.map((v, i) => v - slow[i]);
  const signal = calcEma(macdLine.slice(26 - 9), 9);
  const macd = macdLine.at(-1) ?? 0;
  const sig = signal.at(-1) ?? 0;
  return { macd, signal: sig, hist: macd - sig };
}

// ── SMC Zone Detection ────────────────────────────────────────────────────

function detectZones(candles: Candle[], atrs: number[]): Zone[] {
  const zones: Zone[] = [];
  const n = candles.length;
  if (n < 10) return zones;

  // FVG detection (3-candle imbalance)
  for (let i = 1; i < n - 1; i++) {
    const prev = candles[i - 1];
    const next = candles[i + 1];
    const gap = next.low - prev.high;
    const gap2 = prev.low - next.high;
    if (gap > 0) {
      zones.push({
        time_from: candles[i].time,
        time_to: candles[n - 1].time + (candles[n - 1].time - candles[n - 2]?.time || 300),
        price_high: next.low,
        price_low: prev.high,
        kind: 'FVG_BULL',
      });
    } else if (gap2 > 0) {
      zones.push({
        time_from: candles[i].time,
        time_to: candles[n - 1].time + (candles[n - 1].time - candles[n - 2]?.time || 300),
        price_high: prev.low,
        price_low: next.high,
        kind: 'FVG_BEAR',
      });
    }
  }

  // OB detection: last opposing candle before an impulse (>1.5×ATR)
  for (let i = 3; i < n - 2; i++) {
    const atr = atrs[i] ?? 0;
    if (atr === 0) continue;
    const c = candles[i];
    const next2 = candles[i + 2];
    const bullishImpulse = (next2.close - c.close) > atr * 1.5;
    const bearishImpulse = (c.close - next2.close) > atr * 1.5;
    if (bullishImpulse && c.close < c.open) {
      zones.push({
        time_from: c.time,
        time_to: candles[n - 1].time + 300,
        price_high: c.open,
        price_low: c.low,
        kind: 'OB_BULL',
      });
    } else if (bearishImpulse && c.close > c.open) {
      zones.push({
        time_from: c.time,
        time_to: candles[n - 1].time + 300,
        price_high: c.high,
        price_low: c.open,
        kind: 'OB_BEAR',
      });
    }
  }

  // Keep only the 3 most recent of each type to avoid clutter
  const byKind: Partial<Record<Zone['kind'], Zone[]>> = {};
  for (const z of zones) {
    (byKind[z.kind] ??= []).push(z);
  }
  return Object.values(byKind).flatMap((arr) => arr!.slice(-3));
}

// ── VWAP ────────────────────────────────────────────────────────────────────

function calcVwap(candles: Candle[]): VwapSeries {
  if (candles.length === 0) return [];
  let cumTPV = 0; let cumVol = 0;
  const out: { time: number; value: number }[] = [];
  for (const c of candles) {
    const tp = (c.high + c.low + c.close) / 3;
    const vol = 1; // no volume from engine yet — use tick count proxy
    cumTPV += tp * vol;
    cumVol += vol;
    out.push({ time: c.time, value: cumTPV / cumVol });
  }
  return out;
}

// ── Kill Zone Shaded Ranges ──────────────────────────────────────────────────

function killZoneRanges(candles: Candle[], tfSec: number): ShadedRange[] {
  if (candles.length < 2) return [];
  const ranges: ShadedRange[] = [];
  const first = candles[0].time;
  const last = candles.at(-1)!.time;
  const DAY = 86400;
  // Iterate over days in range
  const startDay = Math.floor(first / DAY) * DAY;
  for (let dayStart = startDay; dayStart <= last + DAY; dayStart += DAY) {
    for (const kz of KILL_ZONES) {
      // kz.start/end are EST minutes; convert to UTC seconds offset
      const kzFromUTC = dayStart + kz.start * 60 + 5 * 3600;
      const kzToUTC = dayStart + kz.end * 60 + 5 * 3600;
      if (kzToUTC < first || kzFromUTC > last + tfSec) continue;
      ranges.push({ time_from: kzFromUTC, time_to: kzToUTC, label: kz.label });
    }
  }
  return ranges;
}

// ── Synthetic History (fallback when engine not live) ──────────────────────

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

// ── Component ────────────────────────────────────────────────────────────────

export default function Charts() {
  const [searchParams, setSearchParams] = useSearchParams();
  const paramSymbol = searchParams.get('symbol');

  const [symbol, setSymbol] = useState(
    paramSymbol && SYMBOLS_13.some((s) => s.name === paramSymbol) ? paramSymbol : SYMBOLS_13[0].name,
  );
  const [tf, setTf] = useState<Timeframe>('M5');
  const [showIndicators, setShowIndicators] = useState(true);
  const [crosshair, setCrosshair] = useState<CrosshairInfo | null>(null);

  const tick = useEngineStore((s) => s.ticks[symbol]);
  const positions = useEngineStore((s) => s.positions);
  const closedTrades = useEngineStore((s) => s.closedTrades);
  const liveMid = tick ? (tick.bid + tick.ask) / 2 : null;

  const [history, setHistory] = useState<Candle[]>([]);
  const tfSec = TF_SECONDS[tf];

  // Sync URL param → state
  useEffect(() => {
    if (paramSymbol && SYMBOLS_13.some((s) => s.name === paramSymbol) && paramSymbol !== symbol) {
      setSymbol(paramSymbol);
    }
  }, [paramSymbol]);

  // Update URL when symbol changes via selector
  const handleSymbolChange = (s: string) => {
    setSymbol(s);
    setSearchParams({ symbol: s }, { replace: true });
  };

  // Build initial history when symbol/tf changes
  useEffect(() => {
    const base = liveMid ?? (
      symbol.includes('JPY') ? 150 :
      symbol.includes('GOLD') || symbol === 'GOLD#' ? 2350 :
      symbol.startsWith('BTC') ? 75000 :
      symbol.startsWith('ETH') ? 3000 : 1.07
    );
    setHistory(syntheticHistory(base, tf, 200));
  }, [symbol, tf]);

  // Update current bar on each tick
  useEffect(() => {
    if (liveMid == null) return;
    const now = Math.floor(Date.now() / 1000);
    const bucket = now - (now % tfSec);
    setHistory((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.time === bucket) {
        return [...prev.slice(0, -1), {
          time: bucket, open: last.open,
          high: Math.max(last.high, liveMid),
          low: Math.min(last.low, liveMid),
          close: liveMid,
        }];
      }
      return [...prev.slice(-199), { time: bucket, open: liveMid, high: liveMid, low: liveMid, close: liveMid }];
    });
  }, [liveMid, tfSec]);

  const closes = useMemo(() => history.map((c) => c.close), [history]);
  const atrs = useMemo(() => calcAtr(history), [history]);

  const emas: EmaSeries[] = useMemo(() => [9, 21, 50].map((p) => ({
    period: p,
    data: calcEma(closes, p).map((v, i) => ({ time: history[i].time, value: v })),
  })), [closes, history]);

  const vwap = useMemo(() => calcVwap(history), [history]);

  const zones = useMemo(() => detectZones(history, atrs), [history, atrs]);

  const shadedRanges = useMemo(() => killZoneRanges(history, tfSec), [history, tfSec]);

  // Trade markers
  const markers: Marker[] = useMemo(() => {
    const out: Marker[] = [];
    const now = Math.floor(Date.now() / 1000);
    const bucket = now - (now % tfSec);

    for (const p of Object.values(positions)) {
      if (p.symbol !== symbol) continue;
      out.push({ time: bucket, price: p.entry, kind: p.direction as 'BUY' | 'SELL', text: `${p.direction} ${p.lot}` });
    }
    for (const t of closedTrades) {
      if (t.symbol !== symbol) continue;
      const age = Date.now() - t.ts;
      if (age > 24 * 3600 * 1000) continue;
      const exitBucket = Math.floor((t.ts / 1000 / tfSec)) * tfSec;
      out.push({ time: exitBucket, price: t.exit, kind: 'EXIT', text: `${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}` });
    }
    return out;
  }, [positions, closedTrades, symbol, tfSec]);

  // Indicator values for right panel
  const rsi = useMemo(() => {
    const vals = calcRsi(closes);
    return vals.at(-1) ?? NaN;
  }, [closes]);
  const atr = useMemo(() => atrs.at(-1) ?? NaN, [atrs]);
  const adx = useMemo(() => calcAdx(history), [history]);
  const macd = useMemo(() => calcMacd(closes), [closes]);
  const last = history.at(-1);
  const isLive = liveMid != null;

  const rsiColor = rsi > 70 ? 'text-accent-red' : rsi < 30 ? 'text-accent-green' : 'text-accent-gold';

  return (
    <section data-testid="page-charts" className="flex h-full flex-col p-4">
      {/* Toolbar */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <select
            data-testid="chart-symbol"
            value={symbol}
            onChange={(e) => handleSymbolChange(e.target.value)}
            className="rounded bg-bg-secondary px-2 py-1 text-sm font-mono text-white border border-white/10"
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
                className={`px-2 py-1 text-xs font-mono transition-colors ${
                  tf === t ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-white/60 hover:text-white'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <span className={`flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
            isLive ? 'bg-accent-green/20 text-accent-green' : 'bg-white/5 text-white/40'
          }`}>
            <span className={`h-1.5 w-1.5 rounded-full ${isLive ? 'animate-pulse bg-accent-green' : 'bg-white/30'}`} />
            {isLive ? 'LIVE' : 'WAITING'}
          </span>
        </div>

        {/* OHLC readout — updates on crosshair */}
        <div className="flex items-center gap-3 font-mono text-xs text-white/60">
          {crosshair?.ohlc ? (
            <>
              <Pill label="O" value={crosshair.ohlc.open.toFixed(5)} />
              <Pill label="H" value={crosshair.ohlc.high.toFixed(5)} color="text-accent-green" />
              <Pill label="L" value={crosshair.ohlc.low.toFixed(5)} color="text-accent-red" />
              <Pill label="C" value={crosshair.ohlc.close.toFixed(5)} />
            </>
          ) : (
            <>
              {emas.map((e, i) => (
                <span key={e.period}>
                  EMA{e.period} <span className={i === 0 ? 'text-accent-cyan' : i === 1 ? 'text-accent-gold' : 'text-[#7c5cff]'}>
                    {e.data.at(-1)?.value.toFixed(5)}
                  </span>
                </span>
              ))}
            </>
          )}
          <button
            onClick={() => setShowIndicators((v) => !v)}
            className="ml-2 rounded bg-bg-secondary px-2 py-0.5 text-[10px] text-white/50 hover:text-white border border-white/10"
          >
            {showIndicators ? '◀ Hide' : 'Indicators ▶'}
          </button>
        </div>
      </div>

      {/* Chart + Indicator panel */}
      <div className="flex flex-1 gap-3 min-h-0">
        <div className="flex-1 min-h-0">
          <CandlestickChart
            candles={history}
            emas={emas}
            markers={markers}
            vwap={vwap}
            zones={zones}
            shadedRanges={shadedRanges}
            height={undefined as any}
            onCrosshair={setCrosshair}
          />
        </div>

        {showIndicators && (
          <aside className="flex w-44 flex-col gap-2 text-[11px]">
            <IndicatorCard title="RSI (14)" value={isNaN(rsi) ? '—' : rsi.toFixed(1)} color={rsiColor}
              sub={rsi > 70 ? 'OVERBOUGHT' : rsi < 30 ? 'OVERSOLD' : 'NEUTRAL'} />
            <IndicatorCard title="ATR (14)" value={isNaN(atr) ? '—' : atr.toFixed(5)} color="text-accent-gold" sub="Volatility" />
            <IndicatorCard title="ADX (14)" value={isNaN(adx) ? '—' : adx.toFixed(1)} color="text-white"
              sub={adx > 25 ? 'TRENDING' : 'RANGING'} />
            <IndicatorCard title="MACD" value={macd.macd.toFixed(5)} color={macd.hist >= 0 ? 'text-accent-green' : 'text-accent-red'}
              sub={`Sig ${macd.signal.toFixed(5)}`} />
            <IndicatorCard title="VWAP" value={vwap.at(-1)?.value.toFixed(5) ?? '—'} color="text-[#ffb800]" sub="Session" />

            <div className="mt-2 rounded-lg border border-white/5 bg-bg-secondary p-2">
              <p className="mb-1 text-[9px] uppercase text-white/40">EMA Stack</p>
              {emas.map((e, i) => {
                const v = e.data.at(-1)?.value;
                const price = last?.close ?? 0;
                const above = price > (v ?? 0);
                return (
                  <div key={e.period} className="flex justify-between py-0.5">
                    <span className={i === 0 ? 'text-accent-cyan' : i === 1 ? 'text-accent-gold' : 'text-[#7c5cff]'}>
                      EMA{e.period}
                    </span>
                    <span className={above ? 'text-accent-green' : 'text-accent-red'}>
                      {above ? '▲' : '▼'}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="rounded-lg border border-white/5 bg-bg-secondary p-2">
              <p className="mb-1 text-[9px] uppercase text-white/40">Zones Detected</p>
              {(['OB_BULL', 'OB_BEAR', 'FVG_BULL', 'FVG_BEAR'] as const).map((k) => {
                const count = zones.filter((z) => z.kind === k).length;
                return (
                  <div key={k} className="flex justify-between py-0.5">
                    <span className={k.includes('BULL') ? 'text-accent-green/70' : 'text-accent-red/70'}>{k}</span>
                    <span className="text-white/60">{count}</span>
                  </div>
                );
              })}
            </div>
          </aside>
        )}
      </div>
    </section>
  );
}

function Pill({ label, value, color = 'text-white' }: { label: string; value: string; color?: string }) {
  return (
    <span>
      <span className="text-white/30">{label} </span>
      <span className={color}>{value}</span>
    </span>
  );
}

function IndicatorCard({ title, value, color, sub }: { title: string; value: string; color: string; sub: string }) {
  return (
    <div className="rounded-lg border border-white/5 bg-bg-secondary p-2">
      <p className="text-[9px] uppercase text-white/40">{title}</p>
      <p className={`mt-0.5 font-mono text-sm font-bold ${color}`}>{value}</p>
      <p className="text-[9px] text-white/40">{sub}</p>
    </div>
  );
}
