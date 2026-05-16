import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13, TIMEFRAMES } from '@/lib/constants';

const TF_COLS = TIMEFRAMES.slice(0, 4);

const BG: Record<'BUY' | 'SELL' | 'HOLD', string> = {
  BUY:  'bg-accent-green/20 border-accent-green/40 text-accent-green',
  SELL: 'bg-accent-red/20 border-accent-red/40 text-accent-red',
  HOLD: 'bg-white/5 border-white/10 text-white/30',
};

const LABEL: Record<'BUY' | 'SELL' | 'HOLD', string> = {
  BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD',
};

function confBar(n: number) {
  return (
    <div className="mt-0.5 flex gap-px justify-center">
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          className={`h-0.5 w-2 rounded-full ${i <= n ? 'bg-current opacity-80' : 'bg-white/10'}`}
        />
      ))}
    </div>
  );
}

function timeAgo(ts: number) {
  const d = Date.now() - ts;
  if (d < 60_000) return `${Math.round(d / 1000)}s`;
  if (d < 3_600_000) return `${Math.round(d / 60_000)}m`;
  return `${Math.round(d / 3_600_000)}h`;
}

export default function SignalScanner() {
  const signals = useEngineStore((s) => s.signals);
  const [filter, setFilter] = useState<'ALL' | 'BUY' | 'SELL'>('ALL');
  const [minConf, setMinConf] = useState(1);
  const navigate = useNavigate();

  const getCell = (symbol: string, tf: string) =>
    signals.find((s) => s.symbol === symbol && s.timeframe === tf) ?? null;

  const buyCount  = signals.filter((s) => s.direction === 'BUY').length;
  const sellCount = signals.filter((s) => s.direction === 'SELL').length;
  const highConf  = signals.filter((s) => s.confluence >= 4).length;

  return (
    <section data-testid="page-signal-scanner" className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-hero text-2xl text-accent-cyan">Signal Scanner</h1>
          <p className="mt-1 text-xs text-white/40">
            Latest signal per symbol × timeframe. Bars below each cell = confluence /5.
          </p>
        </div>
        {/* Summary chips */}
        <div className="flex items-center gap-2 text-xs">
          <span className="rounded-full bg-accent-green/15 px-3 py-1 font-mono text-accent-green">
            {buyCount} BUY
          </span>
          <span className="rounded-full bg-accent-red/15 px-3 py-1 font-mono text-accent-red">
            {sellCount} SELL
          </span>
          <span className="rounded-full bg-accent-gold/15 px-3 py-1 font-mono text-accent-gold">
            {highConf} ≥4/5
          </span>
        </div>
      </div>

      {/* Filter bar */}
      <div className="mb-3 flex items-center gap-3">
        <span className="text-xs text-white/40">Filter:</span>
        {(['ALL', 'BUY', 'SELL'] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`rounded px-3 py-1 text-xs font-bold transition-colors ${
              filter === f
                ? f === 'BUY'  ? 'bg-accent-green text-bg-primary'
                : f === 'SELL' ? 'bg-accent-red text-bg-primary'
                : 'bg-accent-cyan text-bg-primary'
                : 'bg-white/5 text-white/50 hover:text-white'
            }`}
          >
            {f}
          </button>
        ))}
        <span className="ml-4 text-xs text-white/40">Min confluence:</span>
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => setMinConf(n)}
            className={`h-6 w-6 rounded text-xs font-mono transition-colors ${
              minConf === n ? 'bg-accent-cyan text-bg-primary' : 'bg-white/5 text-white/50 hover:text-white'
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {/* Matrix */}
      <div className="flex-1 overflow-auto rounded-lg border border-white/5 bg-bg-secondary">
        <table className="w-full min-w-[520px] text-xs font-mono">
          <thead className="sticky top-0 z-10 bg-bg-secondary">
            <tr className="text-white/40">
              <th className="px-4 py-2 text-left text-[10px] uppercase tracking-wider">Symbol</th>
              {TF_COLS.map((tf) => (
                <th key={tf} className="w-28 px-2 py-2 text-center text-[10px] uppercase tracking-wider">
                  {tf}
                </th>
              ))}
              <th className="px-4 py-2 text-center text-[10px] uppercase tracking-wider">Alert</th>
            </tr>
          </thead>
          <tbody>
            {SYMBOLS_13.map(({ name, kind }) => {
              const rowCells = TF_COLS.map((tf) => getCell(name, tf));
              const rowVisible = filter === 'ALL' || rowCells.some(
                (c) => c && c.direction === filter && c.confluence >= minConf
              );
              if (!rowVisible) return null;

              return (
                <tr
                  key={name}
                  data-testid={`scanner-row-${name}`}
                  className="border-t border-white/5 transition-colors hover:bg-white/[0.02]"
                >
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <span className="text-white/90">{name}</span>
                      <span className="rounded bg-bg-tertiary px-1 py-px text-[8px] uppercase text-white/30">
                        {kind}
                      </span>
                    </div>
                  </td>

                  {TF_COLS.map((tf) => {
                    const hit = getCell(name, tf);
                    const hidden = hit && (
                      (filter !== 'ALL' && hit.direction !== filter) ||
                      hit.confluence < minConf
                    );
                    if (!hit || hidden) {
                      return (
                        <td key={tf} className="px-2 py-2 text-center">
                          <div className="mx-auto flex h-10 w-20 items-center justify-center rounded border border-white/5 bg-white/[0.02] text-white/20">
                            —
                          </div>
                        </td>
                      );
                    }
                    return (
                      <td key={tf} className="px-2 py-2 text-center">
                        <div
                          title={`${name} ${tf}: ${hit.direction} · ${hit.confluence}/5 · ${timeAgo(hit.ts)} ago`}
                          className={`mx-auto flex h-10 w-20 cursor-pointer flex-col items-center justify-center rounded border text-[10px] font-bold transition-opacity hover:opacity-80 ${BG[hit.direction]}`}
                          onClick={() => navigate(`/charts?symbol=${name}`)}
                        >
                          <span>{LABEL[hit.direction]}</span>
                          {confBar(hit.confluence)}
                        </div>
                      </td>
                    );
                  })}

                  <td className="px-4 py-2 text-center">
                    {rowCells.some((c) => c && c.confluence >= 4 && c.direction !== 'HOLD') ? (
                      <span className="animate-pulse rounded-full bg-accent-gold/20 px-2 py-0.5 text-[9px] font-bold text-accent-gold">
                        STRONG
                      </span>
                    ) : rowCells.some((c) => c && c.direction !== 'HOLD') ? (
                      <span className="text-[9px] text-white/30">—</span>
                    ) : (
                      <span className="text-[9px] text-white/20">no sig</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {signals.length === 0 && (
          <p className="py-12 text-center text-xs text-white/30">
            Waiting for engine signals… (connect engine or signals appear on each new bar)
          </p>
        )}
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center gap-4 text-[10px] text-white/40">
        <span>Cell opacity = age of signal.</span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-green" />BUY
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-red" />SELL
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-white/20" />HOLD
        </span>
        <span>Bars = confluence /5. Click cell → open chart.</span>
      </div>
    </section>
  );
}
