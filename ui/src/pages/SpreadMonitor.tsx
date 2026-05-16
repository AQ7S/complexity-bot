import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';

const WARN_MULT = 2.0;   // highlight if spread > 2× rolling avg
const CRIT_MULT = 3.0;   // critical if > 3× rolling avg

function spreadColor(ratio: number) {
  if (ratio >= CRIT_MULT) return 'text-accent-red';
  if (ratio >= WARN_MULT) return 'text-accent-gold';
  return 'text-accent-green';
}

function ratioBar(ratio: number) {
  const pct = Math.min(100, (ratio / 4) * 100);
  const color = ratio >= CRIT_MULT ? 'bg-accent-red' : ratio >= WARN_MULT ? 'bg-accent-gold' : 'bg-accent-green';
  return (
    <div className="h-1.5 w-full rounded-full bg-bg-tertiary">
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function SpreadMonitor() {
  const ticks = useEngineStore((s) => s.ticks);
  const tickHistory = useEngineStore((s) => s.tickHistory);

  const rows = SYMBOLS_13.map(({ name }) => {
    const live = ticks[name];
    const history = tickHistory[name] ?? [];

    const currentSpread = live ? live.spread : null;

    // Rolling average over last N samples (pips or raw)
    const avgSpread = history.length > 0
      ? history.reduce((sum, h) => sum + h.spread, 0) / history.length
      : null;

    const ratio = currentSpread != null && avgSpread && avgSpread > 0
      ? currentSpread / avgSpread
      : null;

    const status = ratio == null ? 'NO_DATA'
      : ratio >= CRIT_MULT ? 'CRITICAL'
      : ratio >= WARN_MULT ? 'WIDENED'
      : 'NORMAL';

    return { name, currentSpread, avgSpread, ratio, status };
  });

  const widened = rows.filter((r) => r.status === 'WIDENED' || r.status === 'CRITICAL').length;
  const critical = rows.filter((r) => r.status === 'CRITICAL').length;

  return (
    <section data-testid="page-spread-monitor" className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-hero text-2xl text-accent-cyan">Spread Monitor</h1>
          <p className="mt-1 text-xs text-white/40">
            Live spread vs {Math.round((rows[0]?.avgSpread ?? 0))} sample rolling average.
            Warn at {WARN_MULT}×, critical at {CRIT_MULT}×.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {critical > 0 && (
            <span className="animate-pulse rounded-full bg-accent-red/20 px-3 py-1 font-mono text-accent-red">
              {critical} CRITICAL
            </span>
          )}
          {widened > 0 && (
            <span className="rounded-full bg-accent-gold/20 px-3 py-1 font-mono text-accent-gold">
              {widened} WIDENED
            </span>
          )}
          {widened === 0 && critical === 0 && (
            <span className="rounded-full bg-accent-green/15 px-3 py-1 font-mono text-accent-green">
              All Normal
            </span>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto rounded-lg border border-white/5 bg-bg-secondary">
        <table className="w-full min-w-[600px] text-xs font-mono">
          <thead className="sticky top-0 z-10 bg-bg-secondary">
            <tr className="text-[10px] uppercase tracking-wider text-white/40">
              <th className="px-4 py-3 text-left">Symbol</th>
              <th className="px-4 py-3 text-right">Live Spread</th>
              <th className="px-4 py-3 text-right">Avg Spread</th>
              <th className="px-4 py-3 text-right">Ratio</th>
              <th className="px-4 py-3 text-left">Spread Bar</th>
              <th className="px-4 py-3 text-center">Status</th>
              <th className="px-4 py-3 text-right">Bid</th>
              <th className="px-4 py-3 text-right">Ask</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ name, currentSpread, avgSpread, ratio, status }) => {
              const live = ticks[name];
              const rowBg = status === 'CRITICAL' ? 'bg-accent-red/5'
                : status === 'WIDENED' ? 'bg-accent-gold/5'
                : '';
              return (
                <tr
                  key={name}
                  data-testid={`spread-row-${name}`}
                  className={`border-t border-white/5 transition-colors ${rowBg}`}
                >
                  <td className="px-4 py-2 text-white/90">{name}</td>

                  {/* Live spread */}
                  <td className={`px-4 py-2 text-right font-bold ${ratio != null ? spreadColor(ratio) : 'text-white/30'}`}>
                    {currentSpread != null
                      ? currentSpread.toFixed(currentSpread < 1 ? 5 : 2)
                      : '—'}
                  </td>

                  {/* Avg spread */}
                  <td className="px-4 py-2 text-right text-white/50">
                    {avgSpread != null
                      ? avgSpread.toFixed(avgSpread < 1 ? 5 : 2)
                      : '—'}
                  </td>

                  {/* Ratio */}
                  <td className={`px-4 py-2 text-right font-bold ${ratio != null ? spreadColor(ratio) : 'text-white/30'}`}>
                    {ratio != null ? `${ratio.toFixed(2)}×` : '—'}
                  </td>

                  {/* Bar */}
                  <td className="px-4 py-2 w-32">
                    {ratio != null ? ratioBar(ratio) : (
                      <div className="h-1.5 w-full rounded-full bg-bg-tertiary" />
                    )}
                  </td>

                  {/* Status badge */}
                  <td className="px-4 py-2 text-center">
                    {status === 'NO_DATA' ? (
                      <span className="text-white/20">—</span>
                    ) : (
                      <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${
                        status === 'CRITICAL' ? 'bg-accent-red/20 text-accent-red'
                        : status === 'WIDENED' ? 'bg-accent-gold/20 text-accent-gold'
                        : 'bg-accent-green/15 text-accent-green'
                      }`}>
                        {status}
                      </span>
                    )}
                  </td>

                  {/* Bid */}
                  <td className="px-4 py-2 text-right text-white/60">
                    {live ? live.bid.toFixed(live.bid > 100 ? 2 : 5) : '—'}
                  </td>

                  {/* Ask */}
                  <td className="px-4 py-2 text-right text-white/60">
                    {live ? live.ask.toFixed(live.ask > 100 ? 2 : 5) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {Object.keys(ticks).length === 0 && (
          <p className="py-12 text-center text-xs text-white/30">
            No live ticks yet — connect engine to populate spread data.
          </p>
        )}
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center gap-6 text-[10px] text-white/40">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-green" />Normal (&lt;{WARN_MULT}×)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-gold" />Widened (≥{WARN_MULT}×)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-red" />Critical (≥{CRIT_MULT}×)
        </span>
        <span>Spread values in price units as reported by MT5.</span>
      </div>
    </section>
  );
}
