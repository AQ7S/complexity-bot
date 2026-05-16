import { useMemo } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';

const SESSIONS = ['Asian', 'London', 'NY'] as const;
type Session = typeof SESSIONS[number];

function getSessionEST(ts: number): Session | null {
  const d = new Date(ts);
  // Convert UTC to EST (UTC-5, no DST adjustment for simplicity)
  const estHour = (d.getUTCHours() - 5 + 24) % 24;
  if (estHour >= 19 || estHour < 2) return 'Asian';
  if (estHour >= 2 && estHour < 10) return 'London';
  if (estHour >= 7 && estHour < 16) return 'NY';
  return null;
}

function cellStyle(winRate: number | null, count: number) {
  if (winRate === null || count === 0) return 'bg-white/[0.03] text-white/20';
  const intensity = Math.min(0.7, Math.max(0.1, Math.abs(winRate - 0.5) * 2));
  if (winRate >= 0.6) return `bg-accent-green/${Math.round(intensity * 100)} text-accent-green`;
  if (winRate >= 0.5) return 'bg-accent-green/10 text-accent-green/70';
  if (winRate <= 0.4) return `bg-accent-red/${Math.round(intensity * 100)} text-accent-red`;
  return 'bg-accent-red/10 text-accent-red/70';
}

export default function SessionPnLHeatmap() {
  const closedTrades = useEngineStore((s) => s.closedTrades);

  const matrix = useMemo(() => {
    const data: Record<string, Record<Session, { wins: number; losses: number; pnl: number }>> = {};

    for (const { name } of SYMBOLS_13) {
      data[name] = { Asian: { wins: 0, losses: 0, pnl: 0 }, London: { wins: 0, losses: 0, pnl: 0 }, NY: { wins: 0, losses: 0, pnl: 0 } };
    }

    for (const trade of closedTrades) {
      const session = trade.ts ? getSessionEST(trade.ts) : null;
      if (!session || !trade.symbol || !data[trade.symbol]) continue;
      const cell = data[trade.symbol][session];
      cell.pnl += trade.pnl;
      if (trade.pnl >= 0) cell.wins += 1;
      else cell.losses += 1;
    }

    return data;
  }, [closedTrades]);

  // Totals per session
  const sessionTotals = SESSIONS.map((session) => {
    let wins = 0, losses = 0, pnl = 0;
    for (const { name } of SYMBOLS_13) {
      wins   += matrix[name]?.[session]?.wins ?? 0;
      losses += matrix[name]?.[session]?.losses ?? 0;
      pnl    += matrix[name]?.[session]?.pnl ?? 0;
    }
    return { session, wins, losses, pnl };
  });

  const totalTrades = closedTrades.length;

  return (
    <section data-testid="page-session-heatmap" className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-hero text-2xl text-accent-cyan">Session P&amp;L Heatmap</h1>
          <p className="mt-1 text-xs text-white/40">
            Win rate per symbol × trading session (EST). {totalTrades} trade{totalTrades !== 1 ? 's' : ''} in history.
          </p>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-white/40">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-5 rounded bg-accent-green/40" />≥60% win
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-5 rounded bg-accent-green/10" />50–60%
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-5 rounded bg-accent-red/10" />40–50%
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-5 rounded bg-accent-red/40" />&lt;40%
          </span>
        </div>
      </div>

      {/* Session totals row */}
      <div className="mb-3 grid grid-cols-3 gap-3">
        {sessionTotals.map(({ session, wins, losses, pnl }) => {
          const total = wins + losses;
          const wr = total > 0 ? wins / total : null;
          return (
            <div key={session} className="rounded-lg border border-white/5 bg-bg-secondary p-3">
              <div className="text-[10px] uppercase tracking-wider text-white/40">{session}</div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className={`text-lg font-mono font-bold ${wr != null && wr >= 0.5 ? 'text-accent-green' : wr != null ? 'text-accent-red' : 'text-white/30'}`}>
                  {wr != null ? `${(wr * 100).toFixed(0)}%` : '—'}
                </span>
                <span className="text-xs text-white/40">{total} trades</span>
              </div>
              <div className={`mt-0.5 text-xs font-mono ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Heatmap table */}
      <div className="flex-1 overflow-auto rounded-lg border border-white/5 bg-bg-secondary">
        <table className="w-full min-w-[480px] text-xs font-mono">
          <thead className="sticky top-0 z-10 bg-bg-secondary">
            <tr className="text-[10px] uppercase tracking-wider text-white/40">
              <th className="px-4 py-3 text-left">Symbol</th>
              {SESSIONS.map((s) => (
                <th key={s} className="px-2 py-3 text-center">{s}</th>
              ))}
              <th className="px-4 py-3 text-right">Total P&amp;L</th>
            </tr>
          </thead>
          <tbody>
            {SYMBOLS_13.map(({ name }) => {
              const symData = matrix[name];
              const totalPnl = SESSIONS.reduce((sum, s) => sum + (symData[s]?.pnl ?? 0), 0);
              const totalW = SESSIONS.reduce((sum, s) => sum + (symData[s]?.wins ?? 0), 0);
              const totalL = SESSIONS.reduce((sum, s) => sum + (symData[s]?.losses ?? 0), 0);
              const totalTrd = totalW + totalL;

              return (
                <tr key={name} className="border-t border-white/5 transition-colors hover:bg-white/[0.02]">
                  <td className="px-4 py-2">
                    <div className="text-white/90">{name}</div>
                    <div className="text-[9px] text-white/30">{totalTrd} trades</div>
                  </td>

                  {SESSIONS.map((session) => {
                    const cell = symData[session];
                    const count = cell.wins + cell.losses;
                    const wr = count > 0 ? cell.wins / count : null;
                    return (
                      <td key={session} className="px-2 py-2 text-center">
                        <div className={`mx-auto flex h-14 w-24 flex-col items-center justify-center rounded text-[10px] ${cellStyle(wr, count)}`}>
                          {count === 0 ? (
                            <span className="text-[9px]">no data</span>
                          ) : (
                            <>
                              <span className="text-sm font-bold">{wr != null ? `${(wr * 100).toFixed(0)}%` : '—'}</span>
                              <span className="text-[9px] opacity-70">{cell.wins}W / {cell.losses}L</span>
                              <span className={`text-[9px] font-mono ${cell.pnl >= 0 ? '' : 'opacity-80'}`}>
                                {cell.pnl >= 0 ? '+' : ''}{cell.pnl.toFixed(0)}
                              </span>
                            </>
                          )}
                        </div>
                      </td>
                    );
                  })}

                  <td className={`px-4 py-2 text-right font-bold ${totalPnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {closedTrades.length === 0 && (
          <p className="py-12 text-center text-xs text-white/30">
            No closed trades yet. Cells populate as trades close.
          </p>
        )}
      </div>
    </section>
  );
}
