import { useEngineStore } from '@/store/engineStore';
import { fmtSignedUsd } from '@/lib/format';

export default function PositionsTable() {
  const positionsMap = useEngineStore((s) => s.positions);
  const positions = Object.values(positionsMap);

  return (
    <section
      data-testid="positions-table"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Open Positions
        </h2>
        <span className="text-xs text-white/40">{positions.length}</span>
      </header>
      {positions.length === 0 ? (
        <p className="py-8 text-center text-xs text-white/40">No open positions.</p>
      ) : (
        <table className="w-full text-left text-xs font-mono">
          <thead className="text-[10px] uppercase text-white/40">
            <tr>
              <th className="py-1">Ticket</th>
              <th>Symbol</th>
              <th>Dir</th>
              <th>Lot</th>
              <th>Entry</th>
              <th>SL</th>
              <th>TP</th>
              <th className="text-right">P&amp;L</th>
              <th className="text-right">R:R</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={p.ticket} className="border-t border-white/5">
                <td className="py-1 text-white/60">{p.ticket}</td>
                <td className="text-white">{p.symbol}</td>
                <td className={p.direction === 'BUY' ? 'text-accent-green' : 'text-accent-red'}>
                  {p.direction}
                </td>
                <td>{p.lot}</td>
                <td>{p.entry}</td>
                <td>{p.sl}</td>
                <td>{p.tp}</td>
                <td className={`text-right ${(p.pnl ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {p.pnl != null ? fmtSignedUsd(p.pnl) : '—'}
                </td>
                <td className="text-right">{p.rr_current?.toFixed(2) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
