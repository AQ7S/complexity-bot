import { useEffect, useMemo, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis, Tooltip } from 'recharts';
import { useEngineStore } from '@/store/engineStore';
import { sendCommand } from '@/hooks/useEngineSocket';
import { fmtSignedUsd, fmtPct } from '@/lib/format';
import { downloadCsv, toCsv } from '@/lib/csv';
import type { TradeRow } from '@/types/ipc-messages';

type Filter = { symbol: string; reason: string };

function metrics(rows: TradeRow[]) {
  const closed = rows.filter((r) => r.close_time && r.pnl != null) as Array<TradeRow & { pnl: number }>;
  const trades = closed.length;
  const wins = closed.filter((r) => r.pnl > 0).length;
  const losses = closed.filter((r) => r.pnl < 0).length;
  const net = closed.reduce((s, r) => s + r.pnl, 0);
  const winRate = trades ? wins / trades : 0;
  const rrs = closed.map((r) => r.rr_achieved ?? 0).filter((v) => Number.isFinite(v));
  const avgRR = rrs.length ? rrs.reduce((s, v) => s + v, 0) / rrs.length : 0;
  // Sharpe over per-trade pnl (dimensionless, gross): mean / stdev × √trades
  const mean = trades ? net / trades : 0;
  const variance = trades
    ? closed.reduce((s, r) => s + (r.pnl - mean) ** 2, 0) / trades : 0;
  const stdev = Math.sqrt(variance);
  const sharpe = stdev > 0 ? (mean / stdev) * Math.sqrt(trades) : 0;
  // Max drawdown over equity curve.
  let peak = 0, eq = 0, maxDD = 0;
  for (const r of closed) {
    eq += r.pnl;
    peak = Math.max(peak, eq);
    maxDD = Math.max(maxDD, peak - eq);
  }
  return { trades, wins, losses, net, winRate, avgRR, sharpe, maxDD };
}

function equityCurve(rows: TradeRow[]) {
  let eq = 0;
  return rows
    .filter((r) => r.close_time && r.pnl != null)
    .sort((a, b) => (a.close_time! < b.close_time! ? -1 : 1))
    .map((r) => { eq += r.pnl ?? 0; return { ts: r.close_time, equity: eq }; });
}

export default function TradeJournal() {
  const trades = useEngineStore((s) => s.tradesHistory);
  const [filter, setFilter] = useState<Filter>({ symbol: '', reason: '' });
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    void sendCommand('cmd_get_trades', { limit: 500 });
  }, []);

  const filtered = useMemo(() => trades.filter((t) =>
    (!filter.symbol || t.symbol === filter.symbol) &&
    (!filter.reason || t.close_reason === filter.reason)
  ), [trades, filter]);
  const m = metrics(filtered);
  const curve = equityCurve(filtered);

  const symbols = Array.from(new Set(trades.map((t) => t.symbol))).sort();
  const reasons = Array.from(new Set(trades.map((t) => t.close_reason).filter(Boolean))) as string[];

  const onExport = () => {
    const csv = toCsv(filtered, [
      'mt5_ticket', 'symbol', 'direction', 'entry_price', 'exit_price',
      'lot_size', 'sl', 'tp', 'pnl', 'rr_achieved', 'open_time', 'close_time',
      'close_reason', 'signal_confluence', 'claude_decision', 'claude_confidence',
    ]);
    downloadCsv(`trades-${new Date().toISOString().slice(0, 10)}.csv`, csv);
  };

  return (
    <section data-testid="page-trade-journal" className="space-y-4 p-6">
      <header className="flex items-center justify-between">
        <h1 className="font-hero text-2xl text-accent-cyan">Trade Journal</h1>
        <div className="flex items-center gap-2">
          <select
            data-testid="filter-symbol"
            value={filter.symbol}
            onChange={(e) => setFilter((f) => ({ ...f, symbol: e.target.value }))}
            className="rounded bg-bg-secondary px-2 py-1 text-xs font-mono"
          >
            <option value="">All symbols</option>
            {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select
            data-testid="filter-reason"
            value={filter.reason}
            onChange={(e) => setFilter((f) => ({ ...f, reason: e.target.value }))}
            className="rounded bg-bg-secondary px-2 py-1 text-xs font-mono"
          >
            <option value="">All reasons</option>
            {reasons.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <button
            type="button"
            onClick={onExport}
            data-testid="export-csv"
            className="rounded bg-accent-cyan/20 px-3 py-1 text-xs text-accent-cyan hover:bg-accent-cyan/30"
          >
            Export CSV
          </button>
        </div>
      </header>

      <div data-testid="metric-cards" className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Trades" value={String(m.trades)} />
        <Metric label="Win Rate" value={fmtPct(m.winRate, 1)} accent={m.winRate >= 0.5 ? 'text-accent-green' : 'text-accent-red'} />
        <Metric label="Avg R:R" value={m.avgRR.toFixed(2)} />
        <Metric label="Sharpe" value={m.sharpe.toFixed(2)} accent={m.sharpe >= 0 ? 'text-accent-green' : 'text-accent-red'} />
        <Metric label="Max DD" value={fmtSignedUsd(-m.maxDD)} accent="text-accent-red" />
      </div>

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">Equity Curve</h2>
        <div className="h-40 w-full">
          {curve.length < 2 ? (
            <p className="flex h-full items-center justify-center text-xs text-white/40">
              Need ≥2 closed trades.
            </p>
          ) : (
            <ResponsiveContainer>
              <LineChart data={curve}>
                <YAxis dataKey="equity" hide domain={['dataMin', 'dataMax']} />
                <Tooltip
                  contentStyle={{ background: '#161b2c', border: 'none' }}
                  formatter={(v: any) => fmtSignedUsd(Number(v))}
                />
                <Line type="monotone" dataKey="equity" dot={false}
                  stroke="#00d4ff" strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section data-testid="trades-table" className="overflow-x-auto rounded-lg border border-white/5 bg-bg-secondary p-2">
        {filtered.length === 0 ? (
          <p className="py-8 text-center text-sm text-white/40">No trades to show.</p>
        ) : (
          <table className="w-full text-left text-xs font-mono">
            <thead className="text-[10px] uppercase text-white/40">
              <tr>
                <th className="px-2 py-1">Ticket</th><th>Sym</th><th>Dir</th>
                <th>Entry</th><th>Exit</th><th className="text-right">P&amp;L</th>
                <th className="text-right">R:R</th><th>Reason</th><th>Closed</th><th />
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => (
                <FragmentRow
                  key={t.id}
                  row={t}
                  expanded={expanded === t.id}
                  onToggle={() => setExpanded((e) => (e === t.id ? null : t.id))}
                />
              ))}
            </tbody>
          </table>
        )}
      </section>
    </section>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-white/5 bg-bg-secondary p-3">
      <p className="text-[10px] uppercase tracking-wider text-white/50">{label}</p>
      <p className={`font-mono text-xl ${accent ?? 'text-white'}`}>{value}</p>
    </div>
  );
}

function FragmentRow({ row, expanded, onToggle }:
  { row: TradeRow; expanded: boolean; onToggle: () => void }) {
  const positive = (row.pnl ?? 0) >= 0;
  return (
    <>
      <tr
        data-testid={`trade-row-${row.id}`}
        className="cursor-pointer border-t border-white/5 hover:bg-bg-tertiary/40"
        onClick={onToggle}
      >
        <td className="px-2 py-1 text-white/60">{row.mt5_ticket}</td>
        <td className="text-white">{row.symbol}</td>
        <td className={row.direction === 'BUY' ? 'text-accent-green' : 'text-accent-red'}>
          {row.direction}
        </td>
        <td>{row.entry_price}</td>
        <td>{row.exit_price ?? '—'}</td>
        <td className={`text-right ${positive ? 'text-accent-green' : 'text-accent-red'}`}>
          {row.pnl != null ? fmtSignedUsd(row.pnl) : '—'}
        </td>
        <td className="text-right">{row.rr_achieved?.toFixed(2) ?? '—'}</td>
        <td className="text-white/60">{row.close_reason ?? '—'}</td>
        <td className="text-white/40">{row.close_time?.slice(0, 16).replace('T', ' ') ?? '—'}</td>
        <td className="text-white/40">{expanded ? '▾' : '▸'}</td>
      </tr>
      {expanded && row.claude_reasoning && (
        <tr className="border-t border-white/5 bg-bg-tertiary/30" data-testid={`trade-reasoning-${row.id}`}>
          <td colSpan={10} className="px-3 py-2 text-xs text-white/70">
            <span className="font-bold text-accent-purple">Claude {row.claude_decision}</span>
            <span className="ml-2 text-white/50">conf {row.claude_confidence}%</span>
            <p className="mt-1 italic">{row.claude_reasoning}</p>
          </td>
        </tr>
      )}
    </>
  );
}
