import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from 'recharts';
import { sendCommand } from '@/hooks/useEngineSocket';
import { SYMBOLS_13, TIMEFRAMES } from '@/lib/constants';
import { useEngineStore } from '@/store/engineStore';

type BacktestResult = {
  equity_curve: { ts: number; equity: number; drawdown_pct: number }[];
  trades: { ts: number; symbol: string; direction: 'BUY' | 'SELL'; pnl: number; rr: number }[];
  summary: {
    total_trades: number; wins: number; losses: number; win_rate: number;
    net_pnl: number; max_drawdown_pct: number; sharpe: number; avg_rr: number;
  };
};

export default function BacktestRunner() {
  const [symbol, setSymbol]     = useState('EURUSD');
  const [tf, setTf]             = useState('M5');
  const [from, setFrom]         = useState(() => {
    const d = new Date();
    d.setMonth(d.getMonth() - 3);
    return d.toISOString().slice(0, 10);
  });
  const [to, setTo]             = useState(() => new Date().toISOString().slice(0, 10));
  const [riskPct, setRiskPct]   = useState('2.0');
  const [minConf, setMinConf]   = useState('3');
  const [running, setRunning]   = useState(false);
  const [result, setResult]     = useState<BacktestResult | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const liveResult = useEngineStore((s) => s.lastBacktestResult);

  useEffect(() => {
    if (!liveResult) return;
    setRunning(false);
    if (liveResult.error) {
      setError(liveResult.error);
      return;
    }
    setError(null);
    setResult({
      equity_curve: [],
      trades: [],
      summary: {
        total_trades: liveResult.total_trades,
        wins: liveResult.wins,
        losses: liveResult.losses,
        win_rate: liveResult.win_rate,
        net_pnl: liveResult.net_pnl_usd,
        max_drawdown_pct: liveResult.max_drawdown_pct / 100,
        sharpe: liveResult.sharpe,
        avg_rr: liveResult.avg_r_multiple,
      },
    });
  }, [liveResult]);

  async function runBacktest() {
    setRunning(true);
    setError(null);
    setResult(null);

    const ok = await sendCommand('cmd_run_backtest', {
      symbol,
      from,
      to,
      strategy_config: {
        timeframe: tf,
        risk_pct: parseFloat(riskPct),
        min_confluence: parseInt(minConf, 10),
      },
    });

    if (!ok) {
      setError('Engine not connected. Start the engine and try again.');
      setRunning(false);
      return;
    }

    setTimeout(() => {
      setRunning((wasRunning) => {
        if (wasRunning) {
          setError('Backtest timed out. Engine may still be processing — check logs.');
          return false;
        }
        return wasRunning;
      });
    }, 120_000);
  }

  const curve = result?.equity_curve ?? [];
  const summary = result?.summary;
  const startEquity = curve[0]?.equity ?? 10000;

  return (
    <section data-testid="page-backtest-runner" className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4">
        <h1 className="font-hero text-2xl text-accent-cyan">Backtest Runner</h1>
        <p className="mt-1 text-xs text-white/40">
          Run a historical backtest on stored DuckDB bars. Results streamed from engine.
        </p>
      </div>

      {/* Config form */}
      <div className="mb-4 rounded-lg border border-white/5 bg-bg-secondary p-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          {/* Symbol */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            >
              {SYMBOLS_13.map(({ name }) => <option key={name} value={name}>{name}</option>)}
            </select>
          </div>

          {/* Timeframe */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">Timeframe</label>
            <select
              value={tf}
              onChange={(e) => setTf(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            >
              {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          {/* From */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">From</label>
            <input
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            />
          </div>

          {/* To */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">To</label>
            <input
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            />
          </div>

          {/* Risk % */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">Risk %</label>
            <input
              type="number"
              value={riskPct}
              min="0.1"
              max="5"
              step="0.1"
              onChange={(e) => setRiskPct(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            />
          </div>

          {/* Min confluence */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-white/40">Min Confluence</label>
            <input
              type="number"
              value={minConf}
              min="1"
              max="5"
              onChange={(e) => setMinConf(e.target.value)}
              className="w-full rounded bg-bg-tertiary px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
            />
          </div>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            type="button"
            onClick={runBacktest}
            disabled={running}
            className="rounded bg-accent-cyan px-6 py-2 text-xs font-bold text-bg-primary transition-opacity hover:opacity-80 disabled:opacity-40"
          >
            {running ? 'Running…' : 'Run Backtest'}
          </button>
          {running && (
            <span className="animate-pulse text-xs text-white/40">
              Waiting for engine response…
            </span>
          )}
          {error && <span className="text-xs text-accent-red">{error}</span>}
        </div>
      </div>

      {/* Cost transparency banner */}
      {liveResult && !liveResult.error && (
        <div className="mb-3 rounded-lg border border-accent-cyan/30 bg-bg-secondary p-3 text-xs text-white/70">
          <span className="mr-3 font-bold uppercase tracking-wider text-accent-cyan">Costs applied</span>
          spread {liveResult.spread_pips_used.toFixed(2)} pips ·
          slippage {liveResult.slippage_pips_used.toFixed(2)} pips ·
          swap long {liveResult.swap_long_pips_used.toFixed(2)} pips/night ·
          swap short {liveResult.swap_short_pips_used.toFixed(2)} pips/night
          {liveResult.sharpe < 0.5 && (
            <span className="ml-3 rounded bg-accent-red/30 px-2 py-0.5 text-[10px] font-bold uppercase text-accent-red">
              Sharpe {liveResult.sharpe.toFixed(2)} &lt; 0.5 — edge insufficient
            </span>
          )}
        </div>
      )}

      {/* Results */}
      {result ? (
        <>
          {/* Summary cards */}
          {summary && (
            <div className="mb-4 grid grid-cols-4 gap-3 sm:grid-cols-8">
              {([
                { label: 'Trades',       value: summary.total_trades,             fmt: (v: number) => String(v) },
                { label: 'Win Rate',     value: summary.win_rate * 100,           fmt: (v: number) => `${v.toFixed(1)}%` },
                { label: 'Net P&L',      value: summary.net_pnl,                  fmt: (v: number) => `${v >= 0 ? '+' : ''}$${v.toFixed(0)}` },
                { label: 'Max DD',       value: summary.max_drawdown_pct * 100,   fmt: (v: number) => `${v.toFixed(1)}%` },
                { label: 'Sharpe',       value: summary.sharpe,                   fmt: (v: number) => v.toFixed(2) },
                { label: 'Avg R:R',      value: summary.avg_rr,                   fmt: (v: number) => v.toFixed(2) },
                { label: 'Wins',         value: summary.wins,                     fmt: (v: number) => String(v) },
                { label: 'Losses',       value: summary.losses,                   fmt: (v: number) => String(v) },
              ] as const).map(({ label, value, fmt: fmtFn }) => (
                <div key={label} className="rounded-lg border border-white/5 bg-bg-secondary p-3">
                  <div className="text-[9px] uppercase tracking-wider text-white/40">{label}</div>
                  <div className={`mt-1 font-mono text-sm font-bold ${
                    label === 'Net P&L'  ? (value >= 0 ? 'text-accent-green' : 'text-accent-red')
                    : label === 'Max DD' ? 'text-accent-red'
                    : label === 'Win Rate' ? (value >= 50 ? 'text-accent-green' : 'text-accent-red')
                    : 'text-white'
                  }`}>
                    {fmtFn(value as number)}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Equity curve */}
          <div className="flex-1 rounded-lg border border-white/5 bg-bg-secondary p-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-wider text-white/60">Equity Curve</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve.map((p) => ({
                t: new Date(p.ts).toLocaleDateString(),
                equity: p.equity,
                dd: -(p.drawdown_pct * 100),
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2238" />
                <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 9 }} />
                <YAxis yAxisId="eq" orientation="left" tick={{ fill: '#64748b', fontSize: 9 }} domain={['auto', 'auto']} />
                <YAxis yAxisId="dd" orientation="right" tick={{ fill: '#64748b', fontSize: 9 }} domain={['auto', 0]} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 11 }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <ReferenceLine yAxisId="eq" y={startEquity} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
                <Line yAxisId="eq" type="monotone" dataKey="equity" stroke="#00ff88" dot={false} strokeWidth={1.5} name="Equity" />
                <Line yAxisId="dd" type="monotone" dataKey="dd" stroke="#ff3b6b" dot={false} strokeWidth={1} name="Drawdown %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="flex flex-1 items-center justify-center rounded-lg border border-white/5 bg-bg-secondary">
          <div className="text-center">
            <div className="text-4xl text-white/10">▷</div>
            <p className="mt-2 text-xs text-white/30">
              Configure parameters above and click Run Backtest.
            </p>
            <p className="mt-1 text-[10px] text-white/20">
              Requires engine running with DuckDB bars loaded.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
