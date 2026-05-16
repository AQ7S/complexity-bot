import { useEffect, useRef, useState } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip, ReferenceLine } from 'recharts';

const MAX_POINTS = 360;
const HEARTBEAT_MS = 5_000;

type Point = { t: number; equity: number; drawdown_pct: number };

export default function DrawdownChart() {
  const account = useEngineStore((s) => s.account);
  const sessionStartEquity = useEngineStore((s) => s.sessionStartEquity);
  const sessionStartTs = useEngineStore((s) => s.sessionStartTs);
  const buf = useRef<Point[]>([]);
  const [snapshot, setSnapshot] = useState<Point[]>([]);

  useEffect(() => {
    if (!account) return;
    buf.current.push({ t: Date.now(), equity: account.equity, drawdown_pct: account.drawdown_pct });
    if (buf.current.length > MAX_POINTS) buf.current.shift();
    setSnapshot([...buf.current]);
  }, [account]);

  useEffect(() => {
    const id = setInterval(() => {
      const last = buf.current.at(-1);
      if (!last) return;
      buf.current.push({ t: Date.now(), equity: last.equity, drawdown_pct: last.drawdown_pct });
      if (buf.current.length > MAX_POINTS) buf.current.shift();
      setSnapshot([...buf.current]);
    }, HEARTBEAT_MS);
    return () => clearInterval(id);
  }, []);

  const baseline = sessionStartEquity ?? account?.equity ?? 0;
  const current = account?.equity ?? baseline;
  const sessionDelta = current - baseline;
  const sessionPct = baseline > 0 ? (sessionDelta / baseline) * 100 : 0;

  const data: Point[] = snapshot.length === 0 && account ? [
    { t: sessionStartTs, equity: baseline, drawdown_pct: 0 },
    { t: Date.now(), equity: baseline, drawdown_pct: 0 },
  ] : snapshot;

  return (
    <section
      data-testid="drawdown-chart"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Equity / Drawdown
        </h2>
        <div className="flex items-center gap-3 text-[10px]">
          <span className={`font-mono ${sessionDelta >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {sessionDelta >= 0 ? '+' : ''}{sessionDelta.toFixed(2)} ({sessionPct.toFixed(2)}%)
          </span>
          {account && (
            <span className={`font-mono ${account.drawdown_pct > 0 ? 'text-accent-red' : 'text-white/40'}`}>
              DD {(account.drawdown_pct * 100).toFixed(2)}%
            </span>
          )}
        </div>
      </header>
      <div className="h-32 w-full">
        {!account ? (
          <p className="flex h-full items-center justify-center text-xs text-white/40">
            Waiting for account snapshot…
          </p>
        ) : (
          <ResponsiveContainer>
            <LineChart data={data}>
              <XAxis dataKey="t" hide />
              <YAxis dataKey="equity" hide domain={[Math.min(baseline * 0.98, ...data.map((d) => d.equity)), Math.max(baseline * 1.02, ...data.map((d) => d.equity))]} />
              <Tooltip
                contentStyle={{ background: '#161b2c', border: 'none' }}
                labelFormatter={(v) => new Date(v as number).toLocaleTimeString()}
                formatter={(v: any, n) => [Number(v).toFixed(2), n === 'equity' ? 'Equity' : 'DD%']}
              />
              <ReferenceLine y={baseline} stroke="#475569" strokeDasharray="3 3" />
              <Line
                type="monotone" dataKey="equity" dot={false}
                stroke={sessionDelta >= 0 ? '#00ff88' : '#ff3b6b'}
                strokeWidth={1.5} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="mt-1 flex justify-between text-[9px] font-mono text-white/40">
        <span>Baseline ${baseline.toFixed(2)}</span>
        <span>Now ${current.toFixed(2)}</span>
      </div>
    </section>
  );
}
