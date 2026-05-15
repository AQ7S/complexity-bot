import { useEffect, useRef, useState } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip } from 'recharts';

const MAX_POINTS = 240;   // 8 minutes at 2s cadence

type Point = { t: number; equity: number; drawdown_pct: number };

export default function DrawdownChart() {
  const account = useEngineStore((s) => s.account);
  const buf = useRef<Point[]>([]);
  const [snapshot, setSnapshot] = useState<Point[]>([]);

  useEffect(() => {
    if (!account) return;
    buf.current.push({ t: Date.now(), equity: account.equity, drawdown_pct: account.drawdown_pct });
    if (buf.current.length > MAX_POINTS) buf.current.shift();
    setSnapshot([...buf.current]);
  }, [account]);

  return (
    <section
      data-testid="drawdown-chart"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Equity / Drawdown
        </h2>
        {account && (
          <span className={`font-mono text-xs ${account.drawdown_pct > 0 ? 'text-accent-red' : 'text-white/40'}`}>
            DD {(account.drawdown_pct * 100).toFixed(2)}%
          </span>
        )}
      </header>
      <div className="h-32 w-full">
        {snapshot.length < 2 ? (
          <p className="flex h-full items-center justify-center text-xs text-white/40">
            Waiting for account snapshots…
          </p>
        ) : (
          <ResponsiveContainer>
            <LineChart data={snapshot}>
              <XAxis dataKey="t" hide />
              <YAxis dataKey="equity" hide domain={['dataMin', 'dataMax']} />
              <Tooltip
                contentStyle={{ background: '#161b2c', border: 'none' }}
                labelFormatter={(v) => new Date(v as number).toLocaleTimeString()}
                formatter={(v: any, n) => [Number(v).toFixed(2), n === 'equity' ? 'Equity' : 'DD%']}
              />
              <Line
                type="monotone" dataKey="equity" dot={false}
                stroke="#00d4ff" strokeWidth={1.5} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}
