import { useState } from 'react';
import { sendCommand } from '@/hooks/useEngineSocket';
import { SYMBOLS_13 } from '@/lib/constants';

type LocalAlert = { symbol: string; direction: 'ABOVE' | 'BELOW'; threshold: number };

export default function PriceAlertManager() {
  const [alerts, setAlerts] = useState<LocalAlert[]>([]);
  const [draft, setDraft] = useState<LocalAlert>({
    symbol: 'EURUSD', direction: 'ABOVE', threshold: 1.08,
  });

  const add = async () => {
    const ok = await sendCommand('cmd_set_alert', { ...draft });
    if (ok) setAlerts((a) => [...a, draft]);
  };

  return (
    <section data-testid="price-alert-manager" className="rounded-lg border border-white/5 bg-bg-secondary p-4">
      <h3 className="text-sm font-bold uppercase tracking-wider text-white/70">Price Alerts</h3>
      <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
        <select
          value={draft.symbol}
          onChange={(e) => setDraft((d) => ({ ...d, symbol: e.target.value }))}
          className="rounded bg-bg-tertiary px-2 py-1 font-mono"
          data-testid="alert-symbol"
        >
          {SYMBOLS_13.map(({ name }) => <option key={name} value={name}>{name}</option>)}
        </select>
        <select
          value={draft.direction}
          onChange={(e) => setDraft((d) => ({ ...d, direction: e.target.value as any }))}
          className="rounded bg-bg-tertiary px-2 py-1"
          data-testid="alert-direction"
        >
          <option>ABOVE</option><option>BELOW</option>
        </select>
        <input
          type="number" step="0.00001" value={draft.threshold}
          onChange={(e) => setDraft((d) => ({ ...d, threshold: Number(e.target.value) }))}
          className="rounded bg-bg-tertiary px-2 py-1 font-mono"
          data-testid="alert-threshold"
        />
        <button onClick={() => void add()}
                data-testid="alert-add"
                className="rounded bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30">
          Add
        </button>
      </div>
      <ul className="mt-2 space-y-1 text-xs font-mono">
        {alerts.map((a, i) => (
          <li key={i} className="rounded bg-bg-tertiary px-2 py-1">
            {a.symbol} {a.direction} {a.threshold}
          </li>
        ))}
      </ul>
    </section>
  );
}
