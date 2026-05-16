import { useEffect, useState } from 'react';
import { Bell, BellOff, Trash2 } from 'lucide-react';
import { sendCommand } from '@/hooks/useEngineSocket';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';

type LocalAlert = {
  id: number;
  symbol: string;
  direction: 'ABOVE' | 'BELOW';
  threshold: number;
  enabled: boolean;
  triggeredAt: number | null;
};

let nextId = 1;

export default function PriceAlertManager() {
  const ticks = useEngineStore((s) => s.ticks);
  const [alerts, setAlerts] = useState<LocalAlert[]>([]);
  const [draft, setDraft] = useState({ symbol: 'EURUSD', direction: 'ABOVE' as 'ABOVE' | 'BELOW', threshold: '' });
  const [adding, setAdding] = useState(false);

  const notifications = useEngineStore((s) => s.notifications);

  // React to price_alert IPC events (triggeredAt update)
  useEffect(() => {
    const latest = notifications[0];
    if (!latest) return;
    setAlerts((prev) => prev.map((a) => {
      const match = `${a.symbol} ${a.direction} ${a.threshold}`;
      if (latest.title?.includes(match) && !a.triggeredAt) {
        return { ...a, triggeredAt: latest.ts };
      }
      return a;
    }));
  }, [notifications]);

  const currentPrice = (symbol: string) => {
    const t = ticks[symbol];
    return t ? (t.bid + t.ask) / 2 : null;
  };

  const add = async () => {
    const threshold = parseFloat(draft.threshold);
    if (isNaN(threshold) || threshold <= 0) return;
    setAdding(true);
    const ok = await sendCommand('cmd_set_alert', {
      symbol: draft.symbol,
      direction: draft.direction,
      threshold,
    });
    setAdding(false);
    if (ok !== false) {
      setAlerts((a) => [...a, {
        id: nextId++,
        symbol: draft.symbol,
        direction: draft.direction,
        threshold,
        enabled: true,
        triggeredAt: null,
      }]);
      setDraft((d) => ({ ...d, threshold: '' }));
    }
  };

  const remove = (id: number) => {
    setAlerts((a) => a.filter((x) => x.id !== id));
  };

  const toggle = (id: number) => {
    setAlerts((a) => a.map((x) => x.id === id ? { ...x, enabled: !x.enabled } : x));
  };

  const dirColor = (d: 'ABOVE' | 'BELOW') => d === 'ABOVE' ? 'text-accent-green' : 'text-accent-red';

  return (
    <section data-testid="price-alert-manager" className="rounded-lg border border-white/5 bg-bg-secondary p-4">
      <header className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white/70">Price Alerts</h3>
        <span className="rounded bg-bg-tertiary px-2 py-0.5 font-mono text-xs text-white/50">
          {alerts.filter((a) => a.enabled && !a.triggeredAt).length} active
        </span>
      </header>

      {/* Add form */}
      <div className="mb-3 grid grid-cols-[1fr_auto_1fr_auto] gap-2 text-xs">
        <select
          value={draft.symbol}
          onChange={(e) => setDraft((d) => ({ ...d, symbol: e.target.value }))}
          className="rounded bg-bg-tertiary px-2 py-1.5 font-mono text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="alert-symbol"
        >
          {SYMBOLS_13.map(({ name }) => <option key={name} value={name}>{name}</option>)}
        </select>

        <select
          value={draft.direction}
          onChange={(e) => setDraft((d) => ({ ...d, direction: e.target.value as 'ABOVE' | 'BELOW' }))}
          className="rounded bg-bg-tertiary px-2 py-1.5 text-white focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="alert-direction"
        >
          <option value="ABOVE">ABOVE ↑</option>
          <option value="BELOW">BELOW ↓</option>
        </select>

        <input
          type="number"
          step="0.00001"
          placeholder="Threshold price"
          value={draft.threshold}
          onChange={(e) => setDraft((d) => ({ ...d, threshold: e.target.value }))}
          className="rounded bg-bg-tertiary px-2 py-1.5 font-mono text-white placeholder:text-white/20 focus:outline-none focus:ring-1 focus:ring-accent-cyan"
          data-testid="alert-threshold"
          onKeyDown={(e) => { if (e.key === 'Enter') void add(); }}
        />

        <button
          type="button"
          onClick={() => void add()}
          disabled={adding || !draft.threshold}
          data-testid="alert-add"
          className="rounded bg-accent-cyan/20 px-3 py-1.5 text-accent-cyan transition-colors hover:bg-accent-cyan/30 disabled:opacity-40"
        >
          {adding ? '…' : 'Add'}
        </button>
      </div>

      {/* Alert list */}
      {alerts.length === 0 ? (
        <p className="py-3 text-center text-[11px] text-white/30">
          No alerts set. Add one above.
        </p>
      ) : (
        <ul className="space-y-1.5">
          {alerts.map((alert) => {
            const price = currentPrice(alert.symbol);
            const triggered = !!alert.triggeredAt;
            const distance = price != null ? Math.abs(price - alert.threshold) : null;

            return (
              <li
                key={alert.id}
                data-testid={`alert-item-${alert.id}`}
                className={`flex items-center gap-2 rounded px-3 py-2 text-xs transition-colors ${
                  triggered ? 'bg-accent-gold/10 border border-accent-gold/20'
                  : !alert.enabled ? 'bg-bg-tertiary opacity-50'
                  : 'bg-bg-tertiary'
                }`}
              >
                {/* Status icon */}
                <span className={triggered ? 'text-accent-gold' : alert.enabled ? 'text-white/40' : 'text-white/20'}>
                  {triggered ? <Bell size={12} /> : alert.enabled ? <Bell size={12} /> : <BellOff size={12} />}
                </span>

                {/* Content */}
                <span className="flex-1 font-mono">
                  <span className="text-white/80">{alert.symbol}</span>
                  {' '}
                  <span className={dirColor(alert.direction)}>{alert.direction}</span>
                  {' '}
                  <span className="text-white">{alert.threshold.toFixed(alert.threshold > 100 ? 2 : 5)}</span>
                </span>

                {/* Status / distance */}
                <span className="text-[10px]">
                  {triggered ? (
                    <span className="text-accent-gold font-bold">TRIGGERED</span>
                  ) : price != null && distance != null ? (
                    <span className="text-white/40">
                      {alert.direction === 'ABOVE'
                        ? price < alert.threshold ? `+${distance.toFixed(alert.threshold > 100 ? 2 : 5)} away` : 'crossed'
                        : price > alert.threshold ? `-${distance.toFixed(alert.threshold > 100 ? 2 : 5)} away` : 'crossed'
                      }
                    </span>
                  ) : (
                    <span className="text-white/20">—</span>
                  )}
                </span>

                {/* Actions */}
                <button
                  type="button"
                  onClick={() => toggle(alert.id)}
                  className="text-white/30 transition-colors hover:text-white"
                  title={alert.enabled ? 'Disable' : 'Enable'}
                >
                  {alert.enabled ? <Bell size={11} /> : <BellOff size={11} />}
                </button>
                <button
                  type="button"
                  onClick={() => remove(alert.id)}
                  className="text-white/30 transition-colors hover:text-accent-red"
                  title="Delete alert"
                  data-testid={`alert-delete-${alert.id}`}
                >
                  <Trash2 size={11} />
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {/* Price reference */}
      {alerts.length > 0 && (
        <div className="mt-2 text-[9px] text-white/30">
          Live prices: {[...new Set(alerts.map((a) => a.symbol))].map((s) => {
            const p = currentPrice(s);
            return p != null ? `${s} ${p.toFixed(p > 100 ? 2 : 5)}` : null;
          }).filter(Boolean).join(' · ')}
        </div>
      )}
    </section>
  );
}
