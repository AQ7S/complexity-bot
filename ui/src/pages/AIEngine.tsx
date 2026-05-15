import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { sendCommand } from '@/hooks/useEngineSocket';
import { useEngineStore } from '@/store/engineStore';

const FAKE_FEATURE_IMPORTANCE = [
  { name: 'EMA21',     value: 0.18 },
  { name: 'RSI14',     value: 0.15 },
  { name: 'ATR14',     value: 0.12 },
  { name: 'OB-dist',   value: 0.11 },
  { name: 'BBwidth',   value: 0.09 },
  { name: 'MACDhist',  value: 0.08 },
  { name: 'VWAPdev',   value: 0.07 },
  { name: 'ADX',       value: 0.06 },
  { name: 'ROC10',     value: 0.05 },
  { name: 'KillZone',  value: 0.04 },
];

export default function AIEngine() {
  const [confirm, setConfirm] = useState<null | 'cnn_lstm' | 'rl_dqn'>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [lastAck, setLastAck] = useState<string | null>(null);

  const triggerRetrain = async (model: 'cnn_lstm' | 'rl_dqn') => {
    setBusy(model);
    const ok = await sendCommand('cmd_manual_retrain', { model });
    setBusy(null);
    setConfirm(null);
    setLastAck(`${model}: ${ok ? 'queued' : 'no engine bridge'}`);
  };

  return (
    <section data-testid="page-ai-engine" className="space-y-4 p-6">
      <h1 className="font-hero text-2xl text-accent-cyan">AI Engine</h1>

      <ModelCardsRow busy={busy} onCnn={() => setConfirm('cnn_lstm')} onRl={() => setConfirm('rl_dqn')} />

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">
          Feature Importance (CNN-LSTM, top 10)
        </h2>
        <div className="h-56">
          <ResponsiveContainer>
            <BarChart data={FAKE_FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 30 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="name" width={70}
                     tick={{ fill: '#cbd5e1', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#161b2c', border: 'none' }}
                       formatter={(v: any) => Number(v).toFixed(3)} />
              <Bar dataKey="value" fill="#00d4ff" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">
          Weekly Debrief (Claude)
        </h2>
        <div className="prose prose-invert max-w-none text-sm text-white/70" data-testid="weekly-debrief">
          <p className="text-white/40">
            No debrief yet. Next run: Monday 06:00 UTC.
          </p>
        </div>
      </section>

      {lastAck && (
        <p className="text-xs text-white/50" data-testid="retrain-ack">{lastAck}</p>
      )}

      {confirm && (
        <div role="dialog" aria-modal="true"
             className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
             data-testid="retrain-confirm">
          <div className="w-80 rounded-lg border border-white/10 bg-bg-secondary p-4 text-sm">
            <h3 className="text-lg font-bold text-accent-cyan">Manual retrain?</h3>
            <p className="mt-2 text-white/70">
              This spawns a low-priority background worker for <code>{confirm}</code>.
              The trading loop is unaffected.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setConfirm(null)}
                      className="rounded px-3 py-1 text-xs text-white/60 hover:bg-bg-tertiary">
                Cancel
              </button>
              <button onClick={() => void triggerRetrain(confirm)}
                      data-testid="retrain-confirm-btn"
                      className="rounded bg-accent-cyan/30 px-3 py-1 text-xs text-accent-cyan">
                Retrain
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function ModelCardsRow({
  busy, onCnn, onRl,
}: {
  busy: string | null; onCnn: () => void; onRl: () => void;
}) {
  const m = useEngineStore((s) => s.modelUpdates);
  const cnn = m.cnn_lstm;
  const rl  = m.rl_dqn;
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <ModelCard
        name="CNN-LSTM"
        version={cnn?.version ?? 'v_colab_20260513_2058'}
        accuracy={cnn?.accuracy ?? 0.4309}
        loss={cnn?.loss ?? null}
        onRetrain={onCnn}
        busy={busy === 'cnn_lstm'}
      />
      <ModelCard
        name="RL DQN"
        version={rl?.version ?? 'rl_dqn_v1777749476'}
        accuracy={rl?.accuracy ?? null}
        loss={rl?.loss ?? null}
        onRetrain={onRl}
        busy={busy === 'rl_dqn'}
      />
    </div>
  );
}

function ModelCard({
  name, version, accuracy, loss, onRetrain, busy,
}: {
  name: string; version: string;
  accuracy: number | null; loss: number | null;
  onRetrain: () => void; busy: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-bg-secondary p-4" data-testid={`model-card-${name}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase text-white/50">{name}</p>
          <p className="font-mono text-sm text-white">{version}</p>
        </div>
        <button
          type="button" onClick={onRetrain} disabled={busy}
          data-testid={`retrain-${name}`}
          className="rounded bg-accent-purple/20 px-3 py-1 text-xs text-accent-purple hover:bg-accent-purple/30 disabled:opacity-40"
        >
          {busy ? 'Working…' : 'Retrain'}
        </button>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 font-mono">
        <div>
          <p className="text-[10px] uppercase text-white/40">Accuracy</p>
          <p className="text-lg text-accent-green">{accuracy != null ? accuracy.toFixed(4) : '—'}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase text-white/40">Loss</p>
          <p className="text-lg text-accent-red">{loss != null ? loss.toFixed(4) : '—'}</p>
        </div>
      </div>
    </div>
  );
}
