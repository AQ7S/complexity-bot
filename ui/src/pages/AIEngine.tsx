import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { sendCommand } from '@/hooks/useEngineSocket';
import { useEngineStore } from '@/store/engineStore';
import type { CalibrationUpdate, ModelPromotionReady, ShadowStatus, WeeklyDebrief } from '@/types/ipc-messages';

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
  const weeklyDebrief = useEngineStore((s) => s.weeklyDebrief);
  const shadowStatus = useEngineStore((s) => s.shadowStatus);
  const promotion = useEngineStore((s) => s.promotionReady);
  const calibration = useEngineStore((s) => s.calibration);

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

      <ShadowAndCalibrationRow shadow={shadowStatus} calibration={calibration} promotion={promotion} />

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

      <WeeklyDebriefPanel debrief={weeklyDebrief} />

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

function WeeklyDebriefPanel({ debrief }: { debrief: WeeklyDebrief | null }) {
  if (!debrief) {
    return (
      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">
          Weekly Debrief (Claude)
        </h2>
        <p className="text-xs text-white/40" data-testid="weekly-debrief">
          No debrief yet. Next run: Monday 06:00 UTC.
        </p>
      </section>
    );
  }

  const { week_start, markdown, trades_count, net_pnl, win_rate } = debrief;
  return (
    <section className="rounded-lg border border-accent-purple/20 bg-bg-secondary p-4" data-testid="weekly-debrief">
      <header className="mb-3 flex flex-wrap items-baseline gap-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-accent-purple">
          Weekly Debrief — {week_start}
        </h2>
        {trades_count != null && (
          <span className="rounded bg-bg-tertiary px-2 py-0.5 font-mono text-xs text-white/60">
            {trades_count} trades
          </span>
        )}
        {net_pnl != null && (
          <span className={`rounded px-2 py-0.5 font-mono text-xs ${net_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {net_pnl >= 0 ? '+' : ''}{net_pnl.toFixed(2)} USD
          </span>
        )}
        {win_rate != null && (
          <span className="rounded bg-bg-tertiary px-2 py-0.5 font-mono text-xs text-white/60">
            {(win_rate * 100).toFixed(1)}% win rate
          </span>
        )}
      </header>
      <div className="prose prose-invert max-w-none text-sm leading-relaxed text-white/80
                      [&_h2]:mt-4 [&_h2]:text-sm [&_h2]:font-bold [&_h2]:text-accent-cyan
                      [&_h3]:mt-3 [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:text-white/70
                      [&_ul]:pl-4 [&_li]:text-white/70 [&_code]:rounded [&_code]:bg-bg-tertiary
                      [&_code]:px-1 [&_code]:text-xs [&_code]:text-accent-gold
                      [&_strong]:text-white whitespace-pre-wrap">
        {markdown}
      </div>
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
  const isTraining = busy || version.startsWith('retrain_starting');
  return (
    <div className={`rounded-lg border bg-bg-secondary p-4 transition-colors ${
      isTraining ? 'border-accent-purple/60' : 'border-white/5'
    }`} data-testid={`model-card-${name}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase text-white/50">{name}</p>
          <p className="font-mono text-sm text-white" data-testid={`model-version-${name}`}>{version}</p>
        </div>
        <div className="flex items-center gap-2">
          {isTraining && (
            <span className="flex items-center gap-1 rounded-full bg-accent-purple/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-accent-purple">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-purple" />
              Training
            </span>
          )}
          <button
            type="button" onClick={onRetrain} disabled={isTraining}
            data-testid={`retrain-${name}`}
            className="rounded bg-accent-purple/20 px-3 py-1 text-xs text-accent-purple hover:bg-accent-purple/30 disabled:opacity-40"
          >
            {isTraining ? 'Working…' : 'Retrain'}
          </button>
        </div>
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

function ShadowAndCalibrationRow({
  shadow, calibration, promotion,
}: {
  shadow: ShadowStatus | null;
  calibration: CalibrationUpdate | null;
  promotion: ModelPromotionReady | null;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <ShadowPanel shadow={shadow} promotion={promotion} />
      <CalibrationPanel calibration={calibration} />
    </div>
  );
}

function ShadowPanel({
  shadow, promotion,
}: {
  shadow: ShadowStatus | null;
  promotion: ModelPromotionReady | null;
}) {
  const active = shadow?.active ?? true;
  const closed = shadow?.closed_count ?? 0;
  const wr = shadow?.win_rate ?? 0;
  const sharpe = shadow?.sharpe ?? 0;
  const avgR = shadow?.avg_r ?? 0;
  const wrPct = (wr * 100).toFixed(1);

  return (
    <div className="rounded-lg border border-white/5 bg-bg-secondary p-4" data-testid="shadow-panel">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">Shadow Mode</h2>
          <span className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            active ? 'bg-accent-purple/20 text-accent-purple' : 'bg-white/10 text-white/60'
          }`}>
            {active ? 'ACTIVE' : 'OFF'}
          </span>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-white/40">No real orders</span>
      </div>
      <div className="grid grid-cols-3 gap-3 font-mono">
        <Metric label="Closed" value={String(closed)} />
        <Metric label="Win Rate" value={`${wrPct}%`}
          tone={wr > 0.55 ? 'green' : wr < 0.45 ? 'red' : 'amber'} />
        <Metric label="Sharpe" value={sharpe.toFixed(2)}
          tone={sharpe > 1.0 ? 'green' : sharpe < 0.5 ? 'red' : 'amber'} />
        <Metric label="Open" value={String(shadow?.open_count ?? 0)} />
        <Metric label="Avg R" value={avgR.toFixed(2)}
          tone={avgR > 0 ? 'green' : 'red'} />
        <Metric label="Cum R" value={(shadow?.cumulative_pnl_r ?? 0).toFixed(1)}
          tone={(shadow?.cumulative_pnl_r ?? 0) > 0 ? 'green' : 'red'} />
      </div>
      {promotion && (
        <div className="mt-3 rounded border border-accent-green/40 bg-accent-green/10 p-2 text-xs"
             data-testid="promotion-banner">
          <div className="font-bold uppercase tracking-wider text-accent-green">
            Candidate ready for promotion
          </div>
          <div className="mt-1 text-white/80">
            Shadow Sharpe {promotion.shadow_sharpe.toFixed(2)} on {promotion.shadow_trades} trades,
            WR {(promotion.shadow_win_rate * 100).toFixed(1)}%, avg R {promotion.avg_r.toFixed(2)}
          </div>
        </div>
      )}
      {closed < 100 && (
        <p className="mt-3 text-[11px] text-white/40">
          Promotion gate: needs ≥100 closed shadow trades, current {closed}.
        </p>
      )}
    </div>
  );
}

function CalibrationPanel({ calibration }: { calibration: CalibrationUpdate | null }) {
  const ece = calibration?.ece_score ?? 0;
  const n = calibration?.n_trades ?? 0;
  const overconfident = calibration?.overconfident ?? false;
  const tone =
    !calibration ? 'muted' :
    ece > 0.20 ? 'red' :
    ece > 0.10 ? 'amber' :
    'green';
  const arcFill = Math.min(1, ece / 0.30);
  const colorClass =
    tone === 'green' ? 'text-accent-green' :
    tone === 'amber' ? 'text-accent-gold' :
    tone === 'red'   ? 'text-accent-red' : 'text-white/40';

  return (
    <div className="rounded-lg border border-white/5 bg-bg-secondary p-4" data-testid="ece-panel">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Calibration (ECE)
        </h2>
        {calibration && (
          <span className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            overconfident ? 'bg-accent-red/20 text-accent-red' : 'bg-accent-green/20 text-accent-green'
          }`}>
            {overconfident ? 'Overconfident' : 'Well calibrated'}
          </span>
        )}
      </div>
      {!calibration ? (
        <p className="text-xs text-white/40" data-testid="ece-gauge">
          Need ≥50 closed shadow trades to compute first ECE.
        </p>
      ) : (
        <>
          <div className="flex items-end gap-4" data-testid="ece-gauge">
            <p className={`font-mono text-4xl ${colorClass}`}>{ece.toFixed(3)}</p>
            <p className="pb-1 text-xs text-white/50">over {n} trades</p>
          </div>
          <div className="mt-3 h-2 w-full overflow-hidden rounded bg-bg-tertiary">
            <div
              className={`h-full transition-all ${
                tone === 'green' ? 'bg-accent-green' :
                tone === 'amber' ? 'bg-accent-gold' : 'bg-accent-red'
              }`}
              style={{ width: `${arcFill * 100}%` }}
            />
          </div>
          <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wider text-white/40">
            <span>Calibrated &lt;0.10</span>
            <span>Suspect 0.10–0.20</span>
            <span>Skeptical &gt;0.20</span>
          </div>
        </>
      )}
    </div>
  );
}

function Metric({
  label, value, tone,
}: {
  label: string; value: string;
  tone?: 'green' | 'red' | 'amber' | 'muted';
}) {
  const cls =
    tone === 'green' ? 'text-accent-green' :
    tone === 'red'   ? 'text-accent-red' :
    tone === 'amber' ? 'text-accent-gold' : 'text-white';
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-white/40">{label}</p>
      <p className={`text-lg ${cls}`}>{value}</p>
    </div>
  );
}
