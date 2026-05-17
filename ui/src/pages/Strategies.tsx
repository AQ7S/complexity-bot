import { useEngineStore } from '@/store/engineStore';
import { sendCommand } from '@/hooks/useEngineSocket';
import type { StrategyHealthFrame, StrategyMode } from '@/types/ipc-messages';

function stateBadgeColor(state: string): string {
  switch (state) {
    case 'ACTIVE':   return 'bg-accent-green/20 text-accent-green border-accent-green/40';
    case 'SHADOW':   return 'bg-accent-purple/20 text-accent-purple border-accent-purple/40';
    case 'PAUSED':   return 'bg-accent-red/20 text-accent-red border-accent-red/40';
    case 'DISABLED': return 'bg-white/10 text-white/40 border-white/20';
    default:         return 'bg-white/10 text-white/40 border-white/20';
  }
}

function ModeButton({ active, onClick, children, color }: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  color: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded px-2 py-1 text-[10px] font-bold uppercase transition-opacity ${
        active ? color + ' opacity-100' : 'bg-bg-tertiary text-white/40 opacity-70 hover:opacity-100'
      }`}
    >
      {children}
    </button>
  );
}

function StrategyCard({ s }: { s: StrategyHealthFrame }) {
  const sharpeColor =
    s.rolling_sharpe >= 1.0 ? 'text-accent-green' :
    s.rolling_sharpe >= 0.5 ? 'text-accent-gold' :
    s.rolling_sharpe >= 0   ? 'text-white/70' :
                              'text-accent-red';
  const pnlColor = s.pnl_today_usd >= 0 ? 'text-accent-green' : 'text-accent-red';

  function setMode(mode: StrategyMode) {
    void sendCommand('cmd_strategy_toggle', { name: s.name, mode });
  }

  // Determine current operator-level mode from the breaker-aware state field.
  const currentMode: StrategyMode =
    s.state === 'DISABLED' ? 'OFF' :
    s.state === 'SHADOW'   ? 'SHADOW' : 'ON';

  return (
    <div
      data-testid={`strategy-card-${s.name}`}
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="font-hero text-sm uppercase tracking-wider text-white">{s.name}</div>
          <div className="text-[10px] uppercase tracking-wider text-white/40">{s.style}</div>
        </div>
        <span
          data-testid={`strategy-state-${s.name}`}
          className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase ${stateBadgeColor(s.state)}`}
        >
          {s.state}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded bg-bg-tertiary px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-white/40">Weight</div>
          <div className="mt-0.5 font-mono text-sm text-white">
            {(s.weight * 100).toFixed(1)}%
          </div>
        </div>
        <div className="rounded bg-bg-tertiary px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-white/40">Sharpe</div>
          <div className={`mt-0.5 font-mono text-sm font-bold ${sharpeColor}`}>
            {s.rolling_sharpe.toFixed(2)}
          </div>
        </div>
        <div className="rounded bg-bg-tertiary px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-white/40">Today's Trades</div>
          <div className="mt-0.5 font-mono text-sm text-white">{s.trades_today}</div>
        </div>
        <div className="rounded bg-bg-tertiary px-2 py-1.5">
          <div className="text-[9px] uppercase tracking-wider text-white/40">Today's P&L</div>
          <div className={`mt-0.5 font-mono text-sm font-bold ${pnlColor}`}>
            {s.pnl_today_usd >= 0 ? '+' : ''}${s.pnl_today_usd.toFixed(0)}
          </div>
        </div>
      </div>

      {s.consecutive_losses > 0 && (
        <div className="mt-2 text-[10px] text-accent-red/80">
          ⚠ {s.consecutive_losses} consecutive losses (circuit breaker at 5)
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-white/40 mr-1">Mode</span>
        <ModeButton
          active={currentMode === 'ON'}
          onClick={() => setMode('ON')}
          color="bg-accent-green/30 text-accent-green"
        >
          ON
        </ModeButton>
        <ModeButton
          active={currentMode === 'SHADOW'}
          onClick={() => setMode('SHADOW')}
          color="bg-accent-purple/30 text-accent-purple"
        >
          Shadow
        </ModeButton>
        <ModeButton
          active={currentMode === 'OFF'}
          onClick={() => setMode('OFF')}
          color="bg-accent-red/30 text-accent-red"
        >
          Off
        </ModeButton>
      </div>
    </div>
  );
}

export default function Strategies() {
  const status = useEngineStore((s) => s.strategyStatus);

  return (
    <section data-testid="page-strategies" className="flex h-full flex-col p-6">
      <div className="mb-4">
        <h1 className="font-hero text-2xl text-accent-cyan">Strategy Allocation</h1>
        <p className="mt-1 text-xs text-white/40">
          Per-strategy risk budgets, circuit-breaker state, and operator overrides.
          Allocation = rolling Sharpe-weighted, water-filled to a 5–50% per-strategy band.
        </p>
      </div>

      {status ? (
        <>
          <div className="mb-4 rounded-lg border border-accent-cyan/30 bg-bg-secondary p-3 text-xs text-white/70">
            <span className="mr-3 font-bold uppercase tracking-wider text-accent-cyan">
              Total daily risk
            </span>
            <span className="font-mono text-base text-white">
              {(status.total_risk_pct * 100).toFixed(2)}%
            </span>
            <span className="ml-3 text-white/40">
              · {status.strategies.length} strategies registered
            </span>
          </div>
          <div className="grid flex-1 auto-rows-max grid-cols-1 gap-3 overflow-auto md:grid-cols-2 xl:grid-cols-3">
            {status.strategies.map((s) => (
              <StrategyCard key={s.name} s={s} />
            ))}
          </div>
        </>
      ) : (
        <div className="flex flex-1 items-center justify-center rounded-lg border border-white/5 bg-bg-secondary">
          <div className="text-center">
            <div className="text-4xl text-white/10">⏚</div>
            <p className="mt-2 text-xs text-white/30">
              Waiting for strategy_status frame from the engine.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
