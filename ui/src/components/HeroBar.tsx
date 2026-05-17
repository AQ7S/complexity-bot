import { motion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { AnimatedNumber } from './AnimatedNumber';
import { fmtUsd, fmtSignedUsd } from '@/lib/format';
import { sendCommand } from '@/hooks/useEngineSocket';

function ageHuman(ms: number): string {
  if (!isFinite(ms)) return 'never';
  if (ms < 2000) return 'just now';
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3600_000) return `${Math.round(ms / 60_000)}m ago`;
  return `${Math.round(ms / 3600_000)}h ago`;
}

export default function HeroBar() {
  const status = useEngineStore((s) => s.engineStatus);
  const account = useEngineStore((s) => s.account);
  const wsConnected = useEngineStore((s) => s.wsConnected);
  const weeklyPnl = useEngineStore((s) => s.weeklyPnl);
  const todayPnl = useEngineStore((s) => s.todayPnl);
  const sessionPnl = useEngineStore((s) => s.sessionPnl);
  const lastTickTs = useEngineStore((s) => s.lastTickTs);
  const [hover, setHover] = useState(false);
  const [, force] = useState(0);

  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const live = status?.status === 'LIVE';
  const starting = !status || status.status === 'STARTING';
  const error = status?.status === 'ERROR';
  const dot =
    error ? 'bg-accent-red' :
    !wsConnected ? 'bg-accent-red' :
    starting ? 'bg-accent-gold' :
    live ? 'bg-accent-green' :
    'bg-white/40';

  const lastTickAge = useMemo(() => {
    const tss = Object.values(lastTickTs);
    if (tss.length === 0) return Infinity;
    return Date.now() - Math.max(...tss);
  }, [lastTickTs]);

  const mt5State = status?.mt5_connected ? 'connected' : 'disconnected';
  const ipcState = wsConnected ? 'connected' : 'disconnected';

  return (
    <header
      data-testid="hero-bar"
      className="flex items-center justify-between border-b border-white/5 bg-bg-secondary px-6 py-4"
    >
      <div className="flex items-center gap-4">
        <h1 className="font-hero text-2xl tracking-widest text-accent-cyan">
          COMPLEXITY ENGINE
        </h1>
        <div
          className="relative"
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
        >
          <motion.span
            aria-label="connection status"
            className={`block h-3 w-3 rounded-full ${dot} cursor-help`}
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 1.4, repeat: Infinity }}
          />
          {hover && (
            <div
              data-testid="status-popover"
              className="absolute left-0 top-6 z-50 min-w-[260px] rounded-lg border border-white/10 bg-bg-tertiary p-3 text-xs shadow-xl"
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="font-bold uppercase tracking-wider text-white/80">Engine State</span>
                <span className={`rounded px-1.5 py-0.5 text-[9px] ${
                  live ? 'bg-accent-green/20 text-accent-green' :
                  error ? 'bg-accent-red/20 text-accent-red' :
                  'bg-white/10 text-white/60'
                }`}>{status?.status ?? '—'}</span>
              </div>
              <Row label="MT5 link" value={mt5State} good={status?.mt5_connected} />
              <Row label="IPC link" value={ipcState} good={wsConnected} />
              <Row label="Last tick" value={ageHuman(lastTickAge)} good={lastTickAge < 30_000} />
              <Row label="Uptime" value={status?.uptime_s != null ? `${Math.floor(status.uptime_s / 60)}m ${status.uptime_s % 60}s` : '—'} good={status?.uptime_s != null} />
              {status?.version && <Row label="Version" value={status.version} good />}
              <div className="mt-2 flex gap-2 border-t border-white/5 pt-2">
                <button
                  type="button"
                  onClick={() => void sendCommand('cmd_pause', { paused: status?.status !== 'PAUSED' })}
                  className="flex-1 rounded bg-white/5 px-2 py-1 text-[10px] uppercase text-white/70 hover:bg-white/10"
                >
                  {status?.status === 'PAUSED' ? 'Resume' : 'Pause'}
                </button>
                <button
                  type="button"
                  onClick={() => void sendCommand('cmd_emergency_close', {})}
                  className="flex-1 rounded bg-accent-red/30 px-2 py-1 text-[10px] uppercase text-accent-red hover:bg-accent-red/50"
                >
                  Emergency
                </button>
              </div>
            </div>
          )}
        </div>
        <span className="text-xs uppercase tracking-wider text-white/60">
          {status?.status ?? 'STARTING'}
        </span>
      </div>

      <div className="flex items-center gap-6 font-mono">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Equity</p>
          <p className="text-2xl text-accent-green">
            <AnimatedNumber value={account?.equity ?? 0} format={fmtUsd} />
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Today P&amp;L</p>
          <p className={`text-lg ${todayPnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            <AnimatedNumber value={todayPnl} format={fmtSignedUsd} />
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Session</p>
          <p className={`text-lg ${sessionPnl >= 0 ? 'text-accent-cyan' : 'text-accent-red'}`}>
            <AnimatedNumber value={sessionPnl} format={fmtSignedUsd} />
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Weekly</p>
          <p className={`text-lg ${weeklyPnl >= 0 ? 'text-accent-gold' : 'text-accent-red'}`}>
            <AnimatedNumber value={weeklyPnl} format={fmtSignedUsd} />
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Open</p>
          <p className="text-2xl text-white">{account?.open_positions ?? 0}</p>
        </div>
      </div>
    </header>
  );
}

function Row({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-white/50">{label}</span>
      <span className={`font-mono ${good ? 'text-accent-green' : 'text-accent-red'}`}>{value}</span>
    </div>
  );
}
