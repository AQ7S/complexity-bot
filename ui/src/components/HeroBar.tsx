import { motion } from 'framer-motion';
import { useEngineStore } from '@/store/engineStore';
import { AnimatedNumber } from './AnimatedNumber';
import { fmtUsd, fmtSignedUsd } from '@/lib/format';

export default function HeroBar() {
  const status = useEngineStore((s) => s.engineStatus);
  const account = useEngineStore((s) => s.account);
  const wsConnected = useEngineStore((s) => s.wsConnected);
  const weeklyPnl = useEngineStore((s) => s.weeklyPnl);

  const live = status?.status === 'LIVE';
  const dot = wsConnected && live ? 'bg-accent-green' : 'bg-accent-red';

  return (
    <header
      data-testid="hero-bar"
      className="flex items-center justify-between border-b border-white/5 bg-bg-secondary px-6 py-4"
    >
      <div className="flex items-center gap-4">
        <h1 className="font-hero text-2xl tracking-widest text-accent-cyan">
          COMPLEXITY ENGINE
        </h1>
        <motion.span
          aria-label="connection status"
          className={`block h-3 w-3 rounded-full ${dot}`}
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 1.4, repeat: Infinity }}
        />
        <span className="text-xs uppercase tracking-wider text-white/60">
          {status?.status ?? 'STARTING'}
        </span>
      </div>

      <div className="flex items-center gap-8 font-mono">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Equity</p>
          <p className="text-2xl text-accent-green">
            <AnimatedNumber value={account?.equity ?? 0} format={fmtUsd} />
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-white/50">Weekly P&amp;L</p>
          <p className={`text-2xl ${weeklyPnl >= 0 ? 'text-accent-gold' : 'text-accent-red'}`}>
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
