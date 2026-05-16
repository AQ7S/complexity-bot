import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { useEngineStore } from '@/store/engineStore';

const COLORS = {
  BUY:  'text-accent-green',
  SELL: 'text-accent-red',
  SKIP: 'text-white/60',
} as const;

function timeAgo(ms: number | null): string {
  if (!ms) return '—';
  const d = Date.now() - ms;
  if (d < 60_000) return `${Math.round(d / 1000)}s ago`;
  if (d < 3600_000) return `${Math.round(d / 60_000)}m ago`;
  return `${Math.round(d / 3600_000)}h ago`;
}

export default function ClaudeFeed() {
  const feed = useEngineStore((s) => s.claudeFeed);
  const stats = useEngineStore((s) => s.claudeStats);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const skipRate = stats.total > 0 ? (stats.skips / stats.total) * 100 : 0;

  const toggle = (i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  return (
    <section
      data-testid="claude-feed"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Claude Decisions
        </h2>
        <span className="rounded bg-bg-tertiary px-2 py-0.5 font-mono text-xs text-white/70">
          {stats.total}
        </span>
      </header>
      <div className="mb-3 grid grid-cols-4 gap-2 text-center">
        <Stat label="Buys"   value={stats.buys}  tone="text-accent-green" />
        <Stat label="Sells"  value={stats.sells} tone="text-accent-red" />
        <Stat label="Skips"  value={stats.skips} tone="text-white/60" />
        <Stat label="Skip %" value={`${skipRate.toFixed(0)}%`} tone="text-accent-gold" />
      </div>
      <p className="mb-2 text-right text-[10px] text-white/40">last {timeAgo(stats.lastTs)}</p>
      {feed.length === 0 ? (
        <p className="py-4 text-center text-xs text-white/40">
          {stats.total === 0 ? 'No Claude decisions yet.' : 'Recent decisions purged.'}
        </p>
      ) : (
        <ul className="space-y-2">
          <AnimatePresence initial={false}>
            {feed.map((c, i) => (
              <motion.li
                key={`${c.symbol}-${i}-${c.reasoning_excerpt.slice(0, 16)}`}
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className="cursor-pointer rounded bg-bg-tertiary p-2 text-xs transition-colors hover:bg-white/5"
                onClick={() => toggle(i)}
                data-testid={`claude-feed-item-${i}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-white">{c.symbol}</span>
                  <span className={`font-bold ${COLORS[c.decision]}`}>
                    {c.decision} · {c.confidence}%
                    <span className="ml-2 text-white/30">{expanded.has(i) ? '▲' : '▼'}</span>
                  </span>
                </div>
                <p className={`mt-1 text-white/60 ${expanded.has(i) ? '' : 'line-clamp-2'}`}>
                  {c.reasoning_excerpt}
                </p>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </section>
  );
}

function Stat({ label, value, tone }: { label: string; value: number | string; tone: string }) {
  return (
    <div className="rounded bg-bg-tertiary px-1 py-1">
      <div className="text-[9px] uppercase text-white/40">{label}</div>
      <div className={`font-mono text-sm ${tone}`}>{value}</div>
    </div>
  );
}
