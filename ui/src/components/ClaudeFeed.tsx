import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { useEngineStore } from '@/store/engineStore';

const COLORS = {
  BUY:  'text-accent-green',
  SELL: 'text-accent-red',
  SKIP: 'text-white/60',
} as const;

export default function ClaudeFeed() {
  const feed = useEngineStore((s) => s.claudeFeed);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

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
        <span className="text-xs text-white/40">{feed.length}</span>
      </header>
      {feed.length === 0 ? (
        <p className="py-8 text-center text-xs text-white/40">No Claude decisions yet.</p>
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
