import type { RegimeChange } from '@/types/ipc-messages';

const COLORS: Record<RegimeChange['regime'], string> = {
  TRENDING_UP:     'bg-accent-green/20 text-accent-green',
  TRENDING_DOWN:   'bg-accent-red/20 text-accent-red',
  RANGING:         'bg-white/10 text-white/70',
  HIGH_VOLATILITY: 'bg-accent-purple/20 text-accent-purple',
};

const LABEL: Record<RegimeChange['regime'], string> = {
  TRENDING_UP: '↑ TREND',
  TRENDING_DOWN: '↓ TREND',
  RANGING: '— RANGE',
  HIGH_VOLATILITY: '⚡ VOL',
};

export default function RegimeBadge({ regime }: { regime?: RegimeChange['regime'] }) {
  if (!regime) {
    return <span className="rounded px-2 py-0.5 text-[10px] uppercase text-white/30">—</span>;
  }
  return (
    <span
      data-testid="regime-badge"
      className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${COLORS[regime]}`}
    >
      {LABEL[regime]}
    </span>
  );
}
