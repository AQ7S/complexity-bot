import { useEngineStore } from '@/store/engineStore';
import { fmtPrice } from '@/lib/format';
import RegimeBadge from './RegimeBadge';
import { ALWAYS_ON } from '@/lib/constants';

export default function SymbolCard({ symbol, kind }: { symbol: string; kind: string }) {
  const tick = useEngineStore((s) => s.ticks[symbol]);
  const regime = useEngineStore((s) => s.regimes[symbol]?.regime);
  const digits = symbol.startsWith('XAU') || symbol.includes('BTC') || symbol.includes('ETH') ? 2 : 5;
  const stale = !tick;
  return (
    <div
      data-testid={`symbol-card-${symbol}`}
      className={`rounded-lg border border-white/5 bg-bg-secondary p-3 transition-colors ${
        stale ? 'opacity-60' : ''
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-bold text-white">{symbol}</span>
          <span className="rounded bg-white/5 px-1 text-[9px] uppercase text-white/40">
            {kind}
          </span>
          {ALWAYS_ON.has(symbol) && (
            <span className="text-[9px] text-accent-cyan">24/7</span>
          )}
        </div>
        <RegimeBadge regime={regime} />
      </div>
      <div className="mt-2 flex items-baseline justify-between font-mono">
        <div>
          <p className="text-[10px] uppercase text-white/40">Bid</p>
          <p className="text-lg text-accent-red">{tick ? fmtPrice(tick.bid, digits) : '—'}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] uppercase text-white/40">Ask</p>
          <p className="text-lg text-accent-green">{tick ? fmtPrice(tick.ask, digits) : '—'}</p>
        </div>
      </div>
      <div className="mt-1 flex items-center justify-between text-[10px] text-white/40">
        <span>Spread</span>
        <span className="font-mono">{tick ? fmtPrice(tick.spread, digits) : '—'}</span>
      </div>
    </div>
  );
}
