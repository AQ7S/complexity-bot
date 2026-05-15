import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13, TIMEFRAMES } from '@/lib/constants';

const ARROW = { BUY: '↑', SELL: '↓', HOLD: '·' } as const;
const COLOR = { BUY: 'text-accent-green', SELL: 'text-accent-red', HOLD: 'text-white/30' } as const;

/**
 * 13×4 matrix — most recent signal per (symbol, timeframe). The engine emits
 * one `signal_detected` per new bar; we pick the freshest seen so far.
 */
export default function SignalScanner() {
  const signals = useEngineStore((s) => s.signals);

  const cell = (symbol: string, tf: string) => {
    const hit = signals.find((s) => s.symbol === symbol && s.timeframe === tf);
    if (!hit) return <span className="text-white/20">·</span>;
    return (
      <span className={`font-bold ${COLOR[hit.direction]}`} title={`${hit.confluence}/5`}>
        {ARROW[hit.direction]}
        <span className="ml-1 text-[9px] text-white/50">{hit.confluence}</span>
      </span>
    );
  };

  return (
    <section data-testid="page-signal-scanner" className="p-6">
      <h1 className="font-hero text-2xl text-accent-cyan">Signal Scanner</h1>
      <p className="mt-1 text-xs text-white/40">
        Latest signal per (symbol × timeframe). Number after arrow = confluence /5.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="text-xs font-mono">
          <thead>
            <tr className="text-white/40">
              <th className="px-3 py-2 text-left">Symbol</th>
              {TIMEFRAMES.slice(0, 4).map((tf) => (
                <th key={tf} className="px-4 py-2">{tf}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SYMBOLS_13.map(({ name }) => (
              <tr key={name} data-testid={`scanner-row-${name}`} className="border-t border-white/5">
                <td className="px-3 py-1 text-white/80">{name}</td>
                {TIMEFRAMES.slice(0, 4).map((tf) => (
                  <td key={tf} className="px-4 py-1 text-center">{cell(name, tf)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
