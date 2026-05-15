import { useState } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { ChevronDown, ChevronUp } from 'lucide-react';

function cellColor(v: number): string {
  // |v| → 0..1; positive = green, negative = red, near-zero = bg-tertiary.
  const a = Math.min(1, Math.abs(v));
  const intensity = Math.round(a * 80);
  if (v >= 0.05) return `rgba(0, 255, 136, ${intensity / 100})`;
  if (v <= -0.05) return `rgba(255, 59, 107, ${intensity / 100})`;
  return 'rgba(255, 255, 255, 0.04)';
}

export default function CorrelationHeatmap() {
  const corr = useEngineStore((s) => s.correlation);
  const [open, setOpen] = useState(false);
  return (
    <section
      data-testid="correlation-heatmap"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Correlation Matrix (15-min)
        </h2>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 text-xs text-white/60 hover:text-white"
          data-testid="heatmap-toggle"
        >
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {open ? 'Hide' : 'Show'}
        </button>
      </header>
      {!corr ? (
        <p className="py-4 text-center text-xs text-white/40">Awaiting first snapshot.</p>
      ) : open ? (
        <div className="mt-2 overflow-x-auto">
          <table className="text-[10px] font-mono">
            <thead>
              <tr>
                <th className="p-1" />
                {corr.symbols.map((s) => (
                  <th key={s} className="rotate-45 p-1 text-left text-white/50">{s}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {corr.matrix.map((row, i) => (
                <tr key={corr.symbols[i]}>
                  <td className="p-1 text-right text-white/60">{corr.symbols[i]}</td>
                  {row.map((v, j) => (
                    <td
                      key={`${i}-${j}`}
                      className="p-1 text-center text-white/80"
                      style={{ background: cellColor(v) }}
                      title={`${v.toFixed(2)}`}
                    >
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
