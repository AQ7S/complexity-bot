import { useEffect, useState } from 'react';
import { useEngineStore } from '@/store/engineStore';
import { ChevronDown, ChevronUp } from 'lucide-react';

const WARMUP_MS = 15 * 60_000;

function cellColor(v: number): string {
  const a = Math.min(1, Math.abs(v));
  const intensity = Math.round(a * 80);
  if (v >= 0.05) return `rgba(0, 255, 136, ${intensity / 100})`;
  if (v <= -0.05) return `rgba(255, 59, 107, ${intensity / 100})`;
  return 'rgba(255, 255, 255, 0.04)';
}

function fmtCountdown(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default function CorrelationHeatmap() {
  const corr = useEngineStore((s) => s.correlation);
  const sessionStartTs = useEngineStore((s) => s.sessionStartTs);
  const lastTickTs = useEngineStore((s) => s.lastTickTs);
  const [open, setOpen] = useState(false);
  const [, force] = useState(0);

  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const firstTick = Object.values(lastTickTs).length > 0 ? Math.min(...Object.values(lastTickTs)) : sessionStartTs;
  const elapsed = Date.now() - firstTick;
  const remaining = WARMUP_MS - elapsed;
  const progress = Math.min(1, elapsed / WARMUP_MS);

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
        <div className="py-4 text-center">
          <p className="text-xs text-white/40">
            Computing correlations from live ticks…
          </p>
          <p className="mt-1 font-mono text-xs text-accent-cyan">
            Available in {fmtCountdown(remaining)}
          </p>
          <div className="mx-auto mt-2 h-1 w-3/4 rounded-full bg-bg-tertiary overflow-hidden">
            <div
              className="h-full bg-accent-cyan transition-all"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
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
      ) : (
        <p className="mt-2 text-center text-[10px] text-white/40">
          {corr.symbols.length}×{corr.symbols.length} matrix ready — click Show to expand.
        </p>
      )}
    </section>
  );
}
