import { useEffect, useMemo, useState } from 'react';
import { KILL_ZONES } from '@/lib/constants';

const DAY_MIN = 24 * 60;

function nowMinutesEST(): number {
  const utc = new Date();
  const minutes = utc.getUTCHours() * 60 + utc.getUTCMinutes() + (utc.getUTCSeconds() / 60);
  return (minutes - 5 * 60 + DAY_MIN) % DAY_MIN;
}

function fmtHHMM(min: number): string {
  const h = Math.floor(min / 60);
  const m = Math.floor(min % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

function fmtCountdown(min: number): string {
  const total = Math.max(0, Math.round(min * 60));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

export default function KillZoneTimeline({ nowProvider }: { nowProvider?: () => number }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const now = (nowProvider ?? nowMinutesEST)();

  const sortedZones = useMemo(() => [...KILL_ZONES].sort((a, b) => a.start - b.start), []);
  const activeZone = sortedZones.find((z) => now >= z.start && now < z.end);
  const nextZone = activeZone
    ? sortedZones[(sortedZones.indexOf(activeZone) + 1) % sortedZones.length]
    : sortedZones.find((z) => z.start > now) ?? sortedZones[0];

  let minsTo = nextZone.start - now;
  if (minsTo < 0) minsTo += DAY_MIN;

  return (
    <section
      data-testid="kill-zone-timeline"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Kill Zones (EST)
        </h2>
        <span className="font-mono text-xs text-white/40">
          {fmtHHMM(now)}
        </span>
      </header>
      <div className="relative h-9 w-full overflow-hidden rounded bg-bg-tertiary">
        {sortedZones.map((z) => {
          const left = (z.start / DAY_MIN) * 100;
          const width = ((z.end - z.start) / DAY_MIN) * 100;
          const active = now >= z.start && now < z.end;
          return (
            <div
              key={z.label}
              data-testid={`kz-${z.label.toLowerCase().replace(/ /g, '-')}`}
              data-active={active}
              className={`absolute top-0 flex h-full items-center justify-center text-[9px] font-bold uppercase tracking-wider transition-colors ${
                active ? 'bg-accent-cyan/60 text-bg-primary' : 'bg-accent-cyan/15 text-white/60'
              }`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`${z.label}: ${fmtHHMM(z.start)}–${fmtHHMM(z.end)}`}
            >
              <span className="truncate px-1">{z.label}</span>
            </div>
          );
        })}
        <div
          className="absolute top-0 h-full w-px bg-accent-gold shadow-[0_0_8px_rgba(255,184,0,0.8)]"
          style={{ left: `${(now / DAY_MIN) * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-white/40">
        <span>00</span><span>06</span><span>12</span><span>18</span><span>24</span>
      </div>
      <div className="mt-2 flex items-center justify-between rounded bg-bg-tertiary/60 px-2 py-1.5 text-[11px]">
        <span className="uppercase tracking-wider text-white/50">
          {activeZone ? (
            <>Active: <span className="text-accent-cyan">{activeZone.label}</span></>
          ) : (
            <>Dead Zone — next: <span className="text-accent-cyan">{nextZone.label}</span></>
          )}
        </span>
        <span className="font-mono text-accent-gold">
          {activeZone ? `${fmtCountdown(activeZone.end - now)} remaining` : `in ${fmtCountdown(minsTo)}`}
        </span>
      </div>
    </section>
  );
}
