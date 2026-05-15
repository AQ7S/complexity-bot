import { useEffect, useState } from 'react';
import { KILL_ZONES } from '@/lib/constants';

const DAY_MIN = 24 * 60;

function nowMinutesEST(): number {
  // EST = UTC-5 (DST not modeled here; the engine handles DST authoritatively).
  const utc = new Date();
  const minutes = utc.getUTCHours() * 60 + utc.getUTCMinutes();
  return (minutes - 5 * 60 + DAY_MIN) % DAY_MIN;
}

export default function KillZoneTimeline({ nowProvider }: { nowProvider?: () => number }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 60_000);
    return () => clearInterval(id);
  }, []);
  void tick;
  const now = (nowProvider ?? nowMinutesEST)();
  return (
    <section
      data-testid="kill-zone-timeline"
      className="rounded-lg border border-white/5 bg-bg-secondary p-4"
    >
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-white/70">
          Kill Zones (EST)
        </h2>
        <span className="text-xs text-white/40">
          {String(Math.floor(now / 60)).padStart(2, '0')}:
          {String(now % 60).padStart(2, '0')}
        </span>
      </header>
      <div className="relative h-6 w-full overflow-hidden rounded bg-bg-tertiary">
        {KILL_ZONES.map((z) => {
          const left = (z.start / DAY_MIN) * 100;
          const width = ((z.end - z.start) / DAY_MIN) * 100;
          const active = now >= z.start && now < z.end;
          return (
            <div
              key={z.label}
              data-testid={`kz-${z.label.toLowerCase().replace(/ /g, '-')}`}
              data-active={active}
              className={`absolute top-0 h-full ${
                active ? 'bg-accent-cyan/60' : 'bg-accent-cyan/20'
              }`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={z.label}
            />
          );
        })}
        <div
          className="absolute top-0 h-full w-px bg-accent-gold"
          style={{ left: `${(now / DAY_MIN) * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-white/40">
        <span>00</span><span>06</span><span>12</span><span>18</span><span>24</span>
      </div>
    </section>
  );
}
