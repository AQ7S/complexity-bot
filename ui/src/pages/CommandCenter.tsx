import HeroBar from '@/components/HeroBar';
import SymbolCard from '@/components/SymbolCard';
import PositionsTable from '@/components/PositionsTable';
import KillZoneTimeline from '@/components/KillZoneTimeline';
import ClaudeFeed from '@/components/ClaudeFeed';
import CorrelationHeatmap from '@/components/CorrelationHeatmap';
import DrawdownChart from '@/components/DrawdownChart';
import { SYMBOLS_13 } from '@/lib/constants';

export default function CommandCenter() {
  return (
    <section data-testid="page-command-center" className="flex h-full flex-col">
      <HeroBar />
      <div className="grid flex-1 grid-cols-12 gap-4 overflow-auto p-4">
        <div className="col-span-8 space-y-4">
          <div
            data-testid="symbol-grid"
            className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5"
          >
            {SYMBOLS_13.map((s) => (
              <SymbolCard key={s.name} symbol={s.name} kind={s.kind} />
            ))}
          </div>
          <PositionsTable />
          <KillZoneTimeline />
        </div>
        <div className="col-span-4 space-y-4">
          <DrawdownChart />
          <ClaudeFeed />
          <CorrelationHeatmap />
        </div>
      </div>
    </section>
  );
}
