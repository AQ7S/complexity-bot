import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, BookOpen, Brain, Settings as Cog, Radar, Workflow, Gauge, Grid3X3, FlaskConical } from 'lucide-react';

const ROUTES = [
  { to: '/',          label: 'Command',  icon: Activity },
  { to: '/charts',    label: 'Charts',   icon: BarChart3 },
  { to: '/scanner',   label: 'Scanner',  icon: Radar },
  { to: '/trace',     label: 'Reasoning',icon: Workflow },
  { to: '/journal',   label: 'Journal',  icon: BookOpen },
  { to: '/ai',        label: 'AI',       icon: Brain },
  { to: '/spread',    label: 'Spreads',  icon: Gauge },
  { to: '/heatmap',   label: 'Heatmap',  icon: Grid3X3 },
  { to: '/backtest',  label: 'Backtest', icon: FlaskConical },
  { to: '/settings',  label: 'Settings', icon: Cog },
] as const;

export default function Sidebar() {
  return (
    <nav data-testid="sidebar" className="flex h-full w-48 flex-col gap-1 bg-bg-secondary p-3">
      <div className="mb-4 px-2 font-hero text-sm tracking-wider text-accent-cyan">
        COMPLEXITY
      </div>
      {ROUTES.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `flex items-center gap-2 rounded px-3 py-2 text-sm transition-colors ${
              isActive
                ? 'bg-bg-tertiary text-accent-cyan'
                : 'text-white/70 hover:bg-bg-tertiary/60 hover:text-white'
            }`
          }
        >
          <Icon size={16} />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
