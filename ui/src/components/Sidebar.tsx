import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, BookOpen, Brain, Settings as Cog, Radar, Workflow, Gauge, Grid3X3, FlaskConical, Layers } from 'lucide-react';
import { useEngineStore } from '@/store/engineStore';
import { fmtUsd } from '@/lib/format';
import NotificationBell from './NotificationBell';

const ROUTES = [
  { to: '/',          label: 'Command',  icon: Activity },
  { to: '/charts',    label: 'Charts',   icon: BarChart3 },
  { to: '/scanner',   label: 'Scanner',  icon: Radar },
  { to: '/trace',     label: 'Reasoning',icon: Workflow },
  { to: '/journal',   label: 'Journal',  icon: BookOpen },
  { to: '/ai',        label: 'AI',       icon: Brain },
  { to: '/spread',    label: 'Spreads',  icon: Gauge },
  { to: '/heatmap',   label: 'Heatmap',  icon: Grid3X3 },
  { to: '/backtest',    label: 'Backtest',  icon: FlaskConical },
  { to: '/strategies',  label: 'Strategies', icon: Layers },
  { to: '/settings',    label: 'Settings',  icon: Cog },
] as const;

function SidebarFooter() {
  const status = useEngineStore((s) => s.engineStatus);
  const wsConnected = useEngineStore((s) => s.wsConnected);
  const account = useEngineStore((s) => s.account);

  const live = status?.status === 'LIVE';
  const error = status?.status === 'ERROR' || !wsConnected;
  const starting = !status || status.status === 'STARTING';
  const dot = error ? 'bg-accent-red' : starting ? 'bg-accent-gold' : live ? 'bg-accent-green' : 'bg-white/40';

  return (
    <div className="mt-auto border-t border-white/5 pt-2">
      {account && (
        <div className="mb-1 px-3 py-1">
          <p className="text-[9px] uppercase tracking-wider text-white/40">Equity</p>
          <p className="font-mono text-xs text-accent-green">{fmtUsd(account.equity)}</p>
        </div>
      )}
      <div className="flex items-center gap-2 px-2 py-1.5">
        <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} />
        <span className="flex-1 truncate text-[10px] uppercase tracking-wider text-white/50">
          {status?.status ?? 'STARTING'}
        </span>
        <NotificationBell dropdownClassName="left-full ml-2 bottom-0" />
      </div>
    </div>
  );
}

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
      <SidebarFooter />
    </nav>
  );
}
