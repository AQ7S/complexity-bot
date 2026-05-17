import { useEffect, useCallback, useState } from 'react';
import { createMemoryRouter, createHashRouter, RouterProvider, Outlet, useNavigate } from 'react-router-dom';
import Sidebar from '@/components/Sidebar';
import PositionSizeCalc from '@/components/PositionSizeCalc';
import CommandCenter from '@/pages/CommandCenter';
import Charts from '@/pages/Charts';
import TradeJournal from '@/pages/TradeJournal';
import AIEngine from '@/pages/AIEngine';
import Settings from '@/pages/Settings';
import SignalScanner from '@/pages/SignalScanner';
import DecisionTrace from '@/pages/DecisionTrace';
import SpreadMonitor from '@/pages/SpreadMonitor';
import SessionPnLHeatmap from '@/pages/SessionPnLHeatmap';
import BacktestRunner from '@/pages/BacktestRunner';
import Strategies from '@/pages/Strategies';
import { useEngineSocket, sendCommand } from '@/hooks/useEngineSocket';
import { useEngineStore } from '@/store/engineStore';

const NAV_KEYS: Record<string, string> = {
  '1': '/',
  '2': '/charts',
  '3': '/scanner',
  '4': '/trace',
  '5': '/journal',
  '6': '/ai',
  '7': '/spread',
  '8': '/heatmap',
  '9': '/backtest',
  '0': '/settings',
};

function NewsWarningBanner() {
  const warn = useEngineStore((s) => s.newsWarning);
  const dismiss = useEngineStore((s) => s.dismissNewsWarning);
  if (!warn) return null;
  const impactColor =
    warn.impact === 'HIGH'   ? 'border-accent-red/60 bg-accent-red/10 text-accent-red' :
    warn.impact === 'MEDIUM' ? 'border-accent-gold/60 bg-accent-gold/10 text-accent-gold' :
                               'border-white/20 bg-white/5 text-white/70';
  return (
    <div className={`flex items-center gap-3 border-b px-4 py-2 text-xs font-mono ${impactColor}`}
         data-testid="news-warning-banner">
      <span className="shrink-0 font-bold">📰 NEWS {warn.impact}</span>
      <span className="flex-1 truncate">
        {warn.event_name} ({warn.currency}) — {warn.time_until_minutes}m away
        {warn.affected_symbols?.length
          ? ` · ${warn.affected_symbols.slice(0, 4).join(', ')}`
          : ''}
      </span>
      <button type="button" onClick={dismiss}
              className="shrink-0 rounded px-2 py-0.5 text-[10px] opacity-60 hover:opacity-100 hover:bg-white/10">
        ✕
      </button>
    </div>
  );
}

function Shell() {
  useEngineSocket();
  const navigate = useNavigate();
  const [confirmEmergency, setConfirmEmergency] = useState(false);

  const handleKey = useCallback((e: KeyboardEvent) => {
    // Skip if user is typing in an input/textarea/select
    const tag = (e.target as HTMLElement).tagName;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;

    // Ctrl+number → navigate
    if (e.ctrlKey && NAV_KEYS[e.key]) {
      e.preventDefault();
      navigate(NAV_KEYS[e.key]);
      return;
    }

    switch (e.key) {
      case ' ': {
        e.preventDefault();
        void sendCommand('cmd_pause', { paused: true });
        break;
      }
      case 'e':
      case 'E': {
        if (!confirmEmergency) {
          setConfirmEmergency(true);
          setTimeout(() => setConfirmEmergency(false), 3000);
        } else {
          setConfirmEmergency(false);
          void sendCommand('cmd_emergency_close', {});
        }
        break;
      }
      case 'r':
      case 'R': {
        window.location.reload();
        break;
      }
    }
  }, [navigate, confirmEmergency]);

  useEffect(() => {
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [handleKey]);

  return (
    <div className="flex h-screen w-screen flex-col bg-bg-primary text-white">
      <NewsWarningBanner />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
      <PositionSizeCalc />
      {/* Emergency confirmation toast */}
      {confirmEmergency && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 animate-pulse rounded-lg border border-accent-red/50 bg-accent-red/20 px-6 py-3 text-sm font-bold text-accent-red shadow-xl">
          Press E again to confirm EMERGENCY CLOSE ALL
        </div>
      )}
    </div>
  );
}

const ROUTES = [
  {
    element: <Shell />,
    children: [
      { path: '/',         element: <CommandCenter /> },
      { path: '/charts',   element: <Charts /> },
      { path: '/journal',  element: <TradeJournal /> },
      { path: '/ai',       element: <AIEngine /> },
      { path: '/scanner',  element: <SignalScanner /> },
      { path: '/trace',    element: <DecisionTrace /> },
      { path: '/spread',   element: <SpreadMonitor /> },
      { path: '/heatmap',  element: <SessionPnLHeatmap /> },
      { path: '/backtest',   element: <BacktestRunner /> },
      { path: '/strategies', element: <Strategies /> },
      { path: '/settings',   element: <Settings /> },
    ],
  },
];

export function AppRouter() {
  return <RouterProvider router={createHashRouter(ROUTES)} />;
}

// Test-only entry: lets Vitest mount any route deterministically without
// touching the browser history API.
export function TestRouter({ initialPath = '/' }: { initialPath?: string }) {
  const router = createMemoryRouter(ROUTES, { initialEntries: [initialPath] });
  return <RouterProvider router={router} />;
}
