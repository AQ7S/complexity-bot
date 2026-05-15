import { createMemoryRouter, createHashRouter, RouterProvider, Outlet } from 'react-router-dom';
import Sidebar from '@/components/Sidebar';
import PositionSizeCalc from '@/components/PositionSizeCalc';
import CommandCenter from '@/pages/CommandCenter';
import Charts from '@/pages/Charts';
import TradeJournal from '@/pages/TradeJournal';
import AIEngine from '@/pages/AIEngine';
import Settings from '@/pages/Settings';
import SignalScanner from '@/pages/SignalScanner';
import DecisionTrace from '@/pages/DecisionTrace';
import { useEngineSocket } from '@/hooks/useEngineSocket';

function Shell() {
  useEngineSocket();
  return (
    <div className="flex h-screen w-screen bg-bg-primary text-white">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
      <PositionSizeCalc />
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
      { path: '/settings', element: <Settings /> },
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
