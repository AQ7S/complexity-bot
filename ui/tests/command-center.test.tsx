import { describe, it, expect, beforeEach } from 'vitest';
import { act } from 'react';
import { render, screen } from '@testing-library/react';
import { TestRouter } from '@/router';
import { useEngineStore } from '@/store/engineStore';
import { SYMBOLS_13 } from '@/lib/constants';

beforeEach(() => {
  useEngineStore.setState({
    wsConnected: false, engineStatus: null, account: null, weeklyPnl: 0,
    ticks: {}, positions: {}, signals: [], claudeFeed: [], regimes: {}, correlation: null,
  });
});

describe('CommandCenter', () => {
  it('renders all 13 SymbolCards', () => {
    render(<TestRouter initialPath="/" />);
    const grid = screen.getByTestId('symbol-grid');
    expect(grid).toBeInTheDocument();
    for (const { name } of SYMBOLS_13) {
      expect(screen.getByTestId(`symbol-card-${name}`)).toBeInTheDocument();
    }
  });

  it('reflects tick updates within the symbol card', () => {
    render(<TestRouter initialPath="/" />);
    act(() => {
      useEngineStore.getState().setTick({
        symbol: 'EURUSD#', bid: 1.07321, ask: 1.07323, spread: 0.00002,
      });
    });
    const card = screen.getByTestId('symbol-card-EURUSD#');
    expect(card).toHaveTextContent('1.07321');
    expect(card).toHaveTextContent('1.07323');
  });

  it('shows hero bar status and equity', () => {
    render(<TestRouter initialPath="/" />);
    act(() => {
      useEngineStore.setState({ wsConnected: true });
      useEngineStore.getState().setEngineStatus({
        status: 'LIVE', uptime_s: 100, mt5_connected: true,
      });
      useEngineStore.getState().setAccount({
        equity: 10_050, balance: 10_000, free_margin: 9_900,
        drawdown_pct: 0.005, open_positions: 1,
      });
    });
    expect(screen.getByTestId('hero-bar')).toHaveTextContent('LIVE');
    expect(screen.getByTestId('hero-bar')).toHaveTextContent('10,050.00');
  });

  it('renders open positions when present', () => {
    render(<TestRouter initialPath="/" />);
    act(() => {
      useEngineStore.getState().upsertPositionOpened({
        ticket: 12345, symbol: 'EURUSD#', direction: 'BUY',
        entry: 1.07, sl: 1.06, tp: 1.08, lot: 0.5,
      });
      useEngineStore.getState().applyTradeUpdate({
        ticket: 12345, current_price: 1.075, pnl: 25.0, rr_current: 0.5,
      });
    });
    const tbl = screen.getByTestId('positions-table');
    expect(tbl).toHaveTextContent('12345');
    expect(tbl).toHaveTextContent('EURUSD#');
    expect(tbl).toHaveTextContent('+$25.00');
  });

  it('appends to the Claude feed', () => {
    render(<TestRouter initialPath="/" />);
    act(() => {
      useEngineStore.getState().pushClaude({
        trade_id: null, symbol: 'GBPUSD', decision: 'SKIP',
        confidence: 35, reasoning_excerpt: 'Conflicting signals.',
      });
    });
    const feed = screen.getByTestId('claude-feed');
    expect(feed).toHaveTextContent('GBPUSD');
    expect(feed).toHaveTextContent('SKIP');
    expect(feed).toHaveTextContent('Conflicting signals.');
  });
});
