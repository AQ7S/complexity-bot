import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TestRouter } from '@/router';
import { useEngineStore } from '@/store/engineStore';
import type { TradeRow } from '@/types/ipc-messages';

const SAMPLE: TradeRow[] = [
  {
    id: 1, mt5_ticket: 1001, symbol: 'EURUSD', direction: 'BUY',
    entry_price: 1.07, exit_price: 1.075, lot_size: 0.5,
    sl: 1.065, tp: 1.080, pnl: 25.0, rr_achieved: 1.0,
    open_time: '2026-05-04T12:00:00Z', close_time: '2026-05-04T12:30:00Z',
    close_reason: 'TP', signal_confluence: 4,
    claude_decision: 'BUY', claude_confidence: 78,
    claude_reasoning: 'Bullish OB with NY open momentum.',
  },
  {
    id: 2, mt5_ticket: 1002, symbol: 'GBPUSD', direction: 'SELL',
    entry_price: 1.252, exit_price: 1.255, lot_size: 0.3,
    sl: 1.260, tp: 1.245, pnl: -18.0, rr_achieved: -0.5,
    open_time: '2026-05-04T13:00:00Z', close_time: '2026-05-04T13:45:00Z',
    close_reason: 'SL', signal_confluence: 3,
    claude_decision: 'SELL', claude_confidence: 60, claude_reasoning: null,
  },
];

beforeEach(() => {
  useEngineStore.setState({
    wsConnected: false, engineStatus: null, account: null, weeklyPnl: 0,
    ticks: {}, positions: {}, signals: [], claudeFeed: [], regimes: {}, correlation: null,
    tradesHistory: [], settingsKv: {},
  });
});

describe('TradeJournal', () => {
  it('renders rows from the trades snapshot + computed metrics', () => {
    render(<TestRouter initialPath="/journal" />);
    act(() => { useEngineStore.getState().setTradesHistory(SAMPLE); });
    const tbl = screen.getByTestId('trades-table');
    expect(tbl).toHaveTextContent('1001');
    expect(tbl).toHaveTextContent('EURUSD');
    expect(tbl).toHaveTextContent('+$25.00');
    expect(tbl).toHaveTextContent('-$18.00');
    expect(screen.getByTestId('metric-cards')).toHaveTextContent('50.0%');
  });

  it('expands the Claude reasoning row on click', () => {
    render(<TestRouter initialPath="/journal" />);
    act(() => { useEngineStore.getState().setTradesHistory(SAMPLE); });
    fireEvent.click(screen.getByTestId('trade-row-1'));
    expect(screen.getByTestId('trade-reasoning-1')).toHaveTextContent('Bullish OB');
  });

  it('exports CSV via the download button', () => {
    const click = vi.fn();
    const orig = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function (this: HTMLAnchorElement) {
      click({ href: this.href, download: this.download });
    };
    (URL as any).createObjectURL = vi.fn(() => 'blob:mock');
    (URL as any).revokeObjectURL = vi.fn();
    try {
      render(<TestRouter initialPath="/journal" />);
      act(() => { useEngineStore.getState().setTradesHistory(SAMPLE); });
      fireEvent.click(screen.getByTestId('export-csv'));
      expect(click).toHaveBeenCalledTimes(1);
      expect(click.mock.calls[0][0].download).toMatch(/^trades-\d{4}-\d{2}-\d{2}\.csv$/);
    } finally {
      HTMLAnchorElement.prototype.click = orig;
    }
  });
});
