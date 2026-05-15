import { describe, it, expect, beforeEach } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TestRouter } from '@/router';
import { useEngineStore } from '@/store/engineStore';

beforeEach(() => {
  useEngineStore.setState({
    wsConnected: false, engineStatus: null, account: null, weeklyPnl: 0,
    ticks: {}, positions: {}, signals: [], claudeFeed: [], regimes: {}, correlation: null,
    tradesHistory: [], settingsKv: {},
  });
});

describe('PositionSizeCalc', () => {
  it('matches the Appendix-E reference example', () => {
    render(<TestRouter initialPath="/" />);
    act(() => {
      useEngineStore.getState().setAccount({
        equity: 10_000, balance: 10_000, free_margin: 9_900,
        drawdown_pct: 0, open_positions: 0,
      });
    });
    fireEvent.click(screen.getByTestId('position-size-toggle'));
    // Defaults already match the EURUSD example: entry 1.073 / sl 1.072 /
    // tickSize 1e-5 / tickValue 1.0 / riskPct 2.0 / equity 10_000 → lot 2.00.
    expect(screen.getByTestId('calc-lot')).toHaveTextContent('2.00');
  });
});
