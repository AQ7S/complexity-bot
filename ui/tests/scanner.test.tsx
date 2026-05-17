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

describe('SignalScanner', () => {
  it('lists every symbol as a row', () => {
    render(<TestRouter initialPath="/scanner" />);
    for (const { name } of SYMBOLS_13) {
      expect(screen.getByTestId(`scanner-row-${name}`)).toBeInTheDocument();
    }
  });

  it('shows the latest direction when a signal arrives', () => {
    render(<TestRouter initialPath="/scanner" />);
    act(() => {
      useEngineStore.getState().pushSignal({
        signal_id: 'a', symbol: 'EURUSD#', timeframe: 'M5',
        direction: 'BUY', confluence: 4,
        sources: { smc: 'BUY', cnn: 'BUY', rl: 'HOLD', killzone: true, news_clear: true },
        claude: null,
      });
    });
    const row = screen.getByTestId('scanner-row-EURUSD#');
    expect(row).toHaveTextContent('BUY');
    expect(row).toHaveTextContent('STRONG');
  });
});
