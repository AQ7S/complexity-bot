import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TestRouter } from '@/router';
import { useEngineStore } from '@/store/engineStore';
import type { StrategyStatus } from '@/types/ipc-messages';

const SAMPLE: StrategyStatus = {
  total_risk_pct: 0.02,
  strategies: [
    {
      name: 'scalping', style: 'scalp', state: 'ACTIVE',
      weight: 0.30, rolling_sharpe: 1.20, consecutive_losses: 0,
      trades_today: 3, pnl_today_usd: 42.5,
      paused_until_ts: 0, shadow_only_until_ts: 0,
    },
    {
      name: 'breakout', style: 'breakout', state: 'SHADOW',
      weight: 0.20, rolling_sharpe: -0.1, consecutive_losses: 2,
      trades_today: 1, pnl_today_usd: -10.0,
      paused_until_ts: 0, shadow_only_until_ts: 0,
    },
    {
      name: 'carry', style: 'carry', state: 'DISABLED',
      weight: 0.0, rolling_sharpe: 0.0, consecutive_losses: 0,
      trades_today: 0, pnl_today_usd: 0.0,
      paused_until_ts: 0, shadow_only_until_ts: 0,
    },
  ],
};

beforeEach(() => {
  useEngineStore.setState({ strategyStatus: SAMPLE });
  // Stub the bridge so sendCommand doesn't no-op silently.
  (window as any).engineBridge = {
    send: vi.fn().mockResolvedValue(true),
    onEvent: () => () => {},
  };
});

describe('Strategies page', () => {
  it('renders cards for each strategy in the snapshot', () => {
    render(<TestRouter initialPath="/strategies" />);
    expect(screen.getByTestId('page-strategies')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-card-scalping')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-card-breakout')).toBeInTheDocument();
    expect(screen.getByTestId('strategy-card-carry')).toBeInTheDocument();
  });

  it('shows state badges from the snapshot', () => {
    render(<TestRouter initialPath="/strategies" />);
    expect(screen.getByTestId('strategy-state-scalping').textContent).toBe('ACTIVE');
    expect(screen.getByTestId('strategy-state-breakout').textContent).toBe('SHADOW');
    expect(screen.getByTestId('strategy-state-carry').textContent).toBe('DISABLED');
  });

  it('dispatches cmd_strategy_toggle when Off button is pressed', () => {
    render(<TestRouter initialPath="/strategies" />);
    const card = screen.getByTestId('strategy-card-scalping');
    const offButton = Array.from(card.querySelectorAll('button')).find(
      (b) => b.textContent?.toLowerCase().trim() === 'off',
    );
    expect(offButton).toBeDefined();
    fireEvent.click(offButton!);
    expect((window as any).engineBridge.send).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'cmd_strategy_toggle',
        data: { name: 'scalping', mode: 'OFF' },
      }),
    );
  });

  it('shows placeholder when no snapshot is available', () => {
    useEngineStore.setState({ strategyStatus: null });
    render(<TestRouter initialPath="/strategies" />);
    expect(screen.getByText(/Waiting for strategy_status frame/)).toBeInTheDocument();
  });
});
