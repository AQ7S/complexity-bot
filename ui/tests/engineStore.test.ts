import { describe, it, expect, beforeEach } from 'vitest';
import { useEngineStore } from '@/store/engineStore';

describe('engineStore', () => {
  beforeEach(() => {
    useEngineStore.setState({
      wsConnected: false, engineStatus: null, account: null, ticks: {},
    });
  });

  it('updates ws status', () => {
    useEngineStore.getState().setWS({ connected: true });
    expect(useEngineStore.getState().wsConnected).toBe(true);
  });

  it('captures account snapshot', () => {
    useEngineStore.getState().setAccount({
      equity: 10_050, balance: 10_000, free_margin: 9_800,
      drawdown_pct: 0.005, open_positions: 1,
    });
    expect(useEngineStore.getState().account?.equity).toBe(10_050);
  });

  it('keeps last tick per symbol', () => {
    const { setTick } = useEngineStore.getState();
    setTick({ symbol: 'EURUSD', bid: 1.07, ask: 1.0701, spread: 0.0001 });
    setTick({ symbol: 'EURUSD', bid: 1.071, ask: 1.0711, spread: 0.0001 });
    setTick({ symbol: 'GBPUSD', bid: 1.25, ask: 1.2501, spread: 0.0001 });
    const ticks = useEngineStore.getState().ticks;
    expect(ticks.EURUSD.bid).toBe(1.071);
    expect(ticks.GBPUSD.bid).toBe(1.25);
  });
});
