import { describe, it, expect, beforeEach, vi } from 'vitest';
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
  // Mock the engine bridge so cmd_emergency_close has somewhere to land.
  (window as any).engineBridge = {
    version: 'test',
    onEvent: () => () => {},
    send: vi.fn().mockResolvedValue(true),
    showWindow: vi.fn(), quit: vi.fn(),
  };
});

describe('Settings page', () => {
  it('renders the encrypted-creds form + kill-zone display', () => {
    render(<TestRouter initialPath="/settings" />);
    expect(screen.getByTestId('settings-creds')).toHaveTextContent('Fernet');
    expect(screen.getByTestId('field-MT5_LOGIN')).toBeInTheDocument();
    expect(screen.getByText(/Kill Zones/i)).toBeInTheDocument();
  });

  it('Emergency Stop sends cmd_emergency_close', async () => {
    render(<TestRouter initialPath="/settings" />);
    fireEvent.click(screen.getByTestId('emergency-stop'));
    // wait a tick for the resolved promise
    await act(() => Promise.resolve());
    const send = (window as any).engineBridge.send;
    const calls = send.mock.calls.map((c: any[]) => c[0]);
    expect(calls.some((c: any) => c?.type === 'cmd_emergency_close')).toBe(true);
  });

  it('Save sends a cmd_settings_update with only changed fields', async () => {
    render(<TestRouter initialPath="/settings" />);
    fireEvent.change(screen.getByTestId('field-MT5_LOGIN'), { target: { value: '12345678' } });
    fireEvent.click(screen.getByTestId('save-settings'));
    await act(() => Promise.resolve());
    const send = (window as any).engineBridge.send;
    const update = send.mock.calls.find((c: any[]) => c[0]?.type === 'cmd_settings_update')?.[0];
    expect(update?.data?.partial).toEqual({ MT5_LOGIN: '12345678' });
  });
});
