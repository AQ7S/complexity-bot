import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TestRouter } from '@/router';

beforeEach(() => {
  (window as any).engineBridge = {
    version: 'test',
    onEvent: () => () => {},
    send: vi.fn().mockResolvedValue(true),
    showWindow: vi.fn(), quit: vi.fn(),
  };
});

describe('AIEngine', () => {
  it('lists both model cards', () => {
    render(<TestRouter initialPath="/ai" />);
    expect(screen.getByTestId('model-card-CNN-LSTM')).toBeInTheDocument();
    expect(screen.getByTestId('model-card-RL DQN')).toBeInTheDocument();
  });

  it('opens the confirmation dialog and dispatches retrain', async () => {
    render(<TestRouter initialPath="/ai" />);
    fireEvent.click(screen.getByTestId('retrain-CNN-LSTM'));
    expect(screen.getByTestId('retrain-confirm')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('retrain-confirm-btn'));
    await act(() => Promise.resolve());
    const send = (window as any).engineBridge.send;
    const call = send.mock.calls.find((c: any[]) => c[0]?.type === 'cmd_manual_retrain');
    expect(call?.[0]?.data?.model).toBe('cnn_lstm');
  });
});
