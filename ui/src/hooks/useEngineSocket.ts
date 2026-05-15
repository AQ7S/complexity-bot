import { useEffect } from 'react';
import { useEngineStore } from '@/store/engineStore';
import type { Frame } from '@/types/ipc-messages';

/**
 * Subscribes to the preload bridge once and routes frames to Zustand.
 * No-op when `window.engineBridge` is absent (browser tests, vite dev).
 */
export function useEngineSocket(): void {
  useEffect(() => {
    const bridge = (window as any).engineBridge;
    if (!bridge?.onEvent) return;
    const off = bridge.onEvent((frame: Frame) => {
      const st = useEngineStore.getState();
      switch (frame.type) {
        case 'ui:ws_status':       st.setWS(frame.data as any); break;
        case 'engine_status':      st.setEngineStatus(frame.data as any); break;
        case 'account_update':     st.setAccount(frame.data as any); break;
        case 'tick_update':        st.setTick(frame.data as any); break;
        case 'trade_opened':       st.upsertPositionOpened(frame.data as any); break;
        case 'trade_updated':      st.applyTradeUpdate(frame.data as any); break;
        case 'trade_closed':       st.closePosition(frame.data as any); break;
        case 'signal_detected':    st.pushSignal(frame.data as any); break;
        case 'claude_feed':        st.pushClaude(frame.data as any); break;
        case 'model_update':       st.setModelUpdate(frame.data as any); break;
        case 'regime_change':      st.setRegime(frame.data as any); break;
        case 'correlation_update': st.setCorrelation(frame.data as any); break;
        case 'trades_snapshot':    st.setTradesHistory((frame.data as any).trades ?? []); break;
        case 'settings_snapshot':  st.setSettingsKv((frame.data as any).values ?? {}); break;
      }
    });
    return off;
  }, []);
}

export function sendCommand(type: string, data: object = {}): Promise<boolean> {
  const bridge = (window as any).engineBridge;
  if (!bridge?.send) return Promise.resolve(false);
  return bridge.send({ type, ts: Date.now(), data });
}
