/**
 * Preload — exposes a typed `engineBridge` to the renderer.
 *
 * The renderer can:
 *   - subscribe to inbound engine frames via `onEvent()`
 *   - send commands back through the WS via `send()`
 *   - request main-process actions (show window, quit)
 */
import { contextBridge, ipcRenderer } from 'electron';

type Frame = { type: string; ts: number; data: any };
type Listener = (frame: Frame) => void;

const listeners = new Set<Listener>();

ipcRenderer.on('engine:event', (_evt, frame: Frame) => {
  for (const l of listeners) {
    try { l(frame); } catch { /* swallow per-listener faults */ }
  }
});

contextBridge.exposeInMainWorld('engineBridge', {
  version: '1.0.0',
  onEvent(listener: Listener): () => void {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  },
  send(frame: object): Promise<boolean> {
    return ipcRenderer.invoke('engine:send', frame);
  },
  showWindow(): Promise<void> {
    return ipcRenderer.invoke('app:show-window');
  },
  quit(): Promise<void> {
    return ipcRenderer.invoke('app:quit');
  },
});

declare global {
  interface Window {
    engineBridge: {
      version: string;
      onEvent(listener: Listener): () => void;
      send(frame: object): Promise<boolean>;
      showWindow(): Promise<void>;
      quit(): Promise<void>;
    };
  }
}
