/**
 * Engine WebSocket client — runs in the Electron main process.
 *
 * Owns the single WS connection to the Python engine on 127.0.0.1:8765,
 * relays inbound frames to the renderer via `webContents.send('engine:event', frame)`,
 * and accepts outbound commands via the `engine:cmd` IPC channel routed
 * back through the preload contextBridge.
 *
 * Reconnects with exponential backoff (1s → 30s cap) so the UI auto-recovers
 * when the engine restarts (e.g., after Task Scheduler relaunch).
 */
import WebSocket from 'ws';
import type { BrowserWindow } from 'electron';

const ENGINE_URL = process.env.ENGINE_WS_URL ?? 'ws://127.0.0.1:8765';
const RECONNECT_BASE_MS = 1000;
const RECONNECT_CAP_MS = 30_000;

export class EngineWS {
  private ws: WebSocket | null = null;
  private getWindow: () => BrowserWindow | null;
  private retryAttempt = 0;
  private retryTimer: NodeJS.Timeout | null = null;
  private stopped = false;

  constructor(getWindow: () => BrowserWindow | null) {
    this.getWindow = getWindow;
  }

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
    if (this.ws) { this.ws.close(); this.ws = null; }
  }

  send(frame: object): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(frame));
      return true;
    }
    return false;
  }

  private connect(): void {
    if (this.stopped) return;
    const ws = new WebSocket(ENGINE_URL);
    this.ws = ws;
    ws.on('open', () => {
      this.retryAttempt = 0;
      this.broadcast({ type: 'ui:ws_status', ts: Date.now(), data: { connected: true } });
    });
    ws.on('message', (raw) => {
      let frame: any;
      try { frame = JSON.parse(raw.toString()); } catch { return; }
      this.broadcast(frame);
    });
    ws.on('close', () => {
      this.broadcast({ type: 'ui:ws_status', ts: Date.now(), data: { connected: false } });
      this.scheduleReconnect();
    });
    ws.on('error', () => {
      // 'close' will follow; reconnect handled there.
    });
  }

  private scheduleReconnect(): void {
    if (this.stopped) return;
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.retryAttempt, RECONNECT_CAP_MS);
    this.retryAttempt += 1;
    this.retryTimer = setTimeout(() => this.connect(), delay);
  }

  private broadcast(frame: object): void {
    const w = this.getWindow();
    if (w && !w.isDestroyed()) {
      w.webContents.send('engine:event', frame);
    }
  }
}
