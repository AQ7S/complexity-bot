/**
 * Native Windows toast dispatcher (main process).
 *
 * Listens for `notification`-typed frames from the engine and renders an
 * Electron `Notification` with the AppUserModelId set so toasts appear in
 * the Action Center under the Complexity Engine identity. Sound playback
 * happens in the renderer (HTML5 Audio over the bundled WAV files in
 * /public/sounds/) since main-process audio APIs are unreliable on Win.
 */
import { Notification } from 'electron';

export function dispatchNotification(frame: { type: string; data: any }): void {
  if (frame.type !== 'notification' || !frame.data) return;
  const { event, title, body } = frame.data;
  if (!Notification.isSupported()) return;
  const n = new Notification({ title: title ?? 'Complexity Engine', body: body ?? '', silent: true });
  n.show();
  // We don't block on user click; tag the event so logs can correlate.
  n.on('click', () => {
    // Window focus is handled by the tray click handler / ipcMain in main.ts.
    // Hooking 'show:main-window' here would create a circular dependency.
  });
  void event;
}
