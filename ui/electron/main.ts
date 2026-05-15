/**
 * Complexity Engine — Electron main process.
 *
 * Owns:
 *   - the 1440×900 dark window (single instance, restored on tray click)
 *   - the system tray (Show / Hide / Quit) with close-to-tray semantics
 *     so closing the X button keeps the engine WS alive
 *   - the WebSocket client to the Python engine on 127.0.0.1:8765
 *   - native Windows toast dispatch when the engine emits `notification` frames
 *   - the AppUserModelId so toasts surface under our brand
 *
 * Renderer ↔ engine flow: renderer → ipcRenderer.send('engine:cmd', frame)
 * → main → EngineWS.send. Inbound: EngineWS → webContents.send('engine:event', frame)
 * → preload contextBridge → renderer subscriber.
 */
import { app, BrowserWindow, Tray, ipcMain } from 'electron';
import path from 'node:path';
import { createTray, destroyTray } from './tray';
import { EngineWS } from './ws_client';
import { dispatchNotification } from './notifications';
import { startEngine, stopEngine } from './engine_process';
import { wireAutoUpdate } from './auto_update';

process.env.APP_USER_MODEL_ID = process.env.APP_USER_MODEL_ID ?? 'com.complexity.engine';
app.setAppUserModelId(process.env.APP_USER_MODEL_ID);

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let engineWS: EngineWS | null = null;

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
    }
  });
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: '#0a0e1a',
    autoHideMenuBar: true,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // Close-to-tray: hide instead of destroying unless we're really quitting.
  mainWindow.on('close', (e) => {
    if (!(app as any).isQuittingForReal) {
      e.preventDefault();
      mainWindow?.hide();
    }
  });

  const devServer = process.env.VITE_DEV_SERVER_URL;
  if (devServer) {
    void mainWindow.loadURL(devServer);
  } else {
    void mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

function setupIpc(): void {
  ipcMain.handle('engine:send', (_evt, frame: object) => {
    return engineWS?.send(frame) ?? false;
  });
  ipcMain.handle('app:show-window', () => {
    if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
  });
  ipcMain.handle('app:quit', () => {
    (app as any).isQuittingForReal = true;
    app.quit();
  });
}

app.whenReady().then(() => {
  createWindow();
  tray = createTray(() => mainWindow);
  setupIpc();
  startEngine((line) => console.log(line));
  engineWS = new EngineWS(() => mainWindow);
  engineWS.start();
  wireAutoUpdate(() => mainWindow);

  // Bridge engine notifications to native Windows toasts.
  if (mainWindow) {
    const wc = mainWindow.webContents;
    const subscriber = (_evt: Electron.IpcMainEvent, frame: any) => dispatchNotification(frame);
    ipcMain.on('engine:notification-out', subscriber);
    // Also dispatch directly when EngineWS emits — keep main as source of truth.
    wc.on('did-finish-load', () => {
      // No-op; renderer subscribes via preload.
    });
  }
});

app.on('window-all-closed', () => {
  // Stay alive in tray on Windows/Linux when the user closes the window.
  if (process.platform === 'darwin') app.quit();
});

app.on('before-quit', () => {
  (app as any).isQuittingForReal = true;
  engineWS?.stop();
  stopEngine();
  destroyTray();
  void tray; // keep a ref so tsc doesn't drop the import
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
  else mainWindow?.show();
});
