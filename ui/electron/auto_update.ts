import { autoUpdater } from 'electron-updater';
import { app, dialog, BrowserWindow } from 'electron';

let _wired = false;

export function wireAutoUpdate(getWindow: () => BrowserWindow | null): void {
  if (_wired) return;
  _wired = true;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.allowDowngrade = false;

  autoUpdater.on('error', (err) => {
    console.error('[updater] error:', err?.message ?? err);
  });
  autoUpdater.on('checking-for-update', () => {
    console.log('[updater] checking for update…');
  });
  autoUpdater.on('update-available', (info) => {
    console.log(`[updater] update available: ${info?.version}`);
    getWindow()?.webContents.send('engine:event', {
      type: 'ui:update_available',
      ts: Date.now(),
      data: { version: info?.version },
    });
  });
  autoUpdater.on('update-not-available', () => {
    console.log('[updater] up to date');
  });
  autoUpdater.on('download-progress', (p) => {
    console.log(`[updater] downloading ${p.percent.toFixed(1)}%`);
    getWindow()?.webContents.send('engine:event', {
      type: 'ui:update_progress',
      ts: Date.now(),
      data: { percent: p.percent, bytes_per_s: p.bytesPerSecond },
    });
  });
  autoUpdater.on('update-downloaded', (info) => {
    console.log(`[updater] downloaded ${info?.version}; will install on quit`);
    const w = getWindow();
    if (!w) return;
    void dialog
      .showMessageBox(w, {
        type: 'info',
        buttons: ['Restart now', 'Later'],
        defaultId: 0,
        title: 'Update ready',
        message: `Complexity Engine ${info?.version} is ready to install.`,
        detail: 'The app will restart to apply the update.',
      })
      .then((res) => {
        if (res.response === 0) {
          (app as any).isQuittingForReal = true;
          autoUpdater.quitAndInstall();
        }
      });
  });

  if (!app.isPackaged) {
    console.log('[updater] dev mode — checks skipped');
    return;
  }
  void autoUpdater.checkForUpdates().catch((err) => {
    console.warn('[updater] initial check failed:', err?.message ?? err);
  });
  setInterval(() => {
    void autoUpdater.checkForUpdates().catch(() => {});
  }, 6 * 60 * 60 * 1000);
}
