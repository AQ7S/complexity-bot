import { app, BrowserWindow, Menu, Tray, nativeImage } from 'electron';
import path from 'node:path';

let tray: Tray | null = null;

export function createTray(getWindow: () => BrowserWindow | null): Tray {
  // Use a tiny in-memory PNG so we don't depend on a bundled icon file.
  // 16×16 transparent square — Windows will fall back to its default if not provided.
  const iconPath = path.join(__dirname, '..', 'public', 'tray.png');
  let img = nativeImage.createFromPath(iconPath);
  if (img.isEmpty()) {
    img = nativeImage.createEmpty();
  }
  tray = new Tray(img);
  tray.setToolTip('Complexity Engine');

  const menu = Menu.buildFromTemplate([
    {
      label: 'Show',
      click: () => {
        const w = getWindow();
        if (w) { w.show(); w.focus(); }
      },
    },
    {
      label: 'Hide',
      click: () => {
        const w = getWindow();
        if (w) w.hide();
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Engine UI',
      click: () => {
        (app as any).isQuittingForReal = true;
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(menu);

  tray.on('click', () => {
    const w = getWindow();
    if (!w) return;
    if (w.isVisible()) w.hide(); else { w.show(); w.focus(); }
  });

  return tray;
}

export function destroyTray(): void {
  if (tray) { tray.destroy(); tray = null; }
}
