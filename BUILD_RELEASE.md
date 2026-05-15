# Building the .exe Installer

This produces a single Windows installer that bundles both the Python engine
and the Electron UI. End users just run the installer; no Python required
on the target machine.

## Prerequisites (build host only — not the end user's box)

1. The engine venv already set up:
   `python -m venv engine\.venv` then
   `engine\.venv\Scripts\pip install -r engine\requirements.txt -r engine\requirements-dev.txt`
   (`requirements-dev.txt` now includes `pyinstaller==6.11.1`).
2. UI deps installed: `cd ui; pnpm install`
3. (Optional) Replace `ui\build\icon.ico` with a real 256×256 .ico — electron-builder
   falls back to the Electron stock icon if missing.

## Build

From the repo root:

```powershell
.\scripts\build_release.ps1
```

That runs three stages:
1. **PyInstaller** packs the engine into `engine\dist\engine\engine.exe` (folder
   bundle including all Python deps + torch DLLs + MetaTrader5 + the bundled WAVs
   and any model checkpoints in `engine\models\checkpoints`).
2. **Vite + tsc** builds the renderer (`ui\dist`) and the Electron main / preload
   (`ui\dist-electron`).
3. **electron-builder** packages everything into
   `ui\release\Complexity Engine-Setup-<version>.exe` (NSIS installer).

Useful flags:
- `-SkipEngine` — reuse the existing PyInstaller bundle (saves ~5 minutes
  while iterating on the UI).
- `-SkipUi` — only rebuild the engine bundle.
- `-Clean` — wipe `engine\build`, `engine\dist`, and `ui\release` first.

## What ends up where on the user's machine

After the user runs the installer (default install dir
`%LOCALAPPDATA%\Programs\complexity-engine-ui`):

```
<install>\
  Complexity Engine.exe        # Electron entry — what the user clicks
  resources\
    app.asar                   # Renderer + Electron main + preload
    engine\
      engine.exe               # Bundled Python engine (PyInstaller folder)
      _internal\               # Torch / numpy / MT5 DLLs
      engine\sounds\*.wav      # 8 notification WAVs
      engine\models\checkpoints\*.pt
```

When the user launches the app, the Electron main process spawns
`resources\engine\engine.exe`. The engine binds the WS server on
`127.0.0.1:8765`; the UI renderer connects to it. On quit, `before-quit` kills
the engine subprocess.

## Pre-shipping checklist

1. **Anthropic + MT5 + Supabase creds** are *not* baked into the installer.
   Each end user sets them via Settings on first launch (saved encrypted to
   the per-user SQLite at `%APPDATA%\complexity-engine\journal.sqlite`).
2. **Code signing** is not configured. Without an EV cert the SmartScreen
   warning will show. Either get a cert (~$300/yr) or document the
   "More info → Run anyway" workaround in the README.
3. **Bundle size**: torch + scipy + MetaTrader5 + supabase add up to ~500 MB
   uncompressed, ~250 MB after NSIS LZMA. Expect a 250–300 MB installer.
4. **First-launch slowness**: PyInstaller `onedir` extracts nothing (it's
   already a folder), so engine startup is comparable to the dev experience —
   ~3 s to import torch + ~5 s to connect to MT5.

## Smoke test the installer

After running `build_release.ps1`:
1. Run the installer, accept the default install dir.
2. Launch from the Start Menu.
3. The window should appear within 5 s.
4. Open Settings → fill in MT5 + Anthropic + Discord → Save.
5. Restart the app; verify the values persist (Fernet round-trip working).
6. CommandCenter should show `WS ●` (green) and the `STARTING → LIVE`
   transition once MT5 connects.

## Distributing

The installer at `ui\release\Complexity Engine-Setup-1.0.0.exe` is the only
file you need to ship. Drop it on a download page, network share, or
internal CDN. There is no auto-updater wired up — bump
`ui\package.json` `"version"` and rebuild for each release.
