/**
 * Spawns the bundled Python engine alongside the Electron app.
 *
 * In `pnpm electron:dev` we don't spawn anything — the developer is
 * expected to run the engine in a separate terminal. In a packaged
 * build the engine binary lives at
 *   <app>/resources/engine/engine.exe
 * (placed there by electron-builder's `extraResources`).
 *
 * Lifecycle: started on app `whenReady`, killed on `before-quit`.
 * stdout/stderr are forwarded to the renderer console for visibility.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { app } from 'electron';

let engineProc: ChildProcess | null = null;

function bundledEnginePath(): string | null {
  if (!app.isPackaged) return null;
  // electron-builder lays extraResources under `process.resourcesPath`.
  const exe = path.join(process.resourcesPath, 'engine', 'engine.exe');
  return fs.existsSync(exe) ? exe : null;
}

export function startEngine(onLog: (line: string) => void): void {
  const exe = bundledEnginePath();
  if (!exe) {
    onLog('[engine] not packaged — expecting external engine on ws://127.0.0.1:8765');
    return;
  }
  const cwd = path.dirname(exe);
  engineProc = spawn(exe, [], { cwd, windowsHide: true });
  onLog(`[engine] spawned pid=${engineProc.pid} from ${exe}`);
  engineProc.stdout?.on('data', (b) => onLog(`[engine] ${b.toString().trimEnd()}`));
  engineProc.stderr?.on('data', (b) => onLog(`[engine!] ${b.toString().trimEnd()}`));
  engineProc.on('exit', (code, sig) => {
    onLog(`[engine] exited code=${code} sig=${sig}`);
    engineProc = null;
  });
}

export function stopEngine(): void {
  if (!engineProc) return;
  try {
    engineProc.kill();    // SIGTERM on POSIX, terminate on Win
  } catch { /* best-effort */ }
  engineProc = null;
}
