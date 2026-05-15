import { useEffect, useState } from 'react';
import { sendCommand } from '@/hooks/useEngineSocket';
import { useEngineStore } from '@/store/engineStore';
import { KILL_ZONES } from '@/lib/constants';
import PriceAlertManager from '@/components/PriceAlertManager';

const ENCRYPTED_FIELDS = [
  { key: 'MT5_LOGIN',                label: 'MT5 Login',           type: 'text'     },
  { key: 'MT5_PASSWORD',             label: 'MT5 Password',        type: 'password' },
  { key: 'MT5_SERVER',               label: 'MT5 Server',          type: 'text'     },
  { key: 'ANTHROPIC_API_KEY',        label: 'Anthropic API key',   type: 'password' },
  { key: 'SUPABASE_URL',             label: 'Supabase URL',        type: 'text'     },
  { key: 'SUPABASE_SERVICE_ROLE_KEY',label: 'Supabase service key',type: 'password' },
  { key: 'DISCORD_WEBHOOK_URL',      label: 'Discord webhook',     type: 'text'     },
  { key: 'FOREX_CALENDAR_API_KEY',   label: 'forex-calendar key',  type: 'password' },
  { key: 'FINNHUB_API_KEY',          label: 'Finnhub key',         type: 'password' },
  { key: 'JBLANKED_API_KEY',         label: 'jblanked key',        type: 'password' },
] as const;

const NOTIFY_TOGGLES = [
  { key: 'NOTIFY_TOAST_ENABLED',   label: 'Windows toasts'  },
  { key: 'NOTIFY_SOUND_ENABLED',   label: 'Sound effects'   },
  { key: 'NOTIFY_DISCORD_ENABLED', label: 'Discord webhooks'},
] as const;

export default function Settings() {
  const settingsKv = useEngineStore((s) => s.settingsKv);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [emergencyAck, setEmergencyAck] = useState<string | null>(null);
  const [autostart, setAutostart] = useState(false);

  useEffect(() => {
    void sendCommand('cmd_get_settings', {});
  }, []);

  useEffect(() => { setDraft({ ...settingsKv }); }, [settingsKv]);

  const set = (k: string, v: string) => setDraft((d) => ({ ...d, [k]: v }));

  const save = async () => {
    const partial: Record<string, string> = {};
    for (const k of Object.keys(draft)) {
      if (draft[k] !== settingsKv[k]) partial[k] = draft[k];
    }
    if (Object.keys(partial).length === 0) return;
    const ok = await sendCommand('cmd_settings_update', { partial });
    if (ok) setSavedAt(Date.now());
  };

  const emergencyStop = async () => {
    const ok = await sendCommand('cmd_emergency_close', {});
    setEmergencyAck(ok ? 'Sent. Engine will close all positions.' : 'No engine bridge.');
  };

  return (
    <section data-testid="page-settings" className="space-y-6 p-6">
      <header className="flex items-center justify-between">
        <h1 className="font-hero text-2xl text-accent-cyan">Settings</h1>
        <button
          type="button"
          onClick={() => void emergencyStop()}
          data-testid="emergency-stop"
          className="rounded bg-accent-red px-4 py-2 text-sm font-bold text-white hover:opacity-90"
        >
          EMERGENCY STOP
        </button>
      </header>
      {emergencyAck && <p className="text-xs text-accent-gold">{emergencyAck}</p>}

      <section data-testid="settings-creds" className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-white/70">
          Credentials (encrypted at rest via Fernet)
        </h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {ENCRYPTED_FIELDS.map((f) => (
            <label key={f.key} className="flex flex-col gap-1 text-xs">
              <span className="text-white/60">{f.label}</span>
              <input
                type={f.type}
                value={draft[f.key] ?? ''}
                onChange={(e) => set(f.key, e.target.value)}
                data-testid={`field-${f.key}`}
                className="rounded bg-bg-tertiary px-2 py-1 font-mono text-white outline-none focus:ring-1 focus:ring-accent-cyan"
                placeholder={settingsKv[f.key] ? '••••••••' : 'unset'}
              />
            </label>
          ))}
        </div>
        <button onClick={() => void save()} data-testid="save-settings"
                className="mt-3 rounded bg-accent-cyan/20 px-4 py-1 text-xs text-accent-cyan hover:bg-accent-cyan/30">
          Save
        </button>
        {savedAt && <span className="ml-3 text-xs text-accent-green">
          Saved at {new Date(savedAt).toLocaleTimeString()}
        </span>}
      </section>

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-white/70">Notifications</h2>
        <div className="flex flex-wrap gap-4 text-xs">
          {NOTIFY_TOGGLES.map((t) => (
            <label key={t.key} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={(draft[t.key] ?? 'true') === 'true'}
                onChange={(e) => set(t.key, e.target.checked ? 'true' : 'false')}
                data-testid={`toggle-${t.key}`}
              />
              <span className="text-white/80">{t.label}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">
          Kill Zones (read-only, EST)
        </h2>
        <ul className="grid grid-cols-2 gap-2 text-xs font-mono text-white/70">
          {KILL_ZONES.map((z) => (
            <li key={z.label} className="rounded bg-bg-tertiary px-2 py-1">
              {z.label}: {String(Math.floor(z.start / 60)).padStart(2, '0')}:00–
              {String(Math.floor(z.end / 60)).padStart(2, '0')}:00
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-lg border border-white/5 bg-bg-secondary p-4">
        <h2 className="mb-2 text-sm font-bold uppercase tracking-wider text-white/70">Startup</h2>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox" checked={autostart}
            onChange={(e) => setAutostart(e.target.checked)}
            data-testid="toggle-autostart"
          />
          <span className="text-white/80">
            Start engine with Windows (runs <code>scripts/install_engine.ps1</code>; toggle off
            invokes <code>uninstall_engine.ps1</code>)
          </span>
        </label>
      </section>

      <PriceAlertManager />
    </section>
  );
}
