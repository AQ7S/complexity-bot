import { useState, useEffect, useRef } from 'react';
import { Bell, X } from 'lucide-react';
import { useEngineStore } from '@/store/engineStore';

const EVENT_TONE: Record<string, string> = {
  TRADE_OPENED: 'text-accent-cyan',
  TRADE_CLOSED_PROFIT: 'text-accent-green',
  TRADE_CLOSED_LOSS: 'text-accent-red',
  SIGNAL_DETECTED: 'text-accent-purple',
  KILL_TRIGGERED: 'text-accent-red',
  NEWS_WARNING: 'text-accent-gold',
  ENGINE_ERROR: 'text-accent-red',
  TRAINING_COMPLETE: 'text-accent-gold',
};

function timeAgo(ms: number): string {
  const d = Date.now() - ms;
  if (d < 60_000) return `${Math.round(d / 1000)}s`;
  if (d < 3600_000) return `${Math.round(d / 60_000)}m`;
  return `${Math.round(d / 3600_000)}h`;
}

export default function NotificationBell({ dropdownClassName }: { dropdownClassName?: string }) {
  const notifications = useEngineStore((s) => s.notifications);
  const unread = useEngineStore((s) => s.notificationsUnread);
  const markRead = useEngineStore((s) => s.markNotificationsRead);
  const clear = useEngineStore((s) => s.clearNotifications);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  const toggle = () => {
    if (!open) markRead();
    setOpen((v) => !v);
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={toggle}
        data-testid="notification-bell"
        className="relative rounded p-2 text-white/70 hover:bg-white/5 hover:text-white"
        aria-label="notifications"
      >
        <Bell size={18} />
        {unread > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-accent-red px-1 text-[9px] font-bold text-white">
            {unread > 9 ? '9+' : unread}
          </span>
        )}
      </button>
      {open && (
        <div className={`absolute z-50 w-80 rounded-lg border border-white/10 bg-bg-secondary shadow-2xl ${dropdownClassName ?? 'right-0 top-10'}`}>
          <div className="flex items-center justify-between border-b border-white/5 px-3 py-2">
            <span className="text-xs font-bold uppercase tracking-wider text-white/70">
              Notifications
            </span>
            <button
              type="button"
              onClick={clear}
              className="text-[10px] uppercase text-white/40 hover:text-white"
            >
              Clear
            </button>
          </div>
          <ul className="max-h-96 overflow-y-auto">
            {notifications.length === 0 ? (
              <li className="px-3 py-6 text-center text-xs text-white/40">No notifications yet.</li>
            ) : notifications.map((n) => (
              <li
                key={n.id}
                className="border-b border-white/5 px-3 py-2 text-xs last:border-b-0"
              >
                <div className="flex items-center justify-between">
                  <span className={`font-bold uppercase tracking-wider ${EVENT_TONE[n.event] ?? 'text-white/70'}`}>
                    {n.event.replace(/_/g, ' ')}
                  </span>
                  <span className="text-[10px] text-white/40">{timeAgo(n.ts)}</span>
                </div>
                <div className="mt-0.5 font-mono text-white/90">{n.title}</div>
                <div className="mt-0.5 text-white/50">{n.body}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

void X;
