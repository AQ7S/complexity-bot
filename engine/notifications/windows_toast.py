"""Windows toast trigger — engine side.

The engine doesn't render the toast directly; it publishes a `notification`
frame on the IPC bus, and Electron's main process owns the OS-level toast
via `new Notification(...)` with the AppUserModelId set per the env. This
keeps notifications working even when the UI is minimised to the tray, but
not running headless (toasts need an interactive desktop session).
"""
from __future__ import annotations

from typing import Literal

from engine.config import settings
from engine.ipc.broadcaster import BUS
from engine.ipc.messages import Notification

EventT = Literal[
    "TRADE_OPENED", "TRADE_CLOSED_PROFIT", "TRADE_CLOSED_LOSS",
    "SIGNAL_DETECTED", "KILL_TRIGGERED", "NEWS_WARNING",
    "ENGINE_ERROR", "TRAINING_COMPLETE",
]

# Event → default WAV (matches engine/sounds/*.wav from Appendix K)
DEFAULT_SOUND: dict[str, str] = {
    "TRADE_OPENED":        "trading_open.wav",
    "TRADE_CLOSED_PROFIT": "profit.wav",
    "TRADE_CLOSED_LOSS":   "loss.wav",
    "SIGNAL_DETECTED":     "signal.wav",
    "KILL_TRIGGERED":      "emergency.wav",
    "NEWS_WARNING":        "news_alert.wav",
    "ENGINE_ERROR":        "error.wav",
    "TRAINING_COMPLETE":   "complete.wav",
}


def notify(event: EventT, *, title: str, body: str, sound: str | None = None) -> bool:
    """Publish a notification frame for the UI to render.

    Returns False if toasts are globally disabled (still no-op publishes
    nothing). Sound selection follows DEFAULT_SOUND if not overridden.
    """
    if not settings.NOTIFY_TOAST_ENABLED:
        return False
    chosen_sound = sound or DEFAULT_SOUND.get(event, "signal.wav")
    if not settings.NOTIFY_SOUND_ENABLED:
        chosen_sound = ""
    BUS.publish("notification", Notification(
        event=event, title=title[:120], body=body[:300], sound=chosen_sound,
    ))
    try:
        from engine.utils.telemetry import record_notification
        record_notification(event)
    except Exception:  # noqa: BLE001 — telemetry must never break notifications
        pass
    return True
