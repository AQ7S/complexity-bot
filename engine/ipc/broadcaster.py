"""In-process pub/sub for engine events.

Producers (tick streamer, consensus loop, risk manager, retrainer) publish
typed payloads via `publish(type_, model)`. The IPC WebSocket server is the
sole subscriber in production, but the design allows N subscribers (e.g.
notifications module, Supabase sync). Each subscriber owns a bounded
asyncio.Queue; slow consumers drop oldest frames rather than back-pressuring
the engine main loop.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel

from engine.ipc.messages import envelope

QUEUE_MAX = 256


class Broadcaster:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[dict]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=QUEUE_MAX)
        async with self._lock:
            self._subs.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        async with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, type_: str, payload: BaseModel | dict | None = None) -> None:
        """Non-blocking fan-out to all subscribers. Drops oldest on overflow."""
        frame = envelope(type_, payload)
        for q in list(self._subs):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.warning("broadcaster dropped frame for slow subscriber type={}", type_)
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


# Process-wide singleton — engine modules import and use directly.
BUS = Broadcaster()
