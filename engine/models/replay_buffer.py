from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAPACITY = 10_000
DEFAULT_RECENT_FRACTION = 0.70


@dataclass(frozen=True)
class TradeExperience:
    symbol: str
    timeframe: str
    timestamp: str
    features: list[float]
    label: int
    pnl: float
    confluence: int
    regime: str


class ExperienceReplay:
    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = int(capacity)
        self._buffer: deque[TradeExperience] = deque(maxlen=self._capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: TradeExperience) -> None:
        self._buffer.append(experience)

    def extend(self, experiences: Iterable[TradeExperience]) -> None:
        for e in experiences:
            self.add(e)

    def sample_mixed(
        self,
        batch_size: int,
        *,
        recent_fraction: float = DEFAULT_RECENT_FRACTION,
        rng: random.Random | None = None,
    ) -> list[TradeExperience]:
        if not self._buffer:
            return []
        rng = rng or random
        recent_n = int(round(batch_size * recent_fraction))
        old_n = batch_size - recent_n
        items = list(self._buffer)
        recent_pool = items[-min(len(items), max(batch_size, recent_n)):]
        recent = rng.sample(recent_pool, min(recent_n, len(recent_pool)))
        if old_n <= 0 or len(items) <= len(recent_pool):
            return recent
        old_pool = items[: -len(recent_pool)]
        if not old_pool:
            return recent
        old = rng.sample(old_pool, min(old_n, len(old_pool)))
        out = recent + old
        rng.shuffle(out)
        return out

    def save(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(e) for e in self._buffer]
        target.write_text(json.dumps(rows, separators=(",", ":")))
        return len(rows)

    def load(self, path: str | Path) -> int:
        source = Path(path)
        if not source.exists():
            return 0
        try:
            rows = json.loads(source.read_text())
        except json.JSONDecodeError:
            return 0
        loaded = 0
        for r in rows:
            try:
                self.add(TradeExperience(**r))
                loaded += 1
            except TypeError:
                continue
        return loaded

    def stats(self) -> dict[str, Any]:
        wins = sum(1 for e in self._buffer if e.pnl > 0)
        losses = sum(1 for e in self._buffer if e.pnl <= 0)
        symbols = {e.symbol for e in self._buffer}
        regimes = {e.regime for e in self._buffer}
        return {
            "count": len(self._buffer),
            "capacity": self._capacity,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self._buffer) if self._buffer else 0.0,
            "symbols_seen": sorted(symbols),
            "regimes_seen": sorted(regimes),
        }
