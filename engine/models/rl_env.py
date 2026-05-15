"""Gymnasium trading environment for the DQN agent.

Observation: the per-bar feature vector (50 dims, z-scored at construction).
Actions: 3 discrete — 0=FLAT, 1=LONG, 2=SHORT.
Reward: realised log return on the bar (position × next-bar log-return)
        minus a small transaction cost when the position changes.

The full feature/return arrays are precomputed in __init__ so step() is
constant-time array indexing — critical for fast 100k-step training on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from engine.models.dataset import build_feature_frame

ACTION_FLAT = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_NAMES = ("FLAT", "LONG", "SHORT")
N_ACTIONS = 3

POSITION_FROM_ACTION = np.array([0, 1, -1], dtype=np.int8)


@dataclass(frozen=True)
class EnvConfig:
    episode_len: int = 1000
    transaction_cost_bps: float = 0.5  # 0.5 bp per position flip (round-trip ≈ 1 bp)
    reward_scale: float = 1_000.0


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        *,
        config: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()

        feats = build_feature_frame(bars).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        # Per-feature z-score (frozen at env construction; no leakage during episodes).
        arr = feats.to_numpy(dtype=np.float32, copy=True)
        mu = arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-9, 1.0, sd)
        arr = (arr - mu) / sd
        self._features = np.clip(arr, -10.0, 10.0).astype(np.float32, copy=False)
        self._n_features = self._features.shape[1]

        close = bars["close"].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.zeros_like(close, dtype=np.float64)
            log_ret[1:] = np.log(close[1:] / close[:-1])
        self._next_log_ret = np.roll(log_ret, -1).astype(np.float32, copy=False)
        self._next_log_ret[-1] = 0.0  # no next bar at the tail

        self._max_start = max(0, len(self._features) - self.config.episode_len - 2)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng(seed)
        self._t: int = 0
        self._step_in_episode: int = 0
        self._position: int = 0
        self._cum_reward: float = 0.0
        self._action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    # ------------------------------------------------------------------ Gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = int(self._rng.integers(0, max(self._max_start, 1)))
        self._step_in_episode = 0
        self._position = 0
        self._cum_reward = 0.0
        return self._features[self._t], {}

    def step(self, action: int):
        action = int(action)
        new_position = int(POSITION_FROM_ACTION[action])
        flipped = new_position != self._position
        self._position = new_position
        self._action_counts[action] += 1

        # Reward = position × next-bar log return − tx_cost on flip
        cost = (self.config.transaction_cost_bps / 10_000.0) if flipped else 0.0
        raw_reward = float(self._position) * float(self._next_log_ret[self._t]) - cost
        reward = raw_reward * self.config.reward_scale
        self._cum_reward += reward

        self._t += 1
        self._step_in_episode += 1
        terminated = False
        truncated = self._step_in_episode >= self.config.episode_len or self._t >= len(self._features) - 1
        obs = self._features[self._t] if not truncated else self._features[-1]
        info: dict[str, Any] = {}
        if truncated:
            info["episode_reward"] = self._cum_reward
            info["action_counts"] = self._action_counts.copy()
            self._action_counts[:] = 0
        return obs, float(reward), terminated, truncated, info

    @property
    def n_features(self) -> int:
        return self._n_features
