"""DQN agent (stable-baselines3) over the TradingEnv.

CLI: `python -m engine.models.rl_agent --train --steps 100000`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from engine.data import duckdb_store
from engine.models.cnn_lstm import CLASSES  # only for vote-name parity
from engine.models.rl_env import (
    ACTION_FLAT, ACTION_LONG, ACTION_SHORT, ACTION_NAMES, EnvConfig, TradingEnv,
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Map our internal action → consensus vote.
ACTION_TO_VOTE = {ACTION_FLAT: "HOLD", ACTION_LONG: "BUY", ACTION_SHORT: "SELL"}


# ------------------------------------------------------------------ data load

def _resample_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
    return df_m1.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def load_bars(symbol: str, days: int, *, db_path: str | None = None) -> pd.DataFrame:
    with duckdb_store.open_store(db_path, read_only=True) as con:
        max_ts = con.execute(
            "SELECT MAX(ts) FROM bars WHERE symbol=? AND timeframe='M1'", [symbol]
        ).fetchone()[0]
        if max_ts is None:
            raise RuntimeError(f"No M1 bars for {symbol}")
        cutoff = max_ts - timedelta(days=days)
        rows = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND timeframe='M1' AND ts >= ?
            ORDER BY ts
            """,
            [symbol, cutoff],
        ).fetchdf()
    return _resample_m5(rows.set_index("ts").sort_index())


# ------------------------------------------------------------------ callbacks


class EpisodeRewardLogger(BaseCallback):
    """Track episode rewards so we can compare first-100 vs last-100 means."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.action_counts_total = np.zeros(3, dtype=np.int64)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_reward" in info:
                self.episode_rewards.append(float(info["episode_reward"]))
                self.action_counts_total += info["action_counts"]
        return True


# ------------------------------------------------------------------ training

@dataclass
class TrainResult:
    checkpoint: Path
    n_episodes: int
    first100_mean: float
    last100_mean: float
    action_distribution: dict[str, float]
    elapsed_s: float


def train(
    *,
    symbol: str = "EURUSD",
    days: int = 365,
    total_steps: int = 100_000,
    episode_len: int = 1000,
    db_path: str | None = None,
    seed: int = 42,
) -> TrainResult:
    bars = load_bars(symbol, days, db_path=db_path)
    logger.info("loaded {} M5 bars: {} → {}", len(bars), bars.index[0], bars.index[-1])
    env = TradingEnv(bars, config=EnvConfig(episode_len=episode_len), seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=seed,
    )
    cb = EpisodeRewardLogger()
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb)
    elapsed = time.time() - t0

    rewards = cb.episode_rewards
    n = len(rewards)
    first100 = float(np.mean(rewards[:100])) if n >= 100 else float(np.mean(rewards or [0.0]))
    last100 = float(np.mean(rewards[-100:])) if n >= 100 else first100

    counts = cb.action_counts_total.astype(np.float64)
    total = max(counts.sum(), 1)
    dist = {ACTION_NAMES[i]: float(counts[i] / total) for i in range(3)}

    version = f"v{int(time.time())}"
    ckpt = CHECKPOINT_DIR / f"rl_dqn_{version}.zip"
    model.save(str(ckpt))
    logger.info(
        "DQN saved → {} | episodes={} first100={:.4f} last100={:.4f} dist={} elapsed={:.1f}s",
        ckpt, n, first100, last100, dist, elapsed,
    )
    return TrainResult(
        checkpoint=ckpt,
        n_episodes=n,
        first100_mean=first100,
        last100_mean=last100,
        action_distribution=dist,
        elapsed_s=elapsed,
    )


# ------------------------------------------------------------------ inference + online


def latest_checkpoint() -> Path | None:
    cands = sorted(CHECKPOINT_DIR.glob("rl_dqn_v*.zip"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def load_agent(checkpoint: Path | None = None) -> DQN:
    ckpt = checkpoint or latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("No DQN checkpoint; train first.")
    return DQN.load(str(ckpt))


def predict_vote(agent: DQN, observation: np.ndarray) -> str:
    """Return BUY|SELL|HOLD for the consensus engine."""
    if observation.ndim == 1:
        observation = observation[None, :]
    action, _ = agent.predict(observation, deterministic=True)
    return ACTION_TO_VOTE[int(action[0])]


def online_update(
    agent: DQN,
    obs: np.ndarray,
    action: int,
    reward: float,
    next_obs: np.ndarray,
    done: bool,
) -> None:
    """Append one transition + take a single gradient step.

    Used by the post-trade learning hook; must complete in <200ms.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
    agent.replay_buffer.add(
        obs, next_obs, np.array([action], dtype=np.int64),
        np.array([reward], dtype=np.float32), np.array([done], dtype=bool), [{}],
    )
    if agent.replay_buffer.size() >= max(agent.batch_size, 32):
        agent.train(gradient_steps=1, batch_size=agent.batch_size)


# ------------------------------------------------------------------ CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--episode-len", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.train:
        print("nothing to do; pass --train")
        return 1
    result = train(
        symbol=args.symbol,
        days=args.days,
        total_steps=args.steps,
        episode_len=args.episode_len,
        seed=args.seed,
    )
    print(json.dumps({
        "checkpoint": str(result.checkpoint),
        "n_episodes": result.n_episodes,
        "first100_mean": result.first100_mean,
        "last100_mean": result.last100_mean,
        "action_distribution": result.action_distribution,
        "elapsed_s": result.elapsed_s,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
