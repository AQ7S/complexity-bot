"""Environment + hardcoded constants loader.

Reads `.env` from the repo root if present (via python-dotenv), exposes typed
getters for the variables the engine needs, and pins the risk constants
specified in §4 of the master plan.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

# --- MT5 ---------------------------------------------------------------------
MT5_LOGIN: int | None = int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None
MT5_PASSWORD: str | None = os.environ.get("MT5_PASSWORD") or None
MT5_SERVER: str = os.environ.get("MT5_SERVER", "XMGlobal-Demo")
MT5_TERMINAL_PATH: str | None = os.environ.get("MT5_TERMINAL_PATH") or None
MT5_TIMEOUT_MS: int = int(os.environ.get("MT5_TIMEOUT_MS", "60000"))
MT5_PORTABLE: bool = os.environ.get("MT5_PORTABLE", "false").lower() == "true"

# --- Storage -----------------------------------------------------------------
DUCKDB_PATH: str = os.environ.get(
    "DUCKDB_PATH", str(REPO_ROOT / "engine" / "data" / "store" / "market.duckdb")
)
SQLITE_PATH: str = os.environ.get(
    "SQLITE_PATH", str(REPO_ROOT / "engine" / "data" / "store" / "journal.sqlite")
)
DATA_RING_BUFFER_TICKS: int = int(os.environ.get("DATA_RING_BUFFER_TICKS", "500000"))

# --- Logging -----------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR: str = os.environ.get("LOG_DIR", str(REPO_ROOT / "engine" / "logs"))

# --- Hardcoded risk constants (§4) ------------------------------------------
RISK_PCT_PER_TRADE = 0.02
INTRADAY_KILL_PCT = 0.03
WEEKLY_KILL_PCT = 0.08
MAX_CONCURRENT_POSITIONS = 5
MAX_CORRELATED_POSITIONS = 2
CORRELATION_THRESHOLD = 0.80
NEWS_PAUSE_MINUTES_BEFORE = 30
ATR_SL_MULTIPLIER = 1.5
TRAIL_ATR_MULTIPLIER = 0.5
PARTIAL_CLOSE_RR = 1.0
PARTIAL_CLOSE_FRACTION = 0.5
SPREAD_WIDENED_MULTIPLIER = 3.0
RETRAIN_EVERY_N_TRADES = 100
RETRAIN_CPU_CEILING_PCT = 80
CONSENSUS_MIN_AGREE = 3
FALLBACK_RISK_PCT = 0.005


def have_mt5_credentials() -> bool:
    return MT5_LOGIN is not None and bool(MT5_PASSWORD) and bool(MT5_SERVER)


# --- Anthropic ---------------------------------------------------------------
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY") or None
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS: int = int(os.environ.get("CLAUDE_MAX_TOKENS", "2048"))
CLAUDE_TIMEOUT_S: int = int(os.environ.get("CLAUDE_TIMEOUT_S", "10"))
CLAUDE_RETRY_MAX: int = int(os.environ.get("CLAUDE_RETRY_MAX", "3"))


def have_anthropic() -> bool:
    return bool(ANTHROPIC_API_KEY) and not ANTHROPIC_API_KEY.startswith("sk-ant-api03-XXXX")


# --- Supabase ----------------------------------------------------------------
SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL") or None
SUPABASE_ANON_KEY: str | None = os.environ.get("SUPABASE_ANON_KEY") or None
SUPABASE_SERVICE_ROLE_KEY: str | None = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or None
SUPABASE_SYNC_INTERVAL_S: int = int(os.environ.get("SUPABASE_SYNC_INTERVAL_S", "15"))


# --- Discord -----------------------------------------------------------------
DISCORD_WEBHOOK_URL: str | None = (
    os.environ.get("DISCORD_WEBHOOK_URL")
    or os.environ.get("DISCORD_WEBHOOK_TRADES")
    or None
)
DISCORD_ERROR_WEBHOOK_URL: str | None = (
    os.environ.get("DISCORD_ERROR_WEBHOOK_URL")
    or os.environ.get("DISCORD_WEBHOOK_ERRORS")
    or None
)


def have_discord() -> bool:
    return bool(DISCORD_WEBHOOK_URL) and DISCORD_WEBHOOK_URL.startswith("http")


# --- Notifications -----------------------------------------------------------
NOTIFY_SOUND_ENABLED: bool = os.environ.get("NOTIFY_SOUND_ENABLED", "true").lower() == "true"
NOTIFY_TOAST_ENABLED: bool = os.environ.get("NOTIFY_TOAST_ENABLED", "true").lower() == "true"
NOTIFY_DISCORD_ENABLED: bool = os.environ.get("NOTIFY_DISCORD_ENABLED", "true").lower() == "true"
APP_USER_MODEL_ID: str = os.environ.get("APP_USER_MODEL_ID", "com.complexity.engine")


# --- IPC ---------------------------------------------------------------------
IPC_HOST: str = os.environ.get("IPC_HOST", "127.0.0.1")
IPC_WS_PORT: int = int(os.environ.get("IPC_WS_PORT", os.environ.get("IPC_PORT", "8765")))
IPC_AUTH_TOKEN: str | None = os.environ.get("IPC_AUTH_TOKEN") or None


def have_supabase() -> bool:
    return (
        bool(SUPABASE_URL) and bool(SUPABASE_SERVICE_ROLE_KEY)
        and not SUPABASE_URL.startswith("https://abcdxyz")
    )
