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


def _user_data_env() -> Path:
    appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    base = Path(appdata) if appdata else Path.home()
    return base / "Complexity Engine" / ".env"


def _exe_sibling_env() -> Path:
    import sys
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else REPO_ROOT
    return base / ".env"


USER_ENV_PATH = _user_data_env()
for candidate in (USER_ENV_PATH, _exe_sibling_env(), REPO_ROOT / ".env"):
    if candidate.is_file():
        load_dotenv(candidate, override=False)

if not USER_ENV_PATH.is_file():
    dev_env = REPO_ROOT / ".env"
    if dev_env.is_file():
        USER_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        USER_ENV_PATH.write_bytes(dev_env.read_bytes())

# --- MT5 ---------------------------------------------------------------------
MT5_LOGIN: int | None = int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None
MT5_PASSWORD: str | None = os.environ.get("MT5_PASSWORD") or None
MT5_SERVER: str = os.environ.get("MT5_SERVER", "XMGlobal-Demo")
MT5_TERMINAL_PATH: str | None = os.environ.get("MT5_TERMINAL_PATH") or None
MT5_TIMEOUT_MS: int = int(os.environ.get("MT5_TIMEOUT_MS", "60000"))
MT5_PORTABLE: bool = os.environ.get("MT5_PORTABLE", "false").lower() == "true"

# --- Storage -----------------------------------------------------------------
_USER_DATA_DIR = USER_ENV_PATH.parent
DUCKDB_PATH: str = os.environ.get(
    "DUCKDB_PATH", str(_USER_DATA_DIR / "store" / "market.duckdb")
)
SQLITE_PATH: str = os.environ.get(
    "SQLITE_PATH", str(_USER_DATA_DIR / "store" / "journal.sqlite")
)
DATA_RING_BUFFER_TICKS: int = int(os.environ.get("DATA_RING_BUFFER_TICKS", "500000"))
Path(DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)

# --- Logging -----------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR: str = os.environ.get("LOG_DIR", str(_USER_DATA_DIR / "logs"))
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

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


# --- Shadow Mode (no real orders) -------------------------------------------
SHADOW_MODE: bool = os.environ.get("SHADOW_MODE", "true").lower() == "true"
SHADOW_HEALTH_TIMEOUT_BARS: int = int(os.environ.get("SHADOW_TIME_EXIT_BARS", "48"))
SHADOW_PROMOTION_MIN_TRADES: int = int(os.environ.get("SHADOW_PROMOTION_MIN_TRADES", "100"))
SHADOW_PROMOTION_WR_FLOOR: float = float(os.environ.get("SHADOW_PROMOTION_WR_FLOOR", "0.50"))
SHADOW_PROMOTION_SHARPE_FACTOR: float = float(os.environ.get("SHADOW_PROMOTION_SHARPE_FACTOR", "1.10"))


def shadow_mode_active() -> bool:
    return SHADOW_MODE


# --- Calibration (ECE) -------------------------------------------------------
ECE_RECOMPUTE_EVERY_N_TRADES: int = int(os.environ.get("ECE_RECOMPUTE_EVERY_N_TRADES", "50"))
ECE_OVERCONFIDENT_THRESHOLD: float = float(os.environ.get("ECE_OVERCONFIDENT_THRESHOLD", "0.20"))


# --- Health endpoint --------------------------------------------------------
IPC_HEALTH_PORT: int = int(os.environ.get("IPC_HEALTH_PORT", "8766"))
EVENT_LOG_PATH: str = os.environ.get(
    "EVENT_LOG_PATH", str(_USER_DATA_DIR / "store" / "events.duckdb")
)


# --- External news APIs -----------------------------------------------------
FOREX_CALENDAR_API_KEY: str | None = os.environ.get("FOREX_CALENDAR_API_KEY") or None
FINNHUB_API_KEY: str | None = os.environ.get("FINNHUB_API_KEY") or None
JBLANKED_API_KEY: str | None = os.environ.get("JBLANKED_API_KEY") or None
FRED_API_KEY: str | None = os.environ.get("FRED_API_KEY") or None


def _key_active(key: str | None) -> bool:
    if not key:
        return False
    if key in ("unset", "none", "null"):
        return False
    if key.lower().startswith(("jb_xxx", "fcp_xxx", "cq_xxx", "cp_xxx", "sk-ant-api03-xxxx", "xxx")):
        return False
    return True


def have_forex_calendar() -> bool: return _key_active(FOREX_CALENDAR_API_KEY)
def have_finnhub() -> bool:        return _key_active(FINNHUB_API_KEY)
def have_jblanked() -> bool:       return _key_active(JBLANKED_API_KEY)
def have_fred() -> bool:           return _key_active(FRED_API_KEY)
