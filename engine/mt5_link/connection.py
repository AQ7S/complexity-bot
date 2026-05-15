"""MT5 terminal initialization + login with retry loop.

Wraps the synchronous `MetaTrader5` package. Async helpers in this codebase
should call these via `asyncio.to_thread` rather than blocking the event loop.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import MetaTrader5 as mt5
from loguru import logger

from engine.config import settings

RETRY_DELAY_S = 5
RETRY_ATTEMPTS = 12


class MT5ConnectionError(RuntimeError):
    pass


def _last_error() -> tuple[int, str]:
    code, desc = mt5.last_error()
    return int(code), str(desc)


def initialize() -> bool:
    """Initialize the terminal + login. Returns True on success."""
    kwargs: dict = {"timeout": settings.MT5_TIMEOUT_MS, "portable": settings.MT5_PORTABLE}
    if settings.MT5_TERMINAL_PATH:
        kwargs["path"] = settings.MT5_TERMINAL_PATH
    if settings.MT5_LOGIN is not None:
        kwargs["login"] = settings.MT5_LOGIN
    if settings.MT5_PASSWORD:
        kwargs["password"] = settings.MT5_PASSWORD
    if settings.MT5_SERVER:
        kwargs["server"] = settings.MT5_SERVER
    ok = mt5.initialize(**kwargs)
    if not ok:
        code, desc = _last_error()
        logger.warning("mt5.initialize failed: ({}) {}", code, desc)
        return False
    return True


def initialize_with_retry(
    *,
    attempts: int = RETRY_ATTEMPTS,
    delay_s: float = RETRY_DELAY_S,
) -> None:
    """Block until MT5 connects or attempts are exhausted."""
    last_err: tuple[int, str] | None = None
    for i in range(1, attempts + 1):
        if initialize():
            info = mt5.terminal_info()
            acct = mt5.account_info()
            logger.info(
                "MT5 connected (attempt {}/{}): build={} login={} server={}",
                i, attempts,
                getattr(info, "build", "?"),
                getattr(acct, "login", "?"),
                getattr(acct, "server", "?"),
            )
            return
        last_err = _last_error()
        if i < attempts:
            time.sleep(delay_s)
    raise MT5ConnectionError(
        f"MT5 init failed after {attempts} attempts; last_error={last_err}"
    )


def shutdown() -> None:
    try:
        mt5.shutdown()
    except Exception as e:  # noqa: BLE001
        logger.warning("mt5.shutdown raised: {}", e)


def is_connected() -> bool:
    info = mt5.terminal_info()
    return bool(info and getattr(info, "connected", False))


@contextmanager
def session(
    *, attempts: int = RETRY_ATTEMPTS, delay_s: float = RETRY_DELAY_S
) -> Iterator[None]:
    initialize_with_retry(attempts=attempts, delay_s=delay_s)
    try:
        yield
    finally:
        shutdown()


def ensure_symbols_visible(symbols: list[str]) -> dict[str, bool]:
    """Mark each symbol as visible in Market Watch. Returns per-symbol success."""
    out: dict[str, bool] = {}
    for s in symbols:
        info = mt5.symbol_info(s)
        if info is None:
            out[s] = False
            logger.warning("symbol {} not found on broker", s)
            continue
        if not info.visible:
            out[s] = bool(mt5.symbol_select(s, True))
        else:
            out[s] = True
    return out
