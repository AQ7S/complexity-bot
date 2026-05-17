from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import requests
from loguru import logger

from engine.config import settings


JBLANKED_BASE_URL = "https://www.jblanked.com/news/api"
JBLANKED_TODAY_PATH = "/forex-factory/calendar/today/"
JBLANKED_WEEK_PATH = "/forex-factory/calendar/week/"
JBLANKED_TIMEOUT_S = 5
JBLANKED_USER_AGENT = "complexity-engine/1.0"
_CREDIT_BACKOFF_MIN = 30 * 60
_credit_blocked_until: float = 0.0


Impact = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


@dataclass(frozen=True)
class JBlankedEvent:
    name: str
    currency: str
    impact: Impact
    forecast: float | None
    previous: float | None
    actual: float | None
    scheduled_at: datetime
    ml_prediction: str | None
    ml_confidence: float | None


def _classify_impact(raw: str | None) -> Impact:
    if not raw:
        return "UNKNOWN"
    r = raw.strip().lower()
    if r.startswith("high") or r == "h" or r == "red":
        return "HIGH"
    if r.startswith("med") or r == "m" or r == "orange":
        return "MEDIUM"
    if r.startswith("low") or r == "l" or r == "yellow":
        return "LOW"
    return "UNKNOWN"


def _to_float(raw) -> float | None:
    if raw in (None, "", "—", "-"):
        return None
    try:
        return float(str(raw).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_dt(raw) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _is_credit_exhausted_body(text: str) -> bool:
    return "requires credits" in (text or "").lower() or "billing" in (text or "").lower()


def _maybe_block_on_credits(r) -> bool:
    global _credit_blocked_until
    import time as _t
    if r.status_code in (401, 403) and _is_credit_exhausted_body(r.text):
        _credit_blocked_until = _t.time() + _CREDIT_BACKOFF_MIN
        logger.warning(
            "jblanked: API credits exhausted — backing off {}min. Refill at "
            "https://www.jblanked.com/api/billing/",
            _CREDIT_BACKOFF_MIN // 60,
        )
        return True
    return False


def _credit_blocked_now() -> bool:
    import time as _t
    return _t.time() < _credit_blocked_until


def fetch_today_events(*, currency: str | None = None) -> list[JBlankedEvent]:
    if not settings.have_jblanked() or _credit_blocked_now():
        return []
    headers = {
        "Authorization": f"Api-Key {settings.JBLANKED_API_KEY}",
        "User-Agent": JBLANKED_USER_AGENT,
        "Accept": "application/json",
    }
    params: dict[str, str] = {}
    if currency:
        params["currency"] = currency
    try:
        r = requests.get(
            f"{JBLANKED_BASE_URL}{JBLANKED_TODAY_PATH}",
            headers=headers, params=params, timeout=JBLANKED_TIMEOUT_S,
        )
    except requests.RequestException as e:
        logger.warning("jblanked fetch failed: {}", e)
        return []
    if _maybe_block_on_credits(r):
        return []
    if r.status_code in (401, 403):
        logger.warning("jblanked auth rejected ({}) — check JBLANKED_API_KEY", r.status_code)
        return []
    if r.status_code != 200:
        logger.warning("jblanked HTTP {}: {}", r.status_code, r.text[:200])
        return []
    try:
        rows = r.json()
    except ValueError as e:
        logger.warning("jblanked JSON decode failed: {}", e)
        return []
    if isinstance(rows, dict):
        rows = rows.get("results") or rows.get("events") or rows.get("data") or []
    out: list[JBlankedEvent] = []
    for row in rows or []:
        try:
            out.append(JBlankedEvent(
                name=str(row.get("name") or row.get("event") or "unknown"),
                currency=str(row.get("currency") or row.get("ccy") or "").upper(),
                impact=_classify_impact(row.get("impact")),
                forecast=_to_float(row.get("forecast")),
                previous=_to_float(row.get("previous")),
                actual=_to_float(row.get("actual")),
                scheduled_at=_to_dt(row.get("date") or row.get("time") or row.get("scheduled_at")),
                ml_prediction=row.get("ml_prediction") or row.get("prediction"),
                ml_confidence=_to_float(row.get("ml_confidence") or row.get("confidence")),
            ))
        except Exception as e:  # noqa: BLE001
            logger.debug("jblanked row parse skipped: {}", e)
    return out


def fetch_week_events(*, currency: str | None = None) -> list[JBlankedEvent]:
    if not settings.have_jblanked() or _credit_blocked_now():
        return []
    headers = {
        "Authorization": f"Api-Key {settings.JBLANKED_API_KEY}",
        "User-Agent": JBLANKED_USER_AGENT,
        "Accept": "application/json",
    }
    params: dict[str, str] = {}
    if currency:
        params["currency"] = currency
    try:
        r = requests.get(
            f"{JBLANKED_BASE_URL}{JBLANKED_WEEK_PATH}",
            headers=headers, params=params, timeout=JBLANKED_TIMEOUT_S,
        )
    except requests.RequestException as e:
        logger.warning("jblanked week fetch failed: {}", e)
        return []
    if _maybe_block_on_credits(r):
        return []
    if r.status_code != 200:
        logger.warning("jblanked week HTTP {}: {}", r.status_code, r.text[:200])
        return []
    try:
        rows = r.json()
    except ValueError:
        return []
    if isinstance(rows, dict):
        rows = rows.get("results") or rows.get("events") or rows.get("data") or []
    out: list[JBlankedEvent] = []
    for row in rows or []:
        try:
            out.append(JBlankedEvent(
                name=str(row.get("name") or row.get("event") or "unknown"),
                currency=str(row.get("currency") or row.get("ccy") or "").upper(),
                impact=_classify_impact(row.get("impact")),
                forecast=_to_float(row.get("forecast")),
                previous=_to_float(row.get("previous")),
                actual=_to_float(row.get("actual")),
                scheduled_at=_to_dt(row.get("date") or row.get("time") or row.get("scheduled_at")),
                ml_prediction=row.get("ml_prediction") or row.get("prediction"),
                ml_confidence=_to_float(row.get("ml_confidence") or row.get("confidence")),
            ))
        except Exception:  # noqa: BLE001
            continue
    return out


def next_high_impact_event(
    *, currency: str | None = None, now: datetime | None = None,
) -> JBlankedEvent | None:
    now = now or datetime.now(timezone.utc)
    events = fetch_today_events(currency=currency)
    upcoming = [e for e in events if e.impact == "HIGH" and e.scheduled_at >= now]
    if not upcoming:
        week = fetch_week_events(currency=currency)
        upcoming = [e for e in week if e.impact == "HIGH" and e.scheduled_at >= now]
    if not upcoming:
        return None
    return min(upcoming, key=lambda e: e.scheduled_at)


def hours_until_high_impact(*, currency: str | None = None) -> float | None:
    event = next_high_impact_event(currency=currency)
    if event is None:
        return None
    return (event.scheduled_at - datetime.now(timezone.utc)).total_seconds() / 3600.0
