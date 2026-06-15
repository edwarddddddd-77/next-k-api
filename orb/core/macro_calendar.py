"""宏观事件日过滤（FOMC / CPI）：静态表 + 可选在线刷新。"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import FrozenSet, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_CPI_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_MONTH_NUM = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# 离线兜底（Fed/BLS 拉取失败时使用；CPI 在部分网络会被 BLS 403）
_BUILTIN_FOMC: FrozenSet[str] = frozenset(
    {
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
        "2025-01-29",
        "2025-03-19",
        "2025-05-07",
        "2025-06-18",
        "2025-07-30",
        "2025-09-17",
        "2025-11-05",
        "2025-12-17",
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
    }
)
_BUILTIN_CPI: FrozenSet[str] = frozenset(
    {
        "2024-01-11",
        "2024-02-13",
        "2024-03-12",
        "2024-04-10",
        "2024-05-15",
        "2024-06-12",
        "2024-07-11",
        "2024-08-14",
        "2024-09-11",
        "2024-10-10",
        "2024-11-13",
        "2024-12-11",
        "2025-01-15",
        "2025-02-12",
        "2025-03-12",
        "2025-04-10",
        "2025-05-13",
        "2025-06-11",
        "2025-07-15",
        "2025-08-13",
        "2025-09-10",
        "2025-10-15",
        "2025-11-13",
        "2025-12-10",
        "2026-01-13",
        "2026-02-13",
        "2026-03-11",
        "2026-04-10",
        "2026-05-12",
        "2026-06-10",
        "2026-07-14",
        "2026-08-12",
        "2026-09-11",
        "2026-10-14",
        "2026-11-10",
        "2026-12-10",
    }
)
_BUILTIN_SKIP: FrozenSet[str] = _BUILTIN_FOMC | _BUILTIN_CPI

_FOMC_YEAR_ANCHOR = re.compile(r"(\d{4})\s+FOMC\s+Meetings</a>")
_FOMC_MEETING = re.compile(
    r"fomc-meeting__month[^>]*><strong>(\w+)</strong>.*?fomc-meeting__date[^>]*>([\d\-]+)",
    re.DOTALL,
)
_CPI_RELEASE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),\s+(20\d{2})",
    re.IGNORECASE,
)


def _month_num(token: str) -> Optional[int]:
    key = token.strip().lower().rstrip(".")
    return _MONTH_NUM.get(key) or _MONTH_NUM.get(key[:3])


@dataclass
class _MacroCache:
    dates: Set[str]
    fomc_dates: Set[str]
    cpi_dates: Set[str]
    fetched_at: float
    fomc_ok: bool
    cpi_ok: bool
    fomc_count: int
    cpi_count: int
    builtin_count: int
    env_count: int


_CACHE: Optional[_MacroCache] = None
_LOCK = threading.Lock()
_LOGGED_SKIP_DAYS: Set[str] = set()


def _truthy(raw: str, *, default: bool = False) -> bool:
    v = (raw if raw is not None else "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def _fetch_enabled() -> bool:
    return _truthy(os.getenv("ORB_MACRO_CALENDAR_FETCH", "1"), default=True)


def _ttl_seconds() -> float:
    raw = (os.getenv("ORB_MACRO_CALENDAR_TTL_HOURS") or "24").strip()
    try:
        hours = max(1.0, float(raw))
    except ValueError:
        hours = 24.0
    return hours * 3600.0


def _env_extra_dates() -> Set[str]:
    out: Set[str] = set()
    extra = (os.getenv("ORB_MACRO_SKIP_DATES") or "").strip()
    for part in extra.replace(";", ",").split(","):
        d = part.strip()
        if d:
            out.add(d)
    return out


def _fomc_announcement_iso(year: int, month_name: str, day_token: str) -> Optional[str]:
    month = _MONTH_NUM.get(month_name.strip().lower())
    if not month:
        return None
    token = day_token.replace("*", "").strip()
    try:
        if "-" in token:
            day = int(token.split("-", 1)[1])
        else:
            day = int(token)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def fetch_fomc_skip_dates(*, timeout: float = 20.0) -> Tuple[Set[str], bool]:
    """从 Fed 官网解析 FOMC 决议公布日（会议第 2 日，美东）。"""
    try:
        resp = requests.get(FOMC_CALENDAR_URL, headers=_BROWSER_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[orb] macro FOMC fetch failed: %s", exc)
        return set(), False

    html = resp.text
    anchors = [(m.group(1), m.start()) for m in _FOMC_YEAR_ANCHOR.finditer(html)]
    if not anchors:
        logger.warning("[orb] macro FOMC parse: no year anchors")
        return set(), False

    out: Set[str] = set()
    for idx, (year_s, pos) in enumerate(anchors):
        try:
            year = int(year_s)
        except ValueError:
            continue
        end = anchors[idx + 1][1] if idx + 1 < len(anchors) else pos + 12_000
        block = html[pos:end]
        for month_name, day_token in _FOMC_MEETING.findall(block):
            iso = _fomc_announcement_iso(year, month_name, day_token)
            if iso:
                out.add(iso)
    if out:
        logger.info("[orb] macro FOMC fetched: %s dates", len(out))
    return out, bool(out)


def fetch_cpi_skip_dates(*, timeout: float = 20.0) -> Tuple[Set[str], bool]:
    """从 BLS CPI 发布表解析发布日（08:30 ET）。部分 IP 可能 403，失败则返回空集。"""
    try:
        resp = requests.get(
            BLS_CPI_SCHEDULE_URL,
            headers={**_BROWSER_HEADERS, "Referer": "https://www.bls.gov/"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(
                "[orb] macro CPI fetch HTTP %s (builtin CPI dates used if live parse unavailable)",
                resp.status_code,
            )
            return set(), False
    except Exception as exc:
        logger.warning("[orb] macro CPI fetch failed: %s", exc)
        return set(), False

    chunk = resp.text
    marker = "Schedule of Releases for the Consumer Price Index"
    if marker in chunk:
        chunk = chunk[chunk.find(marker) : chunk.find(marker) + 12_000]

    out: Set[str] = set()
    for month_name, day_s, year_s in _CPI_RELEASE.findall(chunk):
        month = _month_num(month_name)
        if month is None:
            continue
        try:
            out.add(f"{int(year_s):04d}-{month:02d}-{int(day_s):02d}")
        except ValueError:
            continue
    if not out:
        logger.warning("[orb] macro CPI parse: no release dates in page")
        return set(), False
    logger.info("[orb] macro CPI fetched: %s dates", len(out))
    return out, True


def refresh_macro_calendar(*, force: bool = False) -> _MacroCache:
    """拉取 Fed/BLS 并刷新缓存；失败时保留静态表。"""
    global _CACHE
    now = time.time()
    with _LOCK:
        if (
            not force
            and _CACHE is not None
            and (now - _CACHE.fetched_at) < _ttl_seconds()
        ):
            age = now - _CACHE.fetched_at
            logger.debug(
                "[orb] macro calendar cache hit: age=%.0fs total=%s",
                age,
                len(_CACHE.dates),
            )
            return _CACHE

        fomc: Set[str] = set()
        cpi: Set[str] = set()
        fomc_ok = False
        cpi_ok = False
        fetch_on = _fetch_enabled()
        env_extra = _env_extra_dates()

        if fetch_on:
            fomc, fomc_ok = fetch_fomc_skip_dates()
            if _truthy(os.getenv("ORB_MACRO_CALENDAR_FETCH_CPI", "1"), default=True):
                cpi, cpi_ok = fetch_cpi_skip_dates()
            else:
                logger.info("[orb] macro CPI fetch disabled (ORB_MACRO_CALENDAR_FETCH_CPI=0)")
        else:
            logger.info("[orb] macro calendar fetch disabled (ORB_MACRO_CALENDAR_FETCH=0)")

        fomc_dates = set(_BUILTIN_FOMC) | fomc
        cpi_dates = set(_BUILTIN_CPI) | cpi
        dates = fomc_dates | cpi_dates | env_extra
        _CACHE = _MacroCache(
            dates=dates,
            fomc_dates=fomc_dates,
            cpi_dates=cpi_dates,
            fetched_at=now,
            fomc_ok=fomc_ok,
            cpi_ok=cpi_ok,
            fomc_count=len(fomc),
            cpi_count=len(cpi),
            builtin_count=len(_BUILTIN_SKIP),
            env_count=len(env_extra),
        )
        if fetch_on and not fomc_ok and not cpi_ok:
            logger.warning(
                "[orb] macro calendar live fetch failed; using builtin=%s env_extra=%s total=%s",
                len(_BUILTIN_SKIP),
                len(env_extra),
                len(dates),
            )
        elif fetch_on and not cpi_ok:
            logger.warning(
                "[orb] macro CPI live unavailable; builtin+env CPI dates retained (total=%s)",
                len(dates),
            )
        logger.info(
            "[orb] macro calendar refreshed: total=%s builtin=%s env_extra=%s "
            "fomc=%s(%s) cpi=%s(%s)",
            len(dates),
            len(_BUILTIN_SKIP),
            len(env_extra),
            len(fomc),
            "ok" if fomc_ok else "fallback",
            len(cpi),
            "ok" if cpi_ok else "fallback",
        )
        return _CACHE


def get_macro_skip_dates(*, force_refresh: bool = False) -> Set[str]:
    cache = refresh_macro_calendar(force=force_refresh)
    return set(cache.dates)


def macro_calendar_status() -> dict:
    cache = refresh_macro_calendar()
    age_s = max(0.0, time.time() - cache.fetched_at)
    fetch_on = _fetch_enabled()
    return {
        "total_dates": len(cache.dates),
        "builtin_dates": cache.builtin_count,
        "env_extra_dates": cache.env_count,
        "fomc_live": cache.fomc_ok,
        "cpi_live": cache.cpi_ok,
        "fomc_live_count": cache.fomc_count,
        "cpi_live_count": cache.cpi_count,
        "live_fetch_ok": fetch_on and (cache.fomc_ok or cache.cpi_ok),
        "fetched_at": cache.fetched_at,
        "cache_age_seconds": round(age_s, 1),
        "fetch_enabled": fetch_on,
        "ttl_hours": _ttl_seconds() / 3600.0,
    }


def macro_events_for_day(day: str) -> Tuple[str, ...]:
    """返回当日宏观事件类型：'fomc' / 'cpi'（可并存）。"""
    d = str(day or "").strip()
    if not d:
        return ()
    cache = refresh_macro_calendar()
    out = []
    if d in cache.fomc_dates:
        out.append("fomc")
    if d in cache.cpi_dates:
        out.append("cpi")
    return tuple(out)


def is_macro_skip_day(day: str) -> bool:
    d = str(day or "").strip()
    if not d:
        return False
    hit = d in get_macro_skip_dates()
    if hit and d not in _LOGGED_SKIP_DAYS:
        _LOGGED_SKIP_DAYS.add(d)
        logger.info("[orb] macro skip day %s (no new ORB entries)", d)
    return hit


def clear_macro_calendar_cache() -> None:
    """测试用：清空内存缓存。"""
    global _CACHE
    with _LOCK:
        _CACHE = None
    _LOGGED_SKIP_DAYS.clear()
