"""NYSE 交易日历：周末、法定假日、提前收盘（13:00 ET）。"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import FrozenSet, Set

import pandas as pd

US_EQUITY_MARKETS: FrozenSet[str] = frozenset({"us_equity", "equity", "stock", "stocks"})

# NYSE 全日休市（美东 session_date YYYY-MM-DD）
_BUILTIN_FULL_CLOSE: Set[str] = {
    # 2024
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    # 2025
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    # 2026
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
    # 2027
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-03-26",
    "2027-05-31",
    "2027-06-18",
    "2027-07-05",
    "2027-09-06",
    "2027-11-25",
    "2027-12-24",
}

# NYSE 提前收盘 13:00 ET（感恩节后一日、平安夜、独立日前一日等）
_BUILTIN_EARLY_CLOSE: Set[str] = {
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-07-03",
    "2026-11-27",
    "2026-12-24",
    "2027-11-26",
}

US_EQUITY_EARLY_CLOSE_TIME = "13:00"


def is_us_equity_market(market: str) -> bool:
    return (market or "").strip().lower() in US_EQUITY_MARKETS


@lru_cache(maxsize=1)
def _full_close_dates() -> Set[str]:
    out = set(_BUILTIN_FULL_CLOSE)
    extra = (os.getenv("ORB_US_EQUITY_EXTRA_HOLIDAYS") or "").strip()
    for part in extra.replace(";", ",").split(","):
        d = part.strip()
        if d:
            out.add(d)
    return out


@lru_cache(maxsize=1)
def _early_close_dates() -> Set[str]:
    out = set(_BUILTIN_EARLY_CLOSE)
    extra = (os.getenv("ORB_US_EQUITY_EXTRA_EARLY_CLOSE") or "").strip()
    for part in extra.replace(";", ",").split(","):
        d = part.strip()
        if d:
            out.add(d)
    return out


def normalize_session_date(session_date: str) -> str:
    try:
        return pd.Timestamp(str(session_date).strip()).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(session_date or "").strip()


def is_us_equity_full_holiday(session_date: str) -> bool:
    day = normalize_session_date(session_date)
    return bool(day) and day in _full_close_dates()


def is_us_equity_early_close_day(session_date: str) -> bool:
    day = normalize_session_date(session_date)
    return bool(day) and day in _early_close_dates() and day not in _full_close_dates()


def is_us_equity_trading_day(session_date: str) -> bool:
    """美东 session_date 是否为 NYSE 交易日（Mon–Fri 且非全日休市）。"""
    day = normalize_session_date(session_date)
    if not day:
        return False
    try:
        ts = pd.Timestamp(day)
    except (ValueError, TypeError):
        return False
    if int(ts.weekday()) >= 5:
        return False
    return not is_us_equity_full_holiday(day)


def us_equity_session_close_time(session_date: str, default: str = "16:00") -> str:
    if is_us_equity_early_close_day(session_date):
        return US_EQUITY_EARLY_CLOSE_TIME
    return (default or "16:00").strip() or "16:00"
