"""ORB 会话时区：美东须用 IANA 时区以自动处理冬/夏令时。"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 固定偏移或缩写 — 无 DST，美股会话须映射到 America/New_York
_US_EASTERN_ALIASES = frozenset(
    {
        "est",
        "edt",
        "et",
        "eastern",
        "us/eastern",
        "america/new_york",
        "est5edt",
        "eastern time",
        "eastern standard time",
        "eastern daylight time",
        "us eastern",
        "ny",
        "nyc",
    }
)


def normalize_session_tz(raw: str, *, market: str = "crypto") -> str:
    """
    归一化会话时区。

    美股须用 ``America/New_York``（墙钟 09:30/16:00 随 EST/EDT 自动切换）。
    勿用 ``EST``/``EDT`` 固定偏移，否则夏令时期间会与纳斯达克 RTH 差 1 小时。
    """
    text = (raw or "").strip()
    if not text:
        return "America/New_York" if market == "us_equity" else "UTC"
    key = text.lower().replace(" ", "").replace("_", "/")
    if key in _US_EASTERN_ALIASES or key.replace("/", "") in ("useastern",):
        if text not in ("America/New_York", "US/Eastern"):
            logger.info("[orb] session_tz %r -> America/New_York (DST-aware)", text)
        return "America/New_York"
    try:
        pd.Timestamp.now(tz=text)
    except (TypeError, ValueError, Exception):
        fallback = "America/New_York" if market == "us_equity" else "UTC"
        logger.warning("[orb] invalid session_tz %r, fallback %s", text, fallback)
        return fallback
    return text


def session_tz_abbrev(ms: int, tz: str) -> str:
    """返回该时刻当地时区缩写（EST / EDT 等），便于日志与 API。"""
    try:
        ts = pd.Timestamp(int(ms), unit="ms", tz=normalize_session_tz(tz))
        return str(ts.strftime("%Z"))
    except Exception:
        return ""


def session_utc_offset_hours(ms: int, tz: str) -> Optional[float]:
    """该时刻相对 UTC 的小时偏移（冬令时 -5，夏令时 -4）。"""
    try:
        ts = pd.Timestamp(int(ms), unit="ms", tz=normalize_session_tz(tz))
        off = ts.utcoffset()
        if off is None:
            return None
        return round(off.total_seconds() / 3600.0, 2)
    except Exception:
        return None
