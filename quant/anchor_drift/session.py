"""Anchor Drift 会话窗口（非 RTH / 开盘前强平）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant.common.session import session_day_str
from quant.common.session_paper import in_regular_session
from quant.common.us_equity_calendar import is_us_equity_trading_day

if TYPE_CHECKING:
    from quant.common.config import OrbConfig


def _parse_hm(raw: str, default_h: int, default_m: int) -> tuple[int, int]:
    text = (raw or "").strip()
    if not text or ":" not in text:
        return default_h, default_m
    parts = text.split(":", 1)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return default_h, default_m


def in_preopen_flat_window(
    now_ms: int,
    sess: "OrbConfig",
    *,
    flat_minutes: int,
) -> bool:
    """下一 RTH 开盘前 flat_minutes 内强平（仅美股交易日）。"""
    mins = max(1, int(flat_minutes))
    day = session_day_str(
        int(now_ms),
        tz=sess.session_tz,
        session_open_time=sess.session_open_time,
    )
    if not is_us_equity_trading_day(day):
        return False
    open_h, open_m = _parse_hm(sess.session_open_time or "09:30", 9, 30)
    ts = pd.Timestamp(int(now_ms), unit="ms", tz=sess.session_tz)
    open_ts = ts.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
    flat_start = open_ts - pd.Timedelta(minutes=mins)
    return flat_start <= ts < open_ts


def in_drift_trading_window(
    now_ms: int,
    sess: "OrbConfig",
    *,
    flat_minutes: int,
) -> bool:
    """非 RTH 且不在开盘前强平窗口。"""
    if in_regular_session(sess, now_ms=int(now_ms)):
        return False
    if in_preopen_flat_window(int(now_ms), sess, flat_minutes=flat_minutes):
        return False
    return True


def is_weekend_anchor_session(anchor_session: str) -> bool:
    """锚定会话是否为周五 RTH 收盘（Fri 16:00 → Mon 9:25 周末周期）。"""
    text = (anchor_session or "").strip()
    if not text:
        return False
    try:
        return pd.Timestamp(text).weekday() == 4
    except ValueError:
        return False
