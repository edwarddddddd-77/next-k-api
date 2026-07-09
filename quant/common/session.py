"""ORB 会话与开盘区间。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from quant.common.config import OrbConfig
from quant.common.tz import normalize_session_tz
from quant.common.us_equity_calendar import (
    is_us_equity_market,
    is_us_equity_trading_day,
    us_equity_session_close_time,
)


def _parse_hhmm(raw: str) -> Optional[Tuple[int, int]]:
    text = (raw or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _localize_tz(tz: str) -> str:
    return normalize_session_tz(tz or "UTC")


def session_anchor_ms(open_ms: int, *, tz: str = "UTC", session_open_time: str = "") -> int:
    """会话起点：无 open_time 时为时区日切 0:00；有则为当日墙钟 session_open_time（如 09:30 ET）。"""
    tz_name = _localize_tz(tz)
    ts = pd.Timestamp(int(open_ms), unit="ms", tz=tz_name)
    parsed = _parse_hhmm(session_open_time)
    if parsed is None:
        return int(ts.normalize().value // 1_000_000)
    hour, minute = parsed
    day_open = ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return int(day_open.value // 1_000_000)


def session_day_floor_ms(open_ms: int, tz: str = "UTC", session_open_time: str = "") -> int:
    """兼容旧名：返回当前 asof 所属会话的起点 ms。"""
    return session_anchor_ms(open_ms, tz=tz, session_open_time=session_open_time)


def session_day_str(open_ms: int, tz: str = "UTC", session_open_time: str = "") -> str:
    tz_name = _localize_tz(tz)
    anchor = session_anchor_ms(open_ms, tz=tz_name, session_open_time=session_open_time)
    return pd.Timestamp(int(anchor), unit="ms", tz=tz_name).strftime("%Y-%m-%d")


def session_close_ms(anchor_ms: int, *, tz: str, session_close_time: str) -> Optional[int]:
    parsed = _parse_hhmm(session_close_time)
    if parsed is None:
        return None
    hour, minute = parsed
    ts = pd.Timestamp(int(anchor_ms), unit="ms", tz=_localize_tz(tz))
    close_ts = ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return int(close_ts.value // 1_000_000)


def is_regular_session(
    asof_open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    session_close_time: str = "",
) -> bool:
    """是否在常规交易时段内（含开盘时刻，不含收盘时刻）。"""
    if not (session_open_time or "").strip():
        return True
    anchor = session_anchor_ms(asof_open_ms, tz=tz, session_open_time=session_open_time)
    if int(asof_open_ms) < anchor:
        return False
    close_ms = session_close_ms(anchor, tz=tz, session_close_time=session_close_time)
    if close_ms is None:
        return True
    return int(asof_open_ms) < close_ms


def effective_session_close_time(
    asof_open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    session_close_time: str,
    market: str,
) -> str:
    """按 market 解析当日收盘墙钟（含 NYSE 提前收盘）。"""
    close = (session_close_time or "").strip()
    if is_us_equity_market(market) and (session_open_time or "").strip():
        day = session_day_str(int(asof_open_ms), tz=tz, session_open_time=session_open_time)
        return us_equity_session_close_time(day, close or "16:00")
    return close


def trading_session_block_reason(
    asof_open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    session_close_time: str = "",
    market: str = "crypto",
) -> Optional[str]:
    """None=在交易时段；否则为 FLAT / idle skip 原因码。"""
    if not (session_open_time or "").strip():
        return None
    if is_us_equity_market(market):
        day = session_day_str(int(asof_open_ms), tz=tz, session_open_time=session_open_time)
        if not is_us_equity_trading_day(day):
            ts = pd.Timestamp(day)
            if int(ts.weekday()) >= 5:
                return "weekend"
            return "exchange_holiday"
        close = us_equity_session_close_time(day, session_close_time or "16:00")
    else:
        close = session_close_time
    if not is_regular_session(
        int(asof_open_ms),
        tz=tz,
        session_open_time=session_open_time,
        session_close_time=close,
    ):
        return "outside_regular_session"
    return None


def is_trading_session(
    asof_open_ms: int,
    *,
    tz: str,
    session_open_time: str,
    session_close_time: str = "",
    market: str = "crypto",
) -> bool:
    """是否在可交易时段（美股含 Mon–Fri、NYSE 假日、RTH/提前收盘）。"""
    return trading_session_block_reason(
        asof_open_ms,
        tz=tz,
        session_open_time=session_open_time,
        session_close_time=session_close_time,
        market=market,
    ) is None


def session_slice(
    df: pd.DataFrame,
    asof_open_ms: int,
    *,
    tz: str = "UTC",
    session_open_time: str = "",
) -> pd.DataFrame:
    if df.empty:
        return df
    anchor = session_anchor_ms(asof_open_ms, tz=tz, session_open_time=session_open_time)
    if (session_open_time or "").strip() and int(asof_open_ms) < anchor:
        return df.iloc[0:0].copy()
    return df[(df["open_time"] >= anchor) & (df["open_time"] <= int(asof_open_ms))].copy()


def compute_opening_range(
    session_df: pd.DataFrame,
    *,
    or_minutes: int,
    bar_step_ms: int,
    asof_open_ms: int,
    tz: str = "UTC",
    session_open_time: str = "",
) -> Optional[Dict[str, float]]:
    if session_df.empty:
        return None
    anchor = session_anchor_ms(asof_open_ms, tz=tz, session_open_time=session_open_time)
    if (session_open_time or "").strip() and int(asof_open_ms) < anchor:
        return None
    or_end = anchor + int(or_minutes) * 60_000 - 1
    or_df = session_df[(session_df["open_time"] >= anchor) & (session_df["open_time"] <= or_end)]
    if or_df.empty:
        return None
    need_bars = max(1, int(round(or_minutes * 60_000 / float(bar_step_ms))))
    if len(or_df) < need_bars:
        return None
    hi = float(or_df["high"].max())
    lo = float(or_df["low"].min())
    if hi <= 0 or lo <= 0 or hi <= lo:
        return None
    ref = float(session_df["close"].iloc[-1])
    return {
        "or_high": hi,
        "or_low": lo,
        "or_mid": (hi + lo) / 2.0,
        "or_width_pct": (hi - lo) / ref * 100.0 if ref > 0 else 0.0,
        "or_end_ms": float(or_end),
        "session_date": session_day_str(anchor, tz=tz, session_open_time=session_open_time),
        "session_anchor_ms": float(anchor),
        "or_avg_vol": float(or_df["volume"].mean()),
    }


def extended_fetch_anchor_ms(open_ms: int, cfg: OrbConfig) -> int:
    """K 线拉取起点：RTH session_open_time（如 09:30 ET）。"""
    return session_anchor_ms(
        int(open_ms),
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
    )
