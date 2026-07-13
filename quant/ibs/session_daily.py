"""将 K 线聚合/拉取为 IBS 用 session 日线。"""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd

from quant.common.config import OrbConfig
from quant.common.session import is_regular_session, session_day_str
from quant.common.us_equity_calendar import is_us_equity_trading_day
from quant.ibs.core import SessionDailyBar
from quant.market import fetch_klines_forward, klines_to_df


def aggregate_session_daily(df: pd.DataFrame, *, sess: OrbConfig) -> List[SessionDailyBar]:
    if df.empty:
        return []
    tz = sess.session_tz
    open_time = sess.session_open_time
    close_time = sess.session_close_time
    buckets: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        ms = int(row.open_time)
        if not is_regular_session(
            ms,
            tz=tz,
            session_open_time=open_time,
            session_close_time=close_time,
        ):
            continue
        day = session_day_str(ms, tz=tz, session_open_time=open_time)
        if not is_us_equity_trading_day(day):
            continue
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        b = buckets.get(day)
        if b is None:
            buckets[day] = {
                "session_day": day,
                "open_ms": ms,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
            continue
        b["high"] = max(float(b["high"]), h)
        b["low"] = min(float(b["low"]), l)
        b["close"] = c
    days = sorted(buckets.keys())
    return [
        SessionDailyBar(
            session_day=buckets[d]["session_day"],
            open_ms=int(buckets[d]["open_ms"]),
            open=float(buckets[d]["open"]),
            high=float(buckets[d]["high"]),
            low=float(buckets[d]["low"]),
            close=float(buckets[d]["close"]),
        )
        for d in days
    ]


def _df_to_session_daily(df: pd.DataFrame, *, sess: OrbConfig) -> List[SessionDailyBar]:
    if df.empty:
        return []
    tz = sess.session_tz
    open_time = sess.session_open_time
    out: List[SessionDailyBar] = []
    for row in df.itertuples(index=False):
        ms = int(row.open_time)
        day = session_day_str(ms, tz=tz, session_open_time=open_time)
        if not is_us_equity_trading_day(day):
            continue
        out.append(
            SessionDailyBar(
                session_day=day,
                open_ms=ms,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
            )
        )
    out.sort(key=lambda b: b.session_day)
    return out


def _merge_session_today(
    history: List[SessionDailyBar],
    today_bars: List[SessionDailyBar],
) -> List[SessionDailyBar]:
    if not today_bars:
        return list(history)
    merged = list(history)
    for bar in today_bars:
        if merged and merged[-1].session_day == bar.session_day:
            merged[-1] = bar
        elif not merged or merged[-1].session_day < bar.session_day:
            merged.append(bar)
    return merged


def fetch_exchange_daily_bars(
    symbol: str,
    *,
    days: int,
    exchange_id: str,
    sess: OrbConfig | None = None,
) -> List[SessionDailyBar]:
    sess = sess or OrbConfig.from_env()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(30, int(days)) * 86_400_000
    rows = fetch_klines_forward(symbol, "1d", start_ms, end_ms, exchange_id=exchange_id)
    df = klines_to_df(rows)
    return _df_to_session_daily(df, sess=sess)


def fetch_session_daily_bars(
    symbol: str,
    *,
    days: int,
    exchange_id: str,
    sess: OrbConfig | None = None,
    interval: str = "5m",
) -> List[SessionDailyBar]:
    sess = sess or OrbConfig.from_env()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(30, int(days)) * 86_400_000
    rows = fetch_klines_forward(symbol, interval, start_ms, end_ms, exchange_id=exchange_id)
    df = klines_to_df(rows)
    return aggregate_session_daily(df, sess=sess)


def fetch_daily_bars(
    symbol: str,
    *,
    days: int,
    exchange_id: str,
    sess: OrbConfig | None = None,
    source: str = "session_5m",
    intraday_df: Optional[pd.DataFrame] = None,
    interval: str = "5m",
) -> List[SessionDailyBar]:
    """IBS 日线 — 默认 5m 聚合美东 RTH（对齐 SPY/QQQ 正盘日线，勿用 Bitget 1D）。"""
    sess = sess or OrbConfig.from_env()
    mode = (source or "session_5m").strip().lower()
    if mode in ("session_5m", "session", "5m"):
        history = fetch_session_daily_bars(
            symbol,
            days=days,
            exchange_id=exchange_id,
            sess=sess,
            interval=interval,
        )
        if intraday_df is not None and not intraday_df.empty:
            today = aggregate_session_daily(intraday_df, sess=sess)
            return _merge_session_today(history, today)
        return history
    history = fetch_exchange_daily_bars(symbol, days=days, exchange_id=exchange_id, sess=sess)
    if intraday_df is not None and not intraday_df.empty:
        today = aggregate_session_daily(intraday_df, sess=sess)
        return _merge_session_today(history, today)
    return history
