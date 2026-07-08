"""ORB EOD 强平时刻（含 NYSE 提前收盘）。"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.session import (
    effective_session_close_time,
    is_trading_session,
    session_anchor_ms,
    session_close_ms,
)


def _hm_to_min(hour: int, minute: int) -> int:
    return int(hour) * 60 + int(minute)


def effective_eod_hm(
    *,
    bar_ms: int,
    session_tz: str,
    session_open_time: str,
    session_close_time: str,
    market: str,
    exit_hour: int,
    exit_minute: int,
) -> Tuple[int, int]:
    """当日 EOD 强平墙钟 = min(EXIT, NYSE 实际收市)。"""
    close_str = effective_session_close_time(
        int(bar_ms),
        tz=session_tz,
        session_open_time=session_open_time,
        session_close_time=session_close_time,
        market=market,
    )
    parts = (close_str or "16:00").split(":")
    sess_h, sess_m = int(parts[0]), int(parts[1])
    use_min = min(
        _hm_to_min(int(exit_hour), int(exit_minute)),
        _hm_to_min(sess_h, sess_m),
    )
    return use_min // 60, use_min % 60


def eod_deadline_ms(
    bar_ms: int,
    cfg: OrbConfig,
    *,
    exit_hour: int,
    exit_minute: int,
) -> int | None:
    """当日 EOD 强平截止时刻（epoch ms）。"""
    eh, em = effective_eod_hm(
        bar_ms=int(bar_ms),
        session_tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
        exit_hour=exit_hour,
        exit_minute=exit_minute,
    )
    anchor = session_anchor_ms(
        int(bar_ms),
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
    )
    return session_close_ms(
        anchor,
        tz=cfg.session_tz,
        session_close_time=f"{int(eh):02d}:{int(em):02d}",
    )


def bar_at_or_past_eod(ts: pd.Timestamp, eod_hour: int, eod_minute: int) -> bool:
    return ts.hour > int(eod_hour) or (
        ts.hour == int(eod_hour) and ts.minute >= int(eod_minute)
    )


def bar_is_last_rth_minute(
    bar_ms: int,
    cfg: OrbConfig,
    eod_hour: int,
    eod_minute: int,
) -> bool:
    """最后一根仍在 RTH 内的 1m bar（tick 守卫会拦掉后续 bar）。"""
    if not is_trading_session(
        int(bar_ms),
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
    ):
        return False
    deadline = eod_deadline_ms(
        int(bar_ms),
        cfg,
        exit_hour=int(eod_hour),
        exit_minute=int(eod_minute),
    )
    if deadline is None:
        return False
    return int(bar_ms) + 60_000 >= int(deadline)


def should_eod_flat_bar(
    *,
    bar_ms: int,
    ts: pd.Timestamp,
    cfg: OrbConfig,
    exit_hour: int,
    exit_minute: int,
) -> bool:
    eh, em = effective_eod_hm(
        bar_ms=int(bar_ms),
        session_tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        session_close_time=cfg.session_close_time,
        market=cfg.market,
        exit_hour=exit_hour,
        exit_minute=exit_minute,
    )
    if bar_at_or_past_eod(ts, eh, em):
        return True
    return bar_is_last_rth_minute(int(bar_ms), cfg, eh, em)
