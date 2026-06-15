"""当日 ORB 会话状态与前端提示（休市 / 宏观日）。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.macro_calendar import macro_events_for_day, macro_calendar_status
from orb.core.session import session_day_str
from orb.core.us_equity_calendar import (
    is_us_equity_early_close_day,
    is_us_equity_full_holiday,
    is_us_equity_market,
    is_us_equity_trading_day,
    normalize_session_date,
    us_equity_session_close_time,
)

_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _weekday_label(day: str) -> str:
    try:
        return _WEEKDAY_ZH[int(pd.Timestamp(day).weekday())]
    except (ValueError, TypeError):
        return ""


def _alert(
    *,
    kind: str,
    severity: str,
    title: str,
    message: str,
) -> Dict[str, str]:
    return {"kind": kind, "severity": severity, "title": title, "message": message}


def build_session_today(cfg: Optional[OrbConfig] = None, *, now_ms: Optional[int] = None) -> Dict[str, Any]:
    """构建当日会话提示，供 API / 看板使用。"""
    c = cfg or OrbConfig.from_env()
    t_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    session_date = session_day_str(t_ms, tz=c.session_tz, session_open_time=c.session_open_time)
    day = normalize_session_date(session_date)
    weekday = _weekday_label(day)

    alerts: List[Dict[str, str]] = []
    is_trading_day = True
    non_trading_reason: Optional[str] = None
    early_close = False
    close_time = (c.session_close_time or "16:00").strip() or "16:00"

    if is_us_equity_market(c.market):
        early_close = is_us_equity_early_close_day(day)
        close_time = us_equity_session_close_time(day, c.session_close_time or "16:00")
        is_trading_day = is_us_equity_trading_day(day)
        if not is_trading_day:
            try:
                wd = int(pd.Timestamp(day).weekday())
            except (ValueError, TypeError):
                wd = -1
            if wd >= 5:
                non_trading_reason = "weekend"
                alerts.append(
                    _alert(
                        kind="weekend",
                        severity="block",
                        title="周末休市",
                        message=f"今日 {day}（{weekday}）为美东周末，NYSE 休市，ORB 不扫描、不开新仓。",
                    )
                )
            elif is_us_equity_full_holiday(day):
                non_trading_reason = "exchange_holiday"
                alerts.append(
                    _alert(
                        kind="exchange_holiday",
                        severity="block",
                        title="法定休市",
                        message=f"今日 {day} 为 NYSE 法定休市日，ORB 不扫描、不开新仓。",
                    )
                )
            else:
                non_trading_reason = "closed"
                alerts.append(
                    _alert(
                        kind="non_trading",
                        severity="block",
                        title="非交易日",
                        message=f"今日 {day} 非 NYSE 交易日，ORB 不扫描、不开新仓。",
                    )
                )
        elif early_close:
            alerts.append(
                _alert(
                    kind="early_close",
                    severity="warn",
                    title="提前收盘",
                    message=f"今日 {day} 美东 {close_time} 提前收盘（常规 16:00），持仓将在提前收盘时 EoD 平仓。",
                )
            )

    macro_events = macro_events_for_day(day)
    macro_filter = bool(c.macro_filter)
    macro_status = macro_calendar_status() if macro_filter else None

    for ev in macro_events:
        if ev == "fomc":
            title = "FOMC 决议日"
            msg = f"今日 {day} 为 FOMC 利率决议公布日（通常 14:00 ET）。"
            if macro_filter:
                msg += " 已开启宏观过滤：今日不新开 ORB 仓位（已有持仓仍正常管理）。"
            else:
                msg += " 宏观过滤未开启（ORB_MACRO_FILTER=0），策略仍会尝试开仓。"
            alerts.append(_alert(kind="fomc", severity="warn" if macro_filter else "info", title=title, message=msg))
        elif ev == "cpi":
            title = "CPI 发布日"
            msg = f"今日 {day} 为美国 CPI 发布日（08:30 ET）。"
            if macro_filter:
                msg += " 已开启宏观过滤：今日不新开 ORB 仓位（已有持仓仍正常管理）。"
            else:
                msg += " 宏观过滤未开启（ORB_MACRO_FILTER=0），策略仍会尝试开仓。"
            alerts.append(_alert(kind="cpi", severity="warn" if macro_filter else "info", title=title, message=msg))

    skip_new_entries = (not is_trading_day) or (macro_filter and bool(macro_events))

    return {
        "ok": True,
        "session_date": day,
        "weekday": weekday,
        "session_tz": c.session_tz,
        "market": c.market,
        "is_trading_day": is_trading_day,
        "non_trading_reason": non_trading_reason,
        "early_close": early_close,
        "session_close_time": close_time,
        "macro_events": list(macro_events),
        "macro_filter_enabled": macro_filter,
        "skip_new_entries": skip_new_entries,
        "alerts": alerts,
        "has_alerts": bool(alerts),
        "macro_calendar": macro_status,
    }
