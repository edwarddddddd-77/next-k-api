"""Supertrend lane — env 与调度参数（U 本位永续 · 热度+OI 标的 · 反转平仓）。"""

from __future__ import annotations

import os
from typing import Tuple

FAPI = "https://fapi.binance.com"

ST_ATR_PERIOD = max(1, int(os.getenv("ST_ATR_PERIOD", "10") or 10))
ST_ATR_MULTIPLIER = float(os.getenv("ST_ATR_MULTIPLIER", "3.0") or 3.0)
ST_SOURCE = (os.getenv("ST_SOURCE", "hl2") or "hl2").strip().lower()
ST_ATR_METHOD = (os.getenv("ST_ATR_METHOD", "wilder") or "wilder").strip().lower()
ST_TIMEFRAME = (os.getenv("ST_TIMEFRAME", "5m") or "5m").strip()
ST_KLINE_LIMIT = max(50, int(os.getenv("ST_KLINE_LIMIT", "300") or 300))

ST_UNIVERSE_MODE = (os.getenv("ST_UNIVERSE_MODE", "hot_oi") or "hot_oi").strip().lower()
ST_MAX_SYMBOLS = max(0, int(os.getenv("ST_MAX_SYMBOLS", "0") or 0))
ST_INTER_SYMBOL_SLEEP_SEC = max(
    0.0, float(os.getenv("ST_INTER_SYMBOL_SLEEP_SEC", "0.15") or 0.15)
)

ST_EXIT_MODE = (os.getenv("ST_EXIT_MODE", "reverse_signal") or "reverse_signal").strip().lower()

# 纸面：ST_MARGIN_USDT = 单笔保证金；盈亏按 ST_NOTIONAL_USDT = 保证金 × 杠杆
_legacy_margin = os.getenv("ST_MARGIN_USDT", "").strip() or os.getenv(
    "ST_VIRTUAL_NOTIONAL_USDT", "100"
)
ST_MARGIN_USDT = max(1.0, float(_legacy_margin or 100))
ST_LEVERAGE = max(1.0, float(os.getenv("ST_LEVERAGE", "10") or 10))
ST_NOTIONAL_USDT = ST_MARGIN_USDT * ST_LEVERAGE
# 兼容旧名（= 保证金，非名义）
ST_VIRTUAL_NOTIONAL_USDT = ST_MARGIN_USDT

ST_MAX_OPEN_POSITIONS = max(0, int(os.getenv("ST_MAX_OPEN_POSITIONS", "8") or 8))
ST_MAX_DAILY_LOSS_PCT = max(
    0.0, float(os.getenv("ST_MAX_DAILY_LOSS_PCT", "0.05") or 0.05)
)
ST_ACCOUNT_EQUITY_USDT = max(
    100.0, float(os.getenv("ST_ACCOUNT_EQUITY_USDT", "10000") or 10000)
)

ST_TG_PUSH_MODE = (os.getenv("ST_TG_PUSH_MODE", "actionable") or "actionable").strip().lower()
ST_TG_NOTIFY_RESOLVE = os.getenv("ST_TG_NOTIFY_RESOLVE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

ST_SCHEDULER_ENABLED = os.getenv("ST_SCHEDULER_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ST_SCAN_CRON_SECOND = max(0, min(59, int(os.getenv("ST_SCAN_CRON_SECOND", "30") or 30)))
ST_RESOLVE_INTERVAL_MINUTES = max(
    0, int(os.getenv("ST_RESOLVE_INTERVAL_MINUTES", "0") or 0)
)

# K 线周期 → 收盘后 cron 分钟列表（Asia/Shanghai 与交易所 UTC 边界一致用 UTC 整点 5m）
_TIMEFRAME_CRON_MINUTES: dict[str, Tuple[int, ...]] = {
    "1m": tuple(range(60)),
    "3m": tuple(range(0, 60, 3)),
    "5m": tuple(range(0, 60, 5)),
    "15m": tuple(range(0, 60, 15)),
    "30m": (0, 30),
    "1h": (0,),
    "2h": (0,),
    "4h": (0,),
}


def st_scan_cron_minutes() -> Tuple[int, ...]:
    return _TIMEFRAME_CRON_MINUTES.get(ST_TIMEFRAME, tuple(range(0, 60, 5)))


def st_exit_modes_enabled() -> Tuple[str, ...]:
    raw = ST_EXIT_MODE.replace(" ", "")
    if not raw:
        return ("reverse_signal",)
    return tuple(x for x in raw.split(",") if x)


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")
