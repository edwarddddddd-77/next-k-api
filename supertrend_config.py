"""Supertrend lane — env 与调度参数（U 本位永续 · 热度+OI 标的 · 反转平仓）。"""

from __future__ import annotations

import os
from typing import Optional, Tuple

FAPI = "https://fapi.binance.com"

ST_ATR_PERIOD = max(1, int(os.getenv("ST_ATR_PERIOD", "10") or 10))
ST_ATR_MULTIPLIER = float(os.getenv("ST_ATR_MULTIPLIER", "3.5") or 3.5)
ST_SOURCE = (os.getenv("ST_SOURCE", "hl2") or "hl2").strip().lower()
ST_ATR_METHOD = (os.getenv("ST_ATR_METHOD", "wilder") or "wilder").strip().lower()
ST_TIMEFRAME = (os.getenv("ST_TIMEFRAME", "5m") or "5m").strip()
ST_KLINE_LIMIT = max(50, int(os.getenv("ST_KLINE_LIMIT", "300") or 300))

ST_UNIVERSE_MODE = (os.getenv("ST_UNIVERSE_MODE", "hot_oi") or "hot_oi").strip().lower()
ST_MAX_SYMBOLS = max(0, int(os.getenv("ST_MAX_SYMBOLS", "0") or 0))
ST_INTER_SYMBOL_SLEEP_SEC = max(
    0.0, float(os.getenv("ST_INTER_SYMBOL_SLEEP_SEC", "0.15") or 0.15)
)

ST_EXIT_MODE = (
    os.getenv("ST_EXIT_MODE", "reverse_signal,trail_atr,giveback") or "reverse_signal,trail_atr,giveback"
).strip().lower()

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


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# --- 开仓过滤（横盘减磨损；仅挡新开，不挡 reverse_signal 平仓）---
ST_FILTER_ENABLED = env_truthy("ST_FILTER_ENABLED", default=True)

ST_ADX_PERIOD = max(2, int(os.getenv("ST_ADX_PERIOD", "14") or 14))
ST_ADX_MIN = max(0.0, float(os.getenv("ST_ADX_MIN", "25") or 25))

ST_HTF_TIMEFRAME = (os.getenv("ST_HTF_TIMEFRAME", "15m") or "15m").strip()
ST_HTF_REQUIRE_ALIGN = env_truthy("ST_HTF_REQUIRE_ALIGN", default=True)

ST_MIN_ATR_PCT = max(0.0, float(os.getenv("ST_MIN_ATR_PCT", "0.0015") or 0.0015))
ST_RANGE_LOOKBACK = max(1, int(os.getenv("ST_RANGE_LOOKBACK", "24") or 24))
ST_MAX_RANGE_PCT = max(0.0, float(os.getenv("ST_MAX_RANGE_PCT", "0.012") or 0.012))

ST_ENTRY_CONFIRM_BARS = max(0, int(os.getenv("ST_ENTRY_CONFIRM_BARS", "2") or 2))
# 翻转后允许尝试入场的 K 线窗口（含确认 K 延迟；0=仅 flip 当根）
ST_ENTRY_WINDOW_BARS = max(
    0, int(os.getenv("ST_ENTRY_WINDOW_BARS", "6") or 6)
)
ST_MIN_DIST_ATR = max(0.0, float(os.getenv("ST_MIN_DIST_ATR", "0.3") or 0.3))

ST_CHOP_LOOKBACK = max(1, int(os.getenv("ST_CHOP_LOOKBACK", "48") or 48))
ST_CHOP_MAX_FLIPS = max(0, int(os.getenv("ST_CHOP_MAX_FLIPS", "3") or 3))
ST_CHOP_COOLDOWN_BARS = max(0, int(os.getenv("ST_CHOP_COOLDOWN_BARS", "12") or 12))

ST_COOLDOWN_AFTER_LOSS_MIN = max(
    0, int(os.getenv("ST_COOLDOWN_AFTER_LOSS_MIN", "30") or 30)
)
ST_COOLDOWN_AFTER_WIN_MIN = max(0, int(os.getenv("ST_COOLDOWN_AFTER_WIN_MIN", "0") or 0))
ST_MAX_LOSSES_PER_SYMBOL_PER_DAY = max(
    0, int(os.getenv("ST_MAX_LOSSES_PER_SYMBOL_PER_DAY", "2") or 2)
)

# --- 利润保护（仅对已持仓；优先级 giveback → trail_atr → reverse_signal）---
ST_TRAIL_ATR_MULT = max(0.0, float(os.getenv("ST_TRAIL_ATR_MULT", "1.75") or 1.75))
ST_TRAIL_ARM_ATR = max(0.0, float(os.getenv("ST_TRAIL_ARM_ATR", "1.0") or 1.0))
ST_GIVEBACK_PCT = max(0.0, min(1.0, float(os.getenv("ST_GIVEBACK_PCT", "0.35") or 0.35)))
ST_GIVEBACK_MIN_PEAK_PCT = max(
    0.0, float(os.getenv("ST_GIVEBACK_MIN_PEAK_PCT", "0.006") or 0.006)
)
# 峰值浮盈仅用收盘价（不用影线尖刺）；0/false=影线 MFE 也计入峰值
ST_GIVEBACK_PEAK_USE_CLOSE = env_truthy("ST_GIVEBACK_PEAK_USE_CLOSE", default=True)
# 仅当当前仍为浮盈时才 giveback 平仓（避免尖刺峰值后亏损出场）
ST_GIVEBACK_REQUIRE_POSITIVE_PCT = env_truthy(
    "ST_GIVEBACK_REQUIRE_POSITIVE_PCT", default=True
)
ST_DAILY_PROFIT_LOCK_PCT = max(
    0.0, float(os.getenv("ST_DAILY_PROFIT_LOCK_PCT", "0.02") or 0.02)
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


_TIMEFRAME_MS_MAP: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
}


def st_timeframe_ms(tf: Optional[str] = None) -> int:
    key = (tf or ST_TIMEFRAME or "5m").strip()
    return _TIMEFRAME_MS_MAP.get(key, 300_000)


def st_exit_modes_enabled() -> Tuple[str, ...]:
    raw = ST_EXIT_MODE.replace(" ", "")
    if not raw:
        return ("reverse_signal",)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def st_profit_protect_enabled() -> bool:
    modes = set(st_exit_modes_enabled())
    return "trail_atr" in modes or "giveback" in modes
