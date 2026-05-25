"""接针策略（热度+OI 标的池 · 币安纸面）— env 参数。"""

from __future__ import annotations

import os


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


JIEZHEN_SCHEDULER_ENABLED = env_truthy("JIEZHEN_SCHEDULER_ENABLED", default=True)
JIEZHEN_SCAN_INTERVAL_SEC = max(
    15, int(os.getenv("JIEZHEN_SCAN_INTERVAL_SEC", "60") or 60)
)
JIEZHEN_TRAIL_SCHEDULER_ENABLED = env_truthy(
    "JIEZHEN_TRAIL_SCHEDULER_ENABLED", default=True
)

# 接针 trail 开关独立；阈值默认读 MOM_TRAIL_*（两套策略可配同一套止盈参数）
JIEZHEN_TRAIL_ENABLED = env_truthy("JIEZHEN_TRAIL_ENABLED", default=True)


def _resolve_trail_scan_interval_sec() -> int:
    sec_raw = os.getenv("JIEZHEN_TRAIL_SCAN_INTERVAL_SEC", "").strip()
    if sec_raw:
        return max(5, int(sec_raw or 5))
    # 未单独配置时与动量 trail 间隔一致
    from momentum_config import MOM_TRAIL_SCAN_INTERVAL_SEC

    return MOM_TRAIL_SCAN_INTERVAL_SEC


JIEZHEN_TRAIL_SCAN_INTERVAL_SEC = _resolve_trail_scan_interval_sec()

JIEZHEN_ACCOUNT_EQUITY_USDT = max(
    1.0, float(os.getenv("JIEZHEN_ACCOUNT_EQUITY_USDT", "10000") or 10000)
)
JIEZHEN_LEVERAGE = max(0.0, float(os.getenv("JIEZHEN_LEVERAGE", "0.1") or 0.1))
JIEZHEN_NOTIONAL_USDT = JIEZHEN_ACCOUNT_EQUITY_USDT * JIEZHEN_LEVERAGE

# 标的：worth_watch_hot_oi（需先跑 oi 小时任务）
JIEZHEN_UNIVERSE_MAX = max(1, int(os.getenv("JIEZHEN_UNIVERSE_MAX", "20") or 20))
JIEZHEN_MAX_OPEN_PER_SIDE = max(
    1, int(os.getenv("JIEZHEN_MAX_OPEN_PER_SIDE", "5") or 5)
)
JIEZHEN_MAX_OPEN_TOTAL = max(
    1, int(os.getenv("JIEZHEN_MAX_OPEN_TOTAL", "10") or 10)
)

# 接针逻辑（默认 zhen.py：min 振幅/ATR + 0.8% 下限）
JIEZHEN_KLINE_INTERVAL = (os.getenv("JIEZHEN_KLINE_INTERVAL", "1m") or "1m").strip()
JIEZHEN_KLINE_LIMIT = max(60, int(os.getenv("JIEZHEN_KLINE_LIMIT", "241") or 241))
JIEZHEN_EMA_PERIOD = max(0, int(os.getenv("JIEZHEN_EMA_PERIOD", "240") or 240))
JIEZHEN_ATR_PERIOD = max(5, int(os.getenv("JIEZHEN_ATR_PERIOD", "60") or 60))
JIEZHEN_AMPLITUDE_PERIOD = max(5, int(os.getenv("JIEZHEN_AMPLITUDE_PERIOD", "60") or 60))
JIEZHEN_VALUE_MULTIPLIER = max(
    0.1, float(os.getenv("JIEZHEN_VALUE_MULTIPLIER", "2") or 2)
)
JIEZHEN_MIN_DISTANCE_PCT = max(
    0.1, float(os.getenv("JIEZHEN_MIN_DISTANCE_PCT", "0.8") or 0.8)
)
# zhen_2 用平均而非 min；=1 时使用 (amp+atr)/2，无 0.8% 下限
JIEZHEN_DISTANCE_MODE = (
    os.getenv("JIEZHEN_DISTANCE_MODE", "min") or "min"
).strip().lower()
# 最近 N 根 K 的 low/high 触及接针价才纸面开仓（模拟限价成交）
JIEZHEN_TOUCH_LOOKBACK_BARS = max(
    1, int(os.getenv("JIEZHEN_TOUCH_LOOKBACK_BARS", "3") or 3)
)

JIEZHEN_COOLDOWN_SEC = max(0, int(os.getenv("JIEZHEN_COOLDOWN_SEC", "60") or 60))
JIEZHEN_TRAIL_REOPEN_BLOCK = env_truthy("JIEZHEN_TRAIL_REOPEN_BLOCK", default=True)
_trail_reopen_min = os.getenv("JIEZHEN_TRAIL_REOPEN_COOLDOWN_MIN", "").strip()
JIEZHEN_TRAIL_REOPEN_COOLDOWN_MIN = max(1, int(_trail_reopen_min or "15"))
JIEZHEN_TRAIL_REOPEN_COOLDOWN_SEC = JIEZHEN_TRAIL_REOPEN_COOLDOWN_MIN * 60

JIEZHEN_TG_NOTIFY = env_truthy("JIEZHEN_TG_NOTIFY", default=True)
JIEZHEN_VERBOSE_LOG = env_truthy("JIEZHEN_VERBOSE_LOG", default=True)


def jiezhen_scheduler_enabled() -> bool:
    return JIEZHEN_SCHEDULER_ENABLED


def jz_trail_scheduler_enabled() -> bool:
    return (
        JIEZHEN_SCHEDULER_ENABLED
        and JIEZHEN_TRAIL_SCHEDULER_ENABLED
        and JIEZHEN_TRAIL_ENABLED
    )


def jz_trail_config():
    """独立策略 trail：开关 JIEZHEN_TRAIL_ENABLED；阈值与动量同读 MOM_TRAIL_*。"""
    from momentum_config import mom_trail_config
    from momentum_trail import TrailConfig

    base = mom_trail_config()
    return TrailConfig(
        enabled=JIEZHEN_TRAIL_ENABLED,
        stop_loss_pct=base.stop_loss_pct,
        low_trail_stop_loss_pct=base.low_trail_stop_loss_pct,
        trail_stop_loss_pct=base.trail_stop_loss_pct,
        higher_trail_stop_loss_pct=base.higher_trail_stop_loss_pct,
        low_trail_profit_threshold=base.low_trail_profit_threshold,
        first_trail_profit_threshold=base.first_trail_profit_threshold,
        second_trail_profit_threshold=base.second_trail_profit_threshold,
    )
