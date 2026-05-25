"""动量多一空一 — topMovers 纸面仓位 · env 参数。"""

from __future__ import annotations

import os


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


MOM_SCHEDULER_ENABLED = env_truthy("MOM_SCHEDULER_ENABLED", default=True)
MOM_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("MOM_SCAN_INTERVAL_MINUTES", "15") or 15)
)

# 移动止盈独立定时（与 topMovers 调仓分离，默认每 20 秒）
MOM_TRAIL_SCHEDULER_ENABLED = env_truthy("MOM_TRAIL_SCHEDULER_ENABLED", default=True)


def _resolve_trail_scan_interval_sec() -> int:
    sec_raw = os.getenv("MOM_TRAIL_SCAN_INTERVAL_SEC", "").strip()
    if sec_raw:
        return max(5, int(sec_raw or 5))
    min_raw = os.getenv("MOM_TRAIL_SCAN_INTERVAL_MINUTES", "").strip()
    if min_raw:
        return max(5, int(min_raw or 1) * 60)
    return 20


MOM_TRAIL_SCAN_INTERVAL_SEC = _resolve_trail_scan_interval_sec()
# 兼容旧 API 字段（向上取整分钟，20s → 1）
MOM_TRAIL_SCAN_INTERVAL_MINUTES = max(
    1, (MOM_TRAIL_SCAN_INTERVAL_SEC + 59) // 60
)


def momentum_scheduler_enabled() -> bool:
    return MOM_SCHEDULER_ENABLED


def mom_trail_scheduler_enabled() -> bool:
    return (
        MOM_SCHEDULER_ENABLED
        and MOM_TRAIL_SCHEDULER_ENABLED
        and MOM_TRAIL_ENABLED
    )

# 纸面名义 = 权益 × 杠杆系数（与原脚本 LEVERAGE 语义一致）
MOM_ACCOUNT_EQUITY_USDT = max(
    1.0, float(os.getenv("MOM_ACCOUNT_EQUITY_USDT", "10000") or 10000)
)
MOM_LEVERAGE = max(0.0, float(os.getenv("MOM_LEVERAGE", "0.1") or 0.1))
MOM_NOTIONAL_USDT = MOM_ACCOUNT_EQUITY_USDT * MOM_LEVERAGE

MOM_LONG_EVENT = (os.getenv("MOM_LONG_EVENT", "PULLBACK") or "PULLBACK").strip()
MOM_SHORT_EVENT = (os.getenv("MOM_SHORT_EVENT", "RALLY") or "RALLY").strip()

_raw_bl = (os.getenv("MOM_BLACKLIST", "XNY") or "XNY").strip()
MOM_BLACKLIST = tuple(
    x.strip().upper() for x in _raw_bl.split(",") if x.strip()
)

MOM_ALLOW_USDC = os.getenv("MOM_ALLOW_USDC", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MOM_COOLDOWN_SEC = max(0, int(os.getenv("MOM_COOLDOWN_SEC", "10") or 10))

# 移动止盈平仓后，同标的同方向在冷却期内不再开仓（默认 15 分钟）
MOM_TRAIL_REOPEN_BLOCK = env_truthy("MOM_TRAIL_REOPEN_BLOCK", default=True)
_trail_reopen_min_raw = os.getenv("MOM_TRAIL_REOPEN_COOLDOWN_MIN", "").strip()
MOM_TRAIL_REOPEN_COOLDOWN_MIN = max(
    1,
    int(_trail_reopen_min_raw or "15"),
)
MOM_TRAIL_REOPEN_COOLDOWN_SEC = MOM_TRAIL_REOPEN_COOLDOWN_MIN * 60

MOM_TG_NOTIFY = os.getenv("MOM_TG_NOTIFY", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 扫描 / 止盈详细日志（默认开；=0 仅保留摘要）
MOM_VERBOSE_LOG = env_truthy("MOM_VERBOSE_LOG", default=True)

# ── 开仓过滤（topMovers 事件 + vp_regime 5m；不含费率）────────────────────────
MOM_FILTER_ENABLED = env_truthy("MOM_FILTER_ENABLED", default=True)
# priceChange 幅度门槛（pullback_small / rally_small）；=0 关闭，仍可做事件年龄等过滤
MOM_PRICE_CHANGE_FILTER = env_truthy("MOM_PRICE_CHANGE_FILTER", default=True)
MOM_EVENT_AGE_FILTER = env_truthy("MOM_EVENT_AGE_FILTER", default=False)
MOM_MIN_PULLBACK_PCT = max(0.0, float(os.getenv("MOM_MIN_PULLBACK_PCT", "0.03") or 0.03))
MOM_MIN_RALLY_PCT = max(0.0, float(os.getenv("MOM_MIN_RALLY_PCT", "0.03") or 0.03))
MOM_MAX_EVENT_AGE_MIN = max(1, int(os.getenv("MOM_MAX_EVENT_AGE_MIN", "120") or 120))

MOM_VP_FILTER = env_truthy("MOM_VP_FILTER", default=False)
MOM_VP_INTERVAL = (os.getenv("MOM_VP_INTERVAL", "5m") or "5m").strip().lower()
MOM_VP_STRICT = env_truthy("MOM_VP_STRICT", default=True)


def _parse_scheme_set(name: str, default: str) -> frozenset[str]:
    raw = (os.getenv(name, default) or default).strip()
    return frozenset(x.strip().upper() for x in raw.split(",") if x.strip())


MOM_LONG_VP_SCHEMES = _parse_scheme_set(
    "MOM_LONG_VP_SCHEMES", "MOMENTUM,MEAN_REVERT,WATCH"
)
MOM_SHORT_VP_SCHEMES = _parse_scheme_set(
    "MOM_SHORT_VP_SCHEMES", "MEAN_REVERT,REVERSAL_WATCH,WATCH"
)


def mom_filter_enabled() -> bool:
    return MOM_FILTER_ENABLED


# ── 分档移动止盈（buou_trail 纸面版，默认开；MOM_TRAIL_ENABLED=0 关闭）────
MOM_TRAIL_ENABLED = env_truthy("MOM_TRAIL_ENABLED", default=True)
MOM_TRAIL_STOP_LOSS_PCT = max(0.0, float(os.getenv("MOM_TRAIL_STOP_LOSS_PCT", "3.0") or 3.0))
MOM_TRAIL_LOW_STOP_PCT = max(0.0, float(os.getenv("MOM_TRAIL_LOW_STOP_PCT", "0.3") or 0.3))
MOM_TRAIL_TIER1_DRAWBACK = min(1.0, max(0.0, float(os.getenv("MOM_TRAIL_TIER1_DRAWBACK", "0.3") or 0.3)))
MOM_TRAIL_TIER2_DRAWBACK = min(1.0, max(0.0, float(os.getenv("MOM_TRAIL_TIER2_DRAWBACK", "0.25") or 0.25)))
# 低档阈值与一档同为 2% 时，peak<2% 不进低档保护，避免强标的微利被洗出
MOM_TRAIL_LOW_THRESHOLD_PCT = max(0.0, float(os.getenv("MOM_TRAIL_LOW_THRESHOLD_PCT", "2.0") or 2.0))
MOM_TRAIL_TIER1_THRESHOLD_PCT = max(0.0, float(os.getenv("MOM_TRAIL_TIER1_THRESHOLD_PCT", "2.0") or 2.0))
MOM_TRAIL_TIER2_THRESHOLD_PCT = max(0.0, float(os.getenv("MOM_TRAIL_TIER2_THRESHOLD_PCT", "4.0") or 4.0))


def mom_trail_config():
    from momentum_trail import TrailConfig

    return TrailConfig(
        enabled=MOM_TRAIL_ENABLED,
        stop_loss_pct=MOM_TRAIL_STOP_LOSS_PCT,
        low_trail_stop_loss_pct=MOM_TRAIL_LOW_STOP_PCT,
        trail_stop_loss_pct=MOM_TRAIL_TIER1_DRAWBACK,
        higher_trail_stop_loss_pct=MOM_TRAIL_TIER2_DRAWBACK,
        low_trail_profit_threshold=MOM_TRAIL_LOW_THRESHOLD_PCT,
        first_trail_profit_threshold=MOM_TRAIL_TIER1_THRESHOLD_PCT,
        second_trail_profit_threshold=MOM_TRAIL_TIER2_THRESHOLD_PCT,
    )
