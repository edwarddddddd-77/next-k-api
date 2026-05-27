"""大户多空 + Taker（公开 fapi/futures/data）配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

VALID_PERIODS: Tuple[str, ...] = (
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
)

VALID_UNIVERSES: Tuple[str, ...] = (
    "trend_5m",
    "watchlist",
    "worth_union",
    "focus",
    "hot_oi",
    "explicit",
    "all",
)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TopTraderParams:
    universe: str
    period: str
    pool_max: int
    explicit_symbols: Tuple[str, ...]
    spacing_sec: float
    jitter_sec: float
    retention_days: int
    min_interval_sec: float


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key, str(default)).strip()
    try:
        return int(raw or default)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key, str(default)).strip()
    try:
        return float(raw or default)
    except ValueError:
        return default


def top_trader_params() -> TopTraderParams:
    universe = os.getenv("TOP_TRADER_UNIVERSE", "trend_5m").strip().lower() or "trend_5m"
    if universe not in VALID_UNIVERSES:
        universe = "trend_5m"

    period = os.getenv("TOP_TRADER_PERIOD", "15m").strip().lower() or "15m"
    if period not in VALID_PERIODS:
        period = "15m"

    explicit_raw = os.getenv("TOP_TRADER_SYMBOLS", "").strip()
    explicit = tuple(
        s.strip().upper()
        for s in explicit_raw.split(",")
        if s and s.strip()
    )

    return TopTraderParams(
        universe=universe,
        period=period,
        pool_max=max(0, _int_env("TOP_TRADER_POOL_MAX", 0)),
        explicit_symbols=explicit,
        spacing_sec=max(0.0, _float_env("TOP_TRADER_SPACING_SEC", 1.0)),
        jitter_sec=max(0.0, _float_env("TOP_TRADER_JITTER_SEC", 0.2)),
        retention_days=max(1, _int_env("TOP_TRADER_RETENTION_DAYS", 7)),
        min_interval_sec=max(0.05, _float_env("TOP_TRADER_API_MIN_INTERVAL_SEC", 0.12)),
    )


def top_trader_scheduler_enabled() -> bool:
    return _env_truthy("TOP_TRADER_SCHEDULER_ENABLED", default=False)


def top_trader_trend_note(period: Optional[str] = None) -> str:
    p = (period or top_trader_params().period or "15m").strip().lower()
    return f"{p} 趋势：PosLSR+Taker+OI/价；无 Smart Money 盈利/均价"


def top_trader_snapshot_path_name() -> str:
    return "top_trader_snapshot.json"
