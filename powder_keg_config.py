"""
火药桶宏观雷达 — 默认阈值与环境变量（币安 OI + 费率 + 短周期横盘）。
"""

from __future__ import annotations

import os
from typing import Tuple

# 调度：上海时间每 15 分钟（:02/:17/:32/:47，错开整点其它任务）
POWDER_KEG_CRON_MINUTES: Tuple[int, ...] = (2, 17, 32, 47)

POWDER_KEG_TOP_N: int = 5
# 扫描宇宙：watchlist = 收筹池（accumulation.db / watchlist 表）
POWDER_KEG_UNIVERSE: str = "watchlist"
POWDER_KEG_MIN_VOL_24H_USD: float = 3_000_000.0
POWDER_KEG_MIN_OI_USD: float = 2_000_000.0

# OI 增仓：1h≥阈值 或 6h≥阈值（仅正向变化，不看减仓）
POWDER_KEG_MIN_OI_DELTA_1H_PCT: float = 2.5
POWDER_KEG_MIN_OI_DELTA_6H_PCT: float = 5.0

# 费率极端：|fundingRate|×100（实战级 0.025；正负方向见 allowed_side）
POWDER_KEG_MIN_FR_ABS_PCT: float = 0.025

# 横盘：24h 涨跌幅 + 近 6 根 1h K 振幅（实战级）
POWDER_KEG_MAX_PX_CHG_24H_PCT: float = 10.0
POWDER_KEG_MAX_RANGE_6H_PCT: float = 8.5

# 预筛后深度扫描（OI+K线）上限；收筹池通常远小于该值
POWDER_KEG_OI_SCAN_MAX_SYMBOLS: int = 80
# 逐请求最小间隔（秒），降低 429 / 权重突发
POWDER_KEG_API_MIN_INTERVAL_SEC: float = 0.12
# 每扫完一个标的后的额外间隔（深度阶段）
POWDER_KEG_SLEEP_PER_SYMBOL_SEC: float = 0.10
# 收筹池规模 ≤ 该值时用「按 symbol 询价」(约 2×N weight)；更大则 bulk 一次再过滤 (约 50 weight)
POWDER_KEG_BULK_TICKER_THRESHOLD: int = 40
# OI 历史周期：1h×6 仅需 1 次 openInterestHist（weight 1）；15m 为可选更细粒度
POWDER_KEG_OI_PERIOD: str = "1h"

# 各轮 Top N 快照保留时长（滚动窗口，默认 24h 后自动清理）
POWDER_KEG_RETENTION_HOURS: int = 24


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def env_truthy(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def powder_keg_radar_enabled() -> bool:
    return env_truthy("POWDER_KEG_RADAR_ENABLED", default=True)


def zct_powder_keg_universe_enabled() -> bool:
    """ZCT 实盘扫描：标的来自 powder_keg_watchlist。"""
    return env_truthy("ZCT_POWDER_KEG_UNIVERSE", default=False)


def apply_powder_keg_scan_env(env: dict) -> dict:
    """
    实盘 zct_vwap_signal_scanner 子进程环境。
    默认开启火药桶 universe，并关闭触轨池 universe（触轨 4h job 仍用 touch_pool env）。
    """
    out = dict(env)
    if not str(out.get("ZCT_POWDER_KEG_UNIVERSE", "")).strip():
        out["ZCT_POWDER_KEG_UNIVERSE"] = "1"
    if zct_powder_keg_universe_enabled():
        out["ZCT_TOUCH_POOL_UNIVERSE"] = "0"
    return out


def powder_keg_params() -> dict:
    return {
        "top_n": max(1, min(20, _int_env("POWDER_KEG_TOP_N", POWDER_KEG_TOP_N))),
        "min_vol_24h_usd": _float_env("POWDER_KEG_MIN_VOL_24H_USD", POWDER_KEG_MIN_VOL_24H_USD),
        "min_oi_usd": _float_env("POWDER_KEG_MIN_OI_USD", POWDER_KEG_MIN_OI_USD),
        "min_oi_delta_1h_pct": _float_env(
            "POWDER_KEG_MIN_OI_DELTA_1H_PCT", POWDER_KEG_MIN_OI_DELTA_1H_PCT
        ),
        "min_oi_delta_6h_pct": _float_env(
            "POWDER_KEG_MIN_OI_DELTA_6H_PCT", POWDER_KEG_MIN_OI_DELTA_6H_PCT
        ),
        "min_fr_abs_pct": _float_env("POWDER_KEG_MIN_FR_ABS_PCT", POWDER_KEG_MIN_FR_ABS_PCT),
        "max_px_chg_24h_pct": _float_env(
            "POWDER_KEG_MAX_PX_CHG_24H_PCT", POWDER_KEG_MAX_PX_CHG_24H_PCT
        ),
        "max_range_6h_pct": _float_env(
            "POWDER_KEG_MAX_RANGE_6H_PCT", POWDER_KEG_MAX_RANGE_6H_PCT
        ),
        "oi_scan_max_symbols": _int_env(
            "POWDER_KEG_OI_SCAN_MAX_SYMBOLS", POWDER_KEG_OI_SCAN_MAX_SYMBOLS
        ),
        "api_min_interval_sec": _float_env(
            "POWDER_KEG_API_MIN_INTERVAL_SEC", POWDER_KEG_API_MIN_INTERVAL_SEC
        ),
        "sleep_per_symbol_sec": _float_env(
            "POWDER_KEG_SLEEP_PER_SYMBOL_SEC", POWDER_KEG_SLEEP_PER_SYMBOL_SEC
        ),
        "bulk_ticker_threshold": _int_env(
            "POWDER_KEG_BULK_TICKER_THRESHOLD", POWDER_KEG_BULK_TICKER_THRESHOLD
        ),
        "oi_period": (
            os.getenv("POWDER_KEG_OI_PERIOD", POWDER_KEG_OI_PERIOD).strip().lower()
            or "1h"
        ),
        "retention_hours": max(
            1, _int_env("POWDER_KEG_RETENTION_HOURS", POWDER_KEG_RETENTION_HOURS)
        ),
        "universe": (
            os.getenv("POWDER_KEG_UNIVERSE", POWDER_KEG_UNIVERSE).strip().lower()
            or "watchlist"
        ),
    }
