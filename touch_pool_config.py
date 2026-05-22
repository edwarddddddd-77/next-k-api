"""
触轨池每 2h 全量筛选 — 默认配置（代码真源）。

环境变量可覆盖；未设置时与本模块常量一致。
部署参考：`.env.oi.example` 中已取消注释的 ZCT_TOUCH_POOL_* 项。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

# ── 调度（Asia/Shanghai cron）────────────────────────────────────────────
TOUCH_POOL_CRON_HOURS: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
TOUCH_POOL_CRON_MINUTE: int = 7

# ── walk / 筛选硬阈值 ────────────────────────────────────────────────────
TOUCH_POOL_WALK_HOURS: float = 6.0
TOUCH_POOL_MIN_TOTAL_TRADES: int = 10
TOUCH_POOL_MIN_WIN_LOSS_ABS: int = 10
TOUCH_POOL_MIN_TOUCH_TRADES: int = 10  # 与 MIN_WIN_LOSS_ABS 对齐
TOUCH_POOL_MIN_TOUCH_WIN_RATE: float = 0.80
TOUCH_POOL_MAX_WIN_LOSS_ABS: int = 22  # win+loss 上限；0=关闭
TOUCH_POOL_MIN_PF: float = 1.30
TOUCH_POOL_MAX_CONSEC_LOSSES: int = 1
TOUCH_POOL_MIN_T4_WIN_RATE: float = 0.0  # 0 = 关闭 T4 门控
TOUCH_POOL_BUCKET_HOURS: int = 6
TOUCH_POOL_MAX_EXPIRED_RATIO: float = 1.0
TOUCH_POOL_MIN_TOUCH_SHARE: float = 0.0

TOUCH_POOL_SCAN_PHASE: str = "touch_pool_4h_full"
TOUCH_POOL_SYMBOLS_SOURCE: str = "worth_watch_plus_default_22"


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


def touch_pool_walk_hours() -> float:
    return max(1.0, _float_env("ZCT_TOUCH_POOL_WALK_HOURS", TOUCH_POOL_WALK_HOURS))


def touch_pool_walk_days() -> float:
    return touch_pool_walk_hours() / 24.0


def touch_pool_bucket_hours() -> int:
    return max(1, _int_env("ZCT_TOUCH_POOL_BUCKET_HOURS", TOUCH_POOL_BUCKET_HOURS))


def touch_pool_4h_filter_params() -> Dict[str, Any]:
    """每 2h 全量闭环入库参数（cron 见 TOUCH_POOL_CRON_HOURS）。"""
    t4_raw = _float_env("ZCT_TOUCH_POOL_MIN_T4_WIN_RATE", TOUCH_POOL_MIN_T4_WIN_RATE)
    min_wl = max(0, _int_env("ZCT_TOUCH_POOL_MIN_WIN_LOSS_ABS", TOUCH_POOL_MIN_WIN_LOSS_ABS))
    min_touch = max(0, _int_env("ZCT_TOUCH_POOL_MIN_TOUCH_TRADES", TOUCH_POOL_MIN_TOUCH_TRADES))
    if min_touch < min_wl:
        min_touch = min_wl
    return {
        "days": touch_pool_walk_days(),
        "min_touch_trades": min_touch,
        "strict_greater_touch": False,
        "min_touch_win_rate": _float_env(
            "ZCT_TOUCH_POOL_MIN_TOUCH_WIN_RATE", TOUCH_POOL_MIN_TOUCH_WIN_RATE
        ),
        "strict_greater_rate": False,
        "min_total_trades": _int_env(
            "ZCT_TOUCH_POOL_MIN_TOTAL_TRADES", TOUCH_POOL_MIN_TOTAL_TRADES
        ),
        "max_expired_ratio": TOUCH_POOL_MAX_EXPIRED_RATIO,
        "min_win_loss_abs": min_wl,
        "max_win_loss_abs": max(
            0, _int_env("ZCT_TOUCH_POOL_MAX_WIN_LOSS_ABS", TOUCH_POOL_MAX_WIN_LOSS_ABS)
        ),
        "min_touch_share": TOUCH_POOL_MIN_TOUCH_SHARE,
        "min_profit_factor": _float_env("ZCT_TOUCH_POOL_MIN_PF", TOUCH_POOL_MIN_PF),
        "max_consecutive_losses_at_end": _int_env(
            "ZCT_TOUCH_POOL_MAX_CONSEC_LOSSES", TOUCH_POOL_MAX_CONSEC_LOSSES
        ),
        "min_t4_touch_win_rate": float(t4_raw) if t4_raw > 0 else 0.0,
    }


def touch_pool_4h_cron_slots() -> List[Tuple[int, int]]:
    """(hour, minute) 列表。"""
    raw_h = (
        os.getenv("ZCT_TOUCH_POOL_CRON_HOURS") or ",".join(str(h) for h in TOUCH_POOL_CRON_HOURS)
    ).strip()
    minute = max(0, min(59, _int_env("ZCT_TOUCH_POOL_CRON_MINUTE", TOUCH_POOL_CRON_MINUTE)))
    hours: List[int] = []
    for part in raw_h.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            hours.append(max(0, min(23, int(part))))
        except ValueError:
            continue
    if not hours:
        hours = list(TOUCH_POOL_CRON_HOURS)
    return [(h, minute) for h in sorted(set(hours))]


def touch_pool_default_child_env() -> Dict[str, str]:
    """子进程未显式设置时注入的 ZCT_TOUCH_POOL_* 默认值。"""
    return {
        "ZCT_TOUCH_POOL_UNIVERSE": "1",
        "ZCT_TOUCH_POOL_WALK_HOURS": str(TOUCH_POOL_WALK_HOURS),
        "ZCT_TOUCH_POOL_CRON_HOURS": ",".join(str(h) for h in TOUCH_POOL_CRON_HOURS),
        "ZCT_TOUCH_POOL_CRON_MINUTE": str(TOUCH_POOL_CRON_MINUTE),
        "ZCT_TOUCH_POOL_MIN_TOTAL_TRADES": str(TOUCH_POOL_MIN_TOTAL_TRADES),
        "ZCT_TOUCH_POOL_MIN_WIN_LOSS_ABS": str(TOUCH_POOL_MIN_WIN_LOSS_ABS),
        "ZCT_TOUCH_POOL_MIN_TOUCH_TRADES": str(TOUCH_POOL_MIN_TOUCH_TRADES),
        "ZCT_TOUCH_POOL_MIN_TOUCH_WIN_RATE": str(TOUCH_POOL_MIN_TOUCH_WIN_RATE),
        "ZCT_TOUCH_POOL_MAX_WIN_LOSS_ABS": str(TOUCH_POOL_MAX_WIN_LOSS_ABS),
        "ZCT_TOUCH_POOL_MIN_PF": str(TOUCH_POOL_MIN_PF),
        "ZCT_TOUCH_POOL_MAX_CONSEC_LOSSES": str(TOUCH_POOL_MAX_CONSEC_LOSSES),
        "ZCT_TOUCH_POOL_MIN_T4_WIN_RATE": str(TOUCH_POOL_MIN_T4_WIN_RATE),
        "ZCT_TOUCH_POOL_BUCKET_HOURS": str(TOUCH_POOL_BUCKET_HOURS),
    }


def apply_touch_pool_default_env(env: Dict[str, str]) -> Dict[str, str]:
    """仅填充 env 中缺失的触轨池变量。"""
    out = dict(env)
    for key, val in touch_pool_default_child_env().items():
        if not str(out.get(key, "")).strip():
            out[key] = val
    return out
