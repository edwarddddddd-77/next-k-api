"""往返手续费：开仓 maker/taker 分拆，平仓按 taker。"""

from __future__ import annotations

import os
from typing import Optional

# 限价/post-only 成交 → maker；信号价/市价追单 → 双边 taker
MAKER_ENTRY_MODES = frozenset(
    {
        "fvg_prox",
        "fvg",
        "stoplimit",
        "stoplimit_gap",
        "stoplimit_honest",
        "stoplimit_gap_honest",
        "ob_prox",
    }
)

DEFAULT_FEE_MAKER_BPS = 2.0
DEFAULT_FEE_TAKER_BPS = 4.0


def fee_maker_bps_from_env() -> float:
    raw = (os.getenv("ORB_FEE_MAKER_BPS") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    legacy = (os.getenv("ORB_FEE_BPS_PER_SIDE") or "").strip()
    if legacy:
        try:
            return max(0.0, float(legacy))
        except ValueError:
            pass
    return DEFAULT_FEE_MAKER_BPS


def fee_taker_bps_from_env() -> float:
    raw = (os.getenv("ORB_FEE_TAKER_BPS") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    legacy = (os.getenv("ORB_FEE_BPS_PER_SIDE") or "").strip()
    if legacy:
        try:
            return max(0.0, float(legacy))
        except ValueError:
            pass
    return DEFAULT_FEE_TAKER_BPS


def entry_fee_bps(
    entry_mode: Optional[str],
    *,
    maker_bps: float,
    taker_bps: float,
) -> float:
    mode = (entry_mode or "signal").strip().lower()
    if mode in MAKER_ENTRY_MODES:
        return max(0.0, float(maker_bps))
    return max(0.0, float(taker_bps))


def fee_entry_mode_from_fill(entry_fill: Optional[str]) -> str:
    """Map ORB_ENTRY_FILL → fee entry_mode (maker limit vs taker signal)."""
    mode = (entry_fill or "signal").strip().lower()
    return mode if mode in MAKER_ENTRY_MODES else "signal"


def trade_fee_usdt(
    notional_usdt: float,
    *,
    entry_mode: Optional[str] = "signal",
    maker_bps: Optional[float] = None,
    taker_bps: Optional[float] = None,
    fee_bps_per_side: Optional[float] = None,
) -> float:
    """fee = notional × (open_bps + close_bps) / 10000；close 恒 taker。"""
    n = max(0.0, float(notional_usdt or 0.0))
    if fee_bps_per_side is not None:
        bps = max(0.0, float(fee_bps_per_side))
        return round(n * (bps / 10000.0) * 2.0, 4)
    mk = max(0.0, float(maker_bps if maker_bps is not None else fee_maker_bps_from_env()))
    tk = max(0.0, float(taker_bps if taker_bps is not None else fee_taker_bps_from_env()))
    open_bps = entry_fee_bps(entry_mode, maker_bps=mk, taker_bps=tk)
    close_bps = tk
    return round(n * (open_bps + close_bps) / 10000.0, 4)
