"""观盘台策略开仓信号 — 仅 AVAX F-MR（研究信号）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

LANE_AVAX_F_MR = "avax_f_mr"
VALID_LANES = {LANE_AVAX_F_MR}


def record_strategy_open_signal(
    *,
    lane: str,
    symbol: str,
    side: str,
    entry_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    status: str = "emitted",
    skip_reason: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    bar_ms: Optional[int] = None,
) -> None:
    return None


def list_strategy_signals(*, lane: str, limit: int = 100) -> Dict[str, Any]:
    lane_s = str(lane or "").strip() or LANE_AVAX_F_MR
    if lane_s not in VALID_LANES:
        return {"ok": False, "lane": lane_s, "count": 0, "signals": [], "error": "invalid_lane"}
    lim = max(1, min(int(limit or 100), 500))
    from utils.avax_f_mr_indicator import strategy_signal_feed

    return strategy_signal_feed(mode="gate", limit=lim)
