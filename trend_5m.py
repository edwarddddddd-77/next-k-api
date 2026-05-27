#!/usr/bin/env python3
"""5m 趋势单：大户 PosLSR + Taker + OI/价格上下文 联合判定。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 5m 趋势标的池：hot_oi ∪ focus ∪ chase_fire（不含埋伏/横盘收筹）
TREND_5M_SOURCE_TABLES: Dict[str, str] = {
    "worth_watch_hot_oi": "hot_oi",
    "focus_watch": "focus",
    "worth_watch_chase_fire": "chase_fire",
}

# Pos / Taker 阈值（与前端展示、文档一致）
POS_LSR_LONG_MIN = 1.15
POS_LSR_SHORT_MAX = 0.85
TAKER_LONG_MIN = 1.05
TAKER_SHORT_MAX = 0.95
ACCOUNT_POSITION_DIVERGE_PP = 12.0
OI_DRAIN_VETO_PX_UP = -5.0  # px>0 且 d6h 低于此 → 否决做多（对齐 focus_watch）


def _float(v: Any) -> Optional[float]:
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def load_oi_coin_context() -> Dict[str, Dict[str, Any]]:
    """从 oi_radar_snapshot.json 读取各 symbol 的 d6h / px_chg 等。"""
    db_dir = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent)))
    path = db_dir / "oi_radar_snapshot.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    coin_data = raw.get("coin_data") if isinstance(raw, dict) else None
    if not isinstance(coin_data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym, row in coin_data.items():
        if not isinstance(row, dict):
            continue
        u = str(sym).strip().upper()
        if not u:
            continue
        out[u] = {
            "d6h": _float(row.get("d6h")),
            "px_chg": _float(row.get("px_chg")),
            "fr_pct": _float(row.get("fr_pct")),
            "heat": _float(row.get("heat")),
        }
    return out


def _has_divergence(snap: Dict[str, Any]) -> bool:
    acc_long = float(snap.get("top_account_long_pct") or 0)
    pos_long = float(snap.get("top_position_long_pct") or 0)
    return (
        acc_long - pos_long >= ACCOUNT_POSITION_DIVERGE_PP
        or pos_long - acc_long >= ACCOUNT_POSITION_DIVERGE_PP
    )


def assess_trend_5m(
    snap: Dict[str, Any],
    oi_ctx: Optional[Dict[str, Any]] = None,
    *,
    pool_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    返回 trend_verdict: long | short | avoid | neutral
    及 trend_score (0–100)、trend_reasons、pool_sources。
    """
    pos_lsr = float(snap.get("top_position_lsr") or 0)
    taker = float(snap.get("taker_bsr") or 0)
    tags = list(snap.get("signal_tags") or [])
    oi = oi_ctx or {}

    d6h = _float(oi.get("d6h"))
    px_chg = _float(oi.get("px_chg"))

    avoid: List[str] = []
    if _has_divergence(snap):
        avoid.append("账户/持仓背离")
    if px_chg is not None and d6h is not None and px_chg > 0 and d6h < OI_DRAIN_VETO_PX_UP:
        avoid.append("价涨OI流出")

    long_ok = (
        pos_lsr >= POS_LSR_LONG_MIN
        and taker >= TAKER_LONG_MIN
        and "账户偏多/持仓偏空" not in tags
    )
    short_ok = (
        pos_lsr > 0
        and pos_lsr <= POS_LSR_SHORT_MAX
        and taker > 0
        and taker <= TAKER_SHORT_MAX
        and "账户偏空/持仓偏多" not in tags
    )

    long_reasons: List[str] = []
    short_reasons: List[str] = []

    if long_ok:
        long_reasons.append(f"PosLSR {pos_lsr:.2f} · Taker {taker:.2f}")
        if "大户+Taker同向多" in tags:
            long_reasons.append("大户+Taker同向多")
        if d6h is not None and d6h >= 0:
            long_reasons.append(f"OI 6h {d6h:+.1f}%")
        if px_chg is not None:
            long_reasons.append(f"24h {px_chg:+.1f}%")

    if short_ok:
        short_reasons.append(f"PosLSR {pos_lsr:.2f} · Taker {taker:.2f}")
        if "大户+Taker同向空" in tags:
            short_reasons.append("大户+Taker同向空")
        if d6h is not None:
            short_reasons.append(f"OI 6h {d6h:+.1f}%")
        if px_chg is not None:
            short_reasons.append(f"24h {px_chg:+.1f}%")

    score = 0
    if long_ok and not avoid:
        score = 40
        if "大户+Taker同向多" in tags:
            score += 25
        if d6h is not None and d6h >= 3:
            score += 15
        if px_chg is not None and px_chg > 0:
            score += 10
        if pool_sources and "focus" in pool_sources:
            score += 10
        score = min(100, score)

    short_score = 0
    if short_ok and not avoid:
        short_score = 40
        if "大户+Taker同向空" in tags:
            short_score += 25
        if d6h is not None and d6h <= -3:
            short_score += 15
        if px_chg is not None and px_chg < 0:
            short_score += 10
        if pool_sources and "focus" in pool_sources:
            short_score += 10
        short_score = min(100, short_score)

    if avoid:
        verdict = "avoid"
        final_reasons = avoid + (long_reasons[:1] if long_reasons else short_reasons[:1])
        final_score = 0
    elif long_ok and short_ok:
        if score >= short_score:
            verdict = "long"
            final_reasons = long_reasons
            final_score = score
        else:
            verdict = "short"
            final_reasons = short_reasons
            final_score = short_score
    elif long_ok:
        verdict = "long"
        final_reasons = long_reasons
        final_score = score
    elif short_ok:
        verdict = "short"
        final_reasons = short_reasons
        final_score = short_score
    else:
        verdict = "neutral"
        final_reasons = []
        final_score = 0

    return {
        "trend_verdict": verdict,
        "trend_score": final_score,
        "trend_reasons": final_reasons,
        "pool_sources": list(pool_sources or []),
        "d6h": d6h,
        "px_chg": px_chg,
    }


def partition_trend_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 trend_verdict 分组并排序（score 降序）。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "trend_long": [],
        "trend_short": [],
        "trend_avoid": [],
        "trend_neutral": [],
    }
    key_map = {
        "long": "trend_long",
        "short": "trend_short",
        "avoid": "trend_avoid",
        "neutral": "trend_neutral",
    }
    for it in items:
        v = str(it.get("trend_verdict") or "neutral")
        buckets[key_map.get(v, "trend_neutral")].append(it)
    for k in buckets:
        buckets[k].sort(
            key=lambda x: (-int(x.get("trend_score") or 0), str(x.get("symbol") or ""))
        )
    return buckets
