"""寻优验证：历史资金费率 gate 代理（与纸面 trade_gates 费率逻辑对齐）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from moss_quant import config as cfg

logger = logging.getLogger(__name__)

_FUNDING_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _ms(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def fetch_binance_funding_history(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """拉取 [start_ms, end_ms] 内币安 U 本位资金费率（升序）。"""
    sym = str(symbol or "").upper()
    cache_key = f"{sym}:{start_ms}:{end_ms}"
    if cache_key in _FUNDING_CACHE:
        return _FUNDING_CACHE[cache_key]

    try:
        from binance_fapi import api_get
    except ImportError:
        return []

    rows: List[Dict[str, Any]] = []
    cursor = int(start_ms)
    limit = 1000
    while cursor <= end_ms:
        data = api_get(
            "/fapi/v1/fundingRate",
            {
                "symbol": sym,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        if not isinstance(data, list) or not data:
            break
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                t_ms = int(item.get("fundingTime") or 0)
                fr = float(item.get("fundingRate") or 0)
            except (TypeError, ValueError):
                continue
            rows.append({"time_ms": t_ms, "rate": fr})
        last_ms = int(data[-1].get("fundingTime") or 0)
        if last_ms <= cursor or len(data) < limit:
            break
        cursor = last_ms + 1

    rows.sort(key=lambda r: r["time_ms"])
    _FUNDING_CACHE[cache_key] = rows
    if rows:
        extreme = float(cfg.MOSS_QUANT_GATE_FUNDING_ABS_MAX)
        logger.debug(
            "[moss] gate_proxy %s funding rows=%s extreme>=%s count=%s",
            sym,
            len(rows),
            extreme,
            sum(1 for r in rows if abs(r["rate"]) >= extreme),
        )
    return rows


def funding_extreme_stats(
    df: pd.DataFrame,
    symbol: str,
) -> Dict[str, Any]:
    """
    估算验证窗内「费率极端」占比（多空任一侧会触发 gate bump 的 bar 比例）。
    无费率数据时返回空统计，不惩罚。
    """
    if not cfg.MOSS_QUANT_OPTIMIZE_GATE_PROXY_ENABLED or not cfg.MOSS_QUANT_GATE_FUNDING_EXTREME:
        return {"extreme_ratio": 0.0, "avg_bump": 0.0, "funding_rows": 0}
    if df is None or len(df) < 8 or "timestamp" not in df.columns:
        return {"extreme_ratio": 0.0, "avg_bump": 0.0, "funding_rows": 0}

    ts = pd.to_datetime(df["timestamp"], utc=True)
    start_ms = _ms(ts.iloc[0])
    end_ms = _ms(ts.iloc[-1])
    funding = fetch_binance_funding_history(str(symbol).upper(), start_ms=start_ms, end_ms=end_ms)
    if not funding:
        return {"extreme_ratio": 0.0, "avg_bump": 0.0, "funding_rows": 0}

    extreme = float(cfg.MOSS_QUANT_GATE_FUNDING_ABS_MAX)
    bump = float(cfg.MOSS_QUANT_GATE_FUNDING_BUMP)
    f_times = [int(r["time_ms"]) for r in funding]
    f_rates = [float(r["rate"]) for r in funding]

    extreme_bars = 0
    bump_sum = 0.0
    for bar_ts in ts:
        bar_ms = _ms(bar_ts)
        idx = 0
        for i, t_ms in enumerate(f_times):
            if t_ms <= bar_ms:
                idx = i
        fr = f_rates[idx]
        if abs(fr) >= extreme:
            extreme_bars += 1
            bump_sum += bump

    n = len(ts)
    ratio = extreme_bars / n if n else 0.0
    return {
        "extreme_ratio": round(ratio, 4),
        "avg_bump": round(bump_sum / n, 4) if n else 0.0,
        "funding_rows": len(funding),
    }


def validation_gate_penalty(stats: Optional[Dict[str, Any]]) -> float:
    """按极端费率 bar 占比折算验证收益惩罚（0～scale）。"""
    if not cfg.MOSS_QUANT_OPTIMIZE_GATE_PROXY_ENABLED:
        return 0.0
    ratio = float((stats or {}).get("extreme_ratio") or 0)
    scale = float(cfg.MOSS_QUANT_OPTIMIZE_GATE_PENALTY_SCALE)
    return round(max(0.0, min(scale, ratio * scale)), 6)


def gate_fail_reason(stats: Optional[Dict[str, Any]]) -> Optional[str]:
    """可选：验证窗费率极端占比过高则直接不通过。"""
    if not cfg.MOSS_QUANT_OPTIMIZE_GATE_PROXY_ENABLED:
        return None
    fail_ratio = float(cfg.MOSS_QUANT_OPTIMIZE_GATE_FAIL_RATIO or 0)
    if fail_ratio <= 0:
        return None
    ratio = float((stats or {}).get("extreme_ratio") or 0)
    if ratio >= fail_ratio:
        return f"验证窗费率极端占比{ratio * 100:.0f}%（阈值 {fail_ratio * 100:.0f}%）"
    return None


def validation_reachable_penalty(stats: Optional[Dict[str, Any]]) -> float:
    """验证窗信号可达性过低或子样本 PF 差 → 扣减验证分。"""
    if not cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_ENABLED:
        return 0.0
    s = stats or {}
    scale = float(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_PENALTY_SCALE)
    min_ratio = float(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_MIN_RATIO)
    ratio = float(s.get("reachable_ratio") or 0)
    penalty = 0.0
    if ratio < min_ratio and min_ratio > 0:
        penalty += scale * (1.0 - ratio / min_ratio)
    sub_n = int(s.get("reachable_sub_trades") or 0)
    sub_pf = float(s.get("reachable_sub_pf") or 0)
    min_pf = float(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_MIN_SUB_PF)
    min_sub = int(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_MIN_SUB_TRADES)
    if sub_n >= min_sub and sub_pf < min_pf:
        penalty += scale * 0.5 * (1.0 - max(0.0, sub_pf) / max(min_pf, 1e-9))
    return round(max(0.0, min(scale * 2.0, penalty)), 6)


def reachable_fail_reason(stats: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_ENABLED:
        return None
    fail_ratio = float(cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_FAIL_RATIO or 0)
    if fail_ratio <= 0:
        return None
    ratio = float((stats or {}).get("reachable_ratio") or 0)
    if ratio < fail_ratio:
        return (
            f"验证窗信号可达性{ratio * 100:.2f}%"
            f"（阈值 {fail_ratio * 100:.2f}%）"
        )
    return None
