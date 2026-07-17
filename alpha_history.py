#!/usr/bin/env python3
"""
Alpha 每期总结历史：上新/空投每期落盘，默认保留 180 天。

在持仓刷新后自动写入；GET /api/alpha/history 读取。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
HISTORY_NAME = "alpha_period_history.json"
DEFAULT_RETENTION_DAYS = 180
MAX_SUMMARIES_PER_PERIOD = 48  # 单期最多保留多少次快照总结

_lock = threading.Lock()


def _now_cst() -> datetime:
    return datetime.now(CST)


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        return dt.astimezone(CST)
    except Exception:
        return None


def _retention_days() -> int:
    try:
        return max(30, int(os.getenv("ALPHA_HISTORY_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))))
    except Exception:
        return DEFAULT_RETENTION_DAYS


def _history_path() -> Path:
    return resolve_data_dir() / HISTORY_NAME


def _period_id(symbol: str, start_at_cst: str) -> str:
    return f"{str(symbol or '').upper()}|{start_at_cst}"


def _load_raw() -> Dict[str, Any]:
    path = _history_path()
    if not path.is_file():
        return {"ok": True, "periods": [], "updated_at_cst": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"ok": True, "periods": [], "updated_at_cst": None}
        data.setdefault("periods", [])
        data["ok"] = True
        return data
    except Exception as e:
        logger.warning("alpha history read failed: %s", e)
        return {"ok": True, "periods": [], "updated_at_cst": None, "error": str(e)}


def _save_raw(data: Dict[str, Any]) -> None:
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at_cst"] = _now_cst().isoformat()
    data["ok"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_history(periods: List[Dict[str, Any]], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    now = now or _now_cst()
    cutoff = now - timedelta(days=_retention_days())
    kept: List[Dict[str, Any]] = []
    for p in periods:
        start = _parse_iso(str(p.get("start_at_cst") or ""))
        updated = _parse_iso(str(p.get("updated_at_cst") or ""))
        anchor = start or updated
        if anchor is None or anchor >= cutoff:
            kept.append(p)
    return kept


def _one_liner(agg: Dict[str, Any], phase: Optional[str]) -> str:
    label = str(agg.get("signal_label") or agg.get("signal") or "无信号")
    action = str(agg.get("action") or "").strip()
    movers = agg.get("simultaneous_movers")
    pressure = agg.get("pressure_pp")
    bits = [label]
    if phase:
        bits.append(f"阶段={phase}")
    if movers is not None:
        bits.append(f"同动{movers}")
    if pressure is not None:
        bits.append(f"抛压{pressure}pp")
    if action:
        bits.append(action)
    return " · ".join(bits)


def build_summary_from_watch(
    watch: Dict[str, Any],
    *,
    calendar_item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cal = calendar_item or {}
    agg = watch.get("aggregate") if isinstance(watch.get("aggregate"), dict) else {}
    phase = None
    for c in watch.get("chains") or []:
        a = c.get("analysis") or {}
        if a.get("phase"):
            phase = a.get("phase")
            break
    phase = phase or cal.get("phase") or _phase_hint(cal.get("start_at_cst"))

    chains_brief = []
    for c in watch.get("chains") or []:
        a = c.get("analysis") or {}
        chains_brief.append(
            {
                "chain": c.get("chain"),
                "chain_label": c.get("chain_label"),
                "top10_share_pct": c.get("top10_share_pct"),
                "signal": a.get("signal"),
                "signal_label": a.get("signal_label"),
                "bias": a.get("bias"),
                "pressure_pp": a.get("pressure_pp"),
                "simultaneous_movers": a.get("simultaneous_movers"),
            }
        )

    summary = {
        "at_cst": _now_cst().isoformat(),
        "phase": phase,
        "signal": agg.get("signal"),
        "signal_label": agg.get("signal_label"),
        "bias": agg.get("bias"),
        "action": agg.get("action"),
        "quote": agg.get("quote"),
        "pressure_pp": agg.get("pressure_pp"),
        "simultaneous_movers": agg.get("simultaneous_movers"),
        "outflow_share_pct": agg.get("outflow_share_pct"),
        "has_baseline": agg.get("has_baseline"),
        "playbook_steps": agg.get("playbook_steps") or [],
        "chains": chains_brief,
        "errors": watch.get("errors") or [],
        "one_liner": _one_liner(agg, phase),
    }
    return summary


def _phase_hint(start_at_cst: Any) -> Optional[str]:
    start = _parse_iso(str(start_at_cst or ""))
    if start is None:
        return None
    delta_min = (start - _now_cst()).total_seconds() / 60.0
    if -10 <= delta_min <= 0:
        return "early_window"
    if 0 < delta_min <= 24 * 60:
        return "upcoming"
    if delta_min < -10:
        return "live"
    return "scheduled"


def record_period_summary(
    *,
    symbol: str,
    name: str = "",
    coingecko_id: str = "",
    event: str = "",
    start_at_cst: str = "",
    note: str = "",
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """写入/更新一期总结；返回该期记录。"""
    with _lock:
        data = _load_raw()
        periods: List[Dict[str, Any]] = list(data.get("periods") or [])
        pid = _period_id(symbol, start_at_cst or summary.get("at_cst") or "")
        found = None
        for p in periods:
            if p.get("id") == pid:
                found = p
                break
        if found is None:
            found = {
                "id": pid,
                "symbol": str(symbol or "").upper(),
                "name": name,
                "coingecko_id": coingecko_id,
                "event": event,
                "start_at_cst": start_at_cst,
                "note": note,
                "summaries": [],
            }
            periods.append(found)

        summaries = list(found.get("summaries") or [])
        summaries.insert(0, summary)
        found["summaries"] = summaries[:MAX_SUMMARIES_PER_PERIOD]
        found["latest"] = summary
        found["updated_at_cst"] = summary.get("at_cst") or _now_cst().isoformat()
        found["name"] = name or found.get("name")
        found["note"] = note or found.get("note")
        found["event"] = event or found.get("event")
        found["coingecko_id"] = coingecko_id or found.get("coingecko_id")

        periods = prune_history(periods)
        # 按开盘时间倒序
        periods.sort(key=lambda x: str(x.get("start_at_cst") or x.get("updated_at_cst") or ""), reverse=True)
        data["periods"] = periods
        data["retention_days"] = _retention_days()
        _save_raw(data)
        return found


def record_from_watch_payload(
    watches: List[Dict[str, Any]],
    calendar_items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """持仓刷新后：对每个 watch 写一期总结。"""
    cal_by_id = {}
    for item in calendar_items or []:
        cid = str(item.get("coingecko_id") or "").strip()
        if cid:
            cal_by_id[cid] = item

    written: List[Dict[str, Any]] = []
    for w in watches or []:
        cid = str(w.get("coingecko_id") or "").strip()
        if not cid:
            continue
        cal = cal_by_id.get(cid) or {}
        # 未开盘且无基线的纯 upcoming 也可记一笔「观察中」
        summary = build_summary_from_watch(w, calendar_item=cal)
        period = record_period_summary(
            symbol=str(w.get("symbol") or cal.get("symbol") or ""),
            name=str(w.get("name") or cal.get("name") or ""),
            coingecko_id=cid,
            event=str(cal.get("event") or ""),
            start_at_cst=str(cal.get("start_at_cst") or ""),
            note=str(cal.get("note") or ""),
            summary=summary,
        )
        written.append(period)
    return written


def list_history(limit: int = 50) -> Dict[str, Any]:
    with _lock:
        data = _load_raw()
        periods = prune_history(list(data.get("periods") or []))
        if len(periods) != len(data.get("periods") or []):
            data["periods"] = periods
            data["retention_days"] = _retention_days()
            _save_raw(data)
        periods = periods[: max(1, min(int(limit), 200))]
        return {
            "ok": True,
            "retention_days": _retention_days(),
            "count": len(periods),
            "periods": periods,
            "updated_at_cst": data.get("updated_at_cst"),
            "snapshot_source": "disk",
        }


def get_period(period_id: str) -> Optional[Dict[str, Any]]:
    data = list_history(limit=200)
    for p in data.get("periods") or []:
        if p.get("id") == period_id:
            return p
    return None
