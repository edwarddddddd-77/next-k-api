#!/usr/bin/env python3
"""
Alpha 砸盘窗口纸面交易：现货偏多 + 合约对冲（不真实下单）。

窗口由筹码信号启发：airdrop_dump / airdrop_watch / multi_whale_outflow。
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
PAPER_NAME = "alpha_paper_trades.json"
_lock = threading.Lock()

CHECKLIST = [
    {"id": "chip", "text": "筹码显示抛压/多鲸同出（砸盘窗口）"},
    {"id": "spot_dip", "text": "现货已出现明显急跌/插针（目视或行情）"},
    {"id": "size", "text": "纸面仓位已定：小仓，可承受继续阴跌"},
    {"id": "spot_buy", "text": "纸面：现货买入（偏多）"},
    {"id": "hedge_short", "text": "纸面：合约开空对冲（名义≈现货）"},
    {"id": "wait", "text": "等抛压衰减 / 同动变少，再考虑减空留现货"},
    {"id": "exit", "text": "纸面平仓：现货卖出 + 空单平掉，记盈亏"},
]

WINDOW_SIGNALS = {
    "airdrop_dump": {"open": True, "label": "空投抛压 · 砸盘窗口", "hint": "可纸面：轻仓现货 + 合约对冲"},
    "airdrop_watch": {"open": True, "label": "空投开始动 · 盯砸盘", "hint": "抛压放大再入；先备好对冲计划"},
    "multi_whale_outflow": {"open": True, "label": "多鲸同出 · 流通放大", "hint": "偏砸盘；纸面可试对冲抄底，勿裸多"},
}


def _now_cst() -> datetime:
    return datetime.now(CST)


def _path() -> Path:
    return resolve_data_dir() / PAPER_NAME


def _load() -> Dict[str, Any]:
    path = _path()
    if not path.is_file():
        return {"ok": True, "trades": [], "updated_at_cst": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"ok": True, "trades": [], "updated_at_cst": None}
        data.setdefault("trades", [])
        data["ok"] = True
        return data
    except Exception as e:
        logger.warning("alpha paper read failed: %s", e)
        return {"ok": True, "trades": [], "updated_at_cst": None, "error": str(e)}


def _save(data: Dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["ok"] = True
    data["updated_at_cst"] = _now_cst().isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def window_status(signal: Optional[str] = None, bias: Optional[str] = None) -> Dict[str, Any]:
    sig = str(signal or "")
    hit = WINDOW_SIGNALS.get(sig)
    if hit:
        return {
            "ok": True,
            "window_open": True,
            "signal": sig,
            "label": hit["label"],
            "hint": hit["hint"],
            "checklist": CHECKLIST,
        }
    return {
        "ok": True,
        "window_open": False,
        "signal": sig or None,
        "label": "非砸盘窗口",
        "hint": "等筹码出现空投抛压/多鲸同出，再开纸面对冲。",
        "bias": bias,
        "checklist": CHECKLIST,
    }


def _calc_pnl(trade: Dict[str, Any]) -> Optional[float]:
    """现货盈亏 + 空单盈亏（名义等权简化）。"""
    try:
        spot_in = float(trade.get("spot_entry"))
        fut_in = float(trade.get("fut_entry"))
        size = float(trade.get("size_usd") or 0)
        spot_out = trade.get("spot_exit")
        fut_out = trade.get("fut_exit")
        if spot_out is None or fut_out is None or size <= 0 or spot_in <= 0 or fut_in <= 0:
            return None
        spot_out_f = float(spot_out)
        fut_out_f = float(fut_out)
        # 现货多：涨赚；合约空：跌赚
        spot_pnl = size * (spot_out_f - spot_in) / spot_in
        fut_pnl = size * (fut_in - fut_out_f) / fut_in
        return round(spot_pnl + fut_pnl, 4)
    except Exception:
        return None


def list_paper() -> Dict[str, Any]:
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        for t in trades:
            if t.get("status") == "closed" and t.get("pnl_usd") is None:
                t["pnl_usd"] = _calc_pnl(t)
        open_n = sum(1 for t in trades if t.get("status") == "open")
        closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl_usd") is not None]
        total_pnl = round(sum(float(t["pnl_usd"]) for t in closed), 4) if closed else 0.0
        return {
            "ok": True,
            "trades": trades,
            "open_count": open_n,
            "closed_count": len(closed),
            "total_pnl_usd": total_pnl,
            "checklist": CHECKLIST,
            "updated_at_cst": data.get("updated_at_cst"),
        }


def open_paper(
    *,
    symbol: str,
    spot_entry: float,
    fut_entry: float,
    size_usd: float,
    note: str = "",
    signal: str = "",
    coingecko_id: str = "",
) -> Dict[str, Any]:
    if not str(symbol or "").strip():
        raise ValueError("symbol_required")
    if float(spot_entry) <= 0 or float(fut_entry) <= 0:
        raise ValueError("entry_price_required")
    if float(size_usd) <= 0:
        raise ValueError("size_usd_required")
    trade = {
        "id": str(uuid.uuid4())[:8],
        "status": "open",
        "symbol": str(symbol).strip().upper(),
        "coingecko_id": str(coingecko_id or "").strip() or None,
        "signal": str(signal or "").strip() or None,
        "spot_entry": float(spot_entry),
        "fut_entry": float(fut_entry),
        "size_usd": float(size_usd),
        "spot_exit": None,
        "fut_exit": None,
        "pnl_usd": None,
        "note": str(note or "").strip(),
        "opened_at_cst": _now_cst().isoformat(),
        "closed_at_cst": None,
        "legs": "spot_long + fut_short",
    }
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        trades.insert(0, trade)
        data["trades"] = trades[:200]
        _save(data)
        return {"ok": True, "trade": trade}


def close_paper(
    *,
    trade_id: str,
    spot_exit: float,
    fut_exit: float,
) -> Dict[str, Any]:
    tid = str(trade_id or "").strip()
    if not tid:
        raise ValueError("trade_id_required")
    if float(spot_exit) <= 0 or float(fut_exit) <= 0:
        raise ValueError("exit_price_required")
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        found = None
        for t in trades:
            if str(t.get("id")) == tid:
                found = t
                break
        if not found:
            raise ValueError("trade_not_found")
        if found.get("status") == "closed":
            return {"ok": True, "trade": found, "already_closed": True}
        found["spot_exit"] = float(spot_exit)
        found["fut_exit"] = float(fut_exit)
        found["status"] = "closed"
        found["closed_at_cst"] = _now_cst().isoformat()
        found["pnl_usd"] = _calc_pnl(found)
        _save(data)
        return {"ok": True, "trade": found}
