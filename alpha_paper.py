#!/usr/bin/env python3
"""
Alpha 砸盘窗口纸面交易：现货偏多 + 合约对冲（不真实下单）。

窗口由筹码信号启发：airdrop_dump / airdrop_watch / multi_whale_outflow。
默认程序自动开平（ALPHA_PAPER_AUTO=1）；人工只看账本。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
PAPER_NAME = "alpha_paper_trades.json"
_lock = threading.Lock()

CHECKLIST = [
    {"id": "chip", "text": "程序：筹码砸盘窗口亮起 → 自动开仓"},
    {"id": "price", "text": "程序：拉币安现货/合约价（缺合约则两腿同用现货价）"},
    {"id": "size", "text": "程序：固定名义（ALPHA_PAPER_SIZE_USD）"},
    {"id": "spot_buy", "text": "纸面：现货买入（偏多）"},
    {"id": "hedge_short", "text": "纸面：合约开空对冲（名义≈现货）"},
    {"id": "wait", "text": "程序：窗口熄灭或超时 → 自动平仓"},
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
            "hint": "程序自动托管：窗口亮起会自己开纸面仓。",
            "checklist": CHECKLIST,
        }
    return {
        "ok": True,
        "window_open": False,
        "signal": sig or None,
        "label": "非砸盘窗口",
        "hint": "程序等待空投抛压/多鲸同出；亮起后自动开平，人工不干预。",
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


def _env_truthy(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _paper_size() -> float:
    try:
        return max(10.0, float(os.getenv("ALPHA_PAPER_SIZE_USD", "500")))
    except Exception:
        return 500.0


def _max_open() -> int:
    try:
        return max(1, int(os.getenv("ALPHA_PAPER_MAX_OPEN", "3")))
    except Exception:
        return 3


def _max_hold_hours() -> float:
    try:
        return max(1.0, float(os.getenv("ALPHA_PAPER_MAX_HOLD_HOURS", "12")))
    except Exception:
        return 12.0


def _hours_held(opened_at_cst: Any) -> float:
    try:
        opened = datetime.fromisoformat(str(opened_at_cst))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=CST)
        return max(0.0, (_now_cst() - opened).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def list_paper() -> Dict[str, Any]:
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        dirty = False
        for t in trades:
            if t.get("status") == "closed" and t.get("pnl_usd") is None:
                t["pnl_usd"] = _calc_pnl(t)
                dirty = True
        if dirty:
            data["trades"] = trades
            _save(data)
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
            "auto": {
                "enabled": _env_truthy("ALPHA_PAPER_AUTO", default=True),
                "size_usd": _paper_size(),
                "max_open": _max_open(),
                "max_hold_hours": _max_hold_hours(),
                "hint": "砸盘窗口亮起自动开；窗口熄灭或超时自动平。人工不干预。",
            },
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
    auto: bool = False,
    price_source: str = "",
) -> Dict[str, Any]:
    if not str(symbol or "").strip():
        raise ValueError("symbol_required")
    if float(spot_entry) <= 0 or float(fut_entry) <= 0:
        raise ValueError("entry_price_required")
    if float(size_usd) <= 0:
        raise ValueError("size_usd_required")
    sym = str(symbol).strip().upper()
    trade = {
        "id": str(uuid.uuid4())[:8],
        "status": "open",
        "symbol": sym,
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
        "auto": bool(auto),
        "price_source": str(price_source or "").strip() or None,
        "close_reason": None,
    }
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        if auto:
            for t in trades:
                if t.get("status") == "open" and str(t.get("symbol") or "").upper() == sym:
                    return {"ok": False, "skipped": "already_open", "trade": t}
        trades.insert(0, trade)
        data["trades"] = trades[:200]
        _save(data)
        return {"ok": True, "trade": trade}


def close_paper(
    *,
    trade_id: str,
    spot_exit: float,
    fut_exit: float,
    close_reason: str = "",
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
        found["close_reason"] = str(close_reason or "").strip() or None
        found["pnl_usd"] = _calc_pnl(found)
        _save(data)
        return {"ok": True, "trade": found}


def _bn_price(url: str, symbol: str) -> Optional[float]:
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, dict) and data.get("price") is not None:
            px = float(data["price"])
            return px if px > 0 else None
    except Exception as e:
        logger.debug("bn price %s %s: %s", url, symbol, e)
    return None


def fetch_symbol_prices(symbol: str) -> Optional[Tuple[float, float, str]]:
    """返回 (spot, fut, source)。缺合约时两腿同用现货价。"""
    base = str(symbol or "").strip().upper().replace("/", "")
    if not base:
        return None
    if base.endswith("USDT"):
        pair = base
    else:
        pair = f"{base}USDT"

    spot = _bn_price("https://api.binance.com/api/v3/ticker/price", pair)
    fut = _bn_price("https://fapi.binance.com/fapi/v1/ticker/price", pair)
    if spot and fut:
        return spot, fut, "binance_spot+fut"
    if spot:
        return spot, spot, "binance_spot_only"
    if fut:
        return fut, fut, "binance_fut_only"
    return None


def _watch_signal(watch: Dict[str, Any]) -> str:
    agg = watch.get("aggregate") or {}
    return str(agg.get("signal") or "")


def auto_manage_from_watches(watches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据筹码扫描结果自动开/平砸盘纸面仓。
    开：砸盘窗口信号；平：窗口熄灭或超时。
    """
    if not _env_truthy("ALPHA_PAPER_AUTO", default=True):
        return {"ok": True, "enabled": False, "opened": [], "closed": []}

    size = _paper_size()
    max_open = _max_open()
    max_hours = _max_hold_hours()
    by_sym: Dict[str, Dict[str, Any]] = {}
    by_cid: Dict[str, Dict[str, Any]] = {}
    for w in watches or []:
        sym = str(w.get("symbol") or "").strip().upper()
        cid = str(w.get("coingecko_id") or "").strip().lower()
        if sym:
            by_sym[sym] = w
        if cid:
            by_cid[cid] = w

    book = list_paper()
    opens = [t for t in (book.get("trades") or []) if t.get("status") == "open"]
    opened: list = []
    closed: list = []

    # —— 先平 ——
    for t in opens:
        sym = str(t.get("symbol") or "").upper()
        cid = str(t.get("coingecko_id") or "").strip().lower()
        watch = (by_cid.get(cid) if cid else None) or by_sym.get(sym)
        sig = _watch_signal(watch) if watch else ""
        hours = _hours_held(t.get("opened_at_cst"))

        reason = None
        if hours >= max_hours:
            reason = f"持仓满 {max_hours:g}h 超时平仓"
        elif watch is not None and sig not in WINDOW_SIGNALS:
            reason = f"砸盘窗口熄灭（当前信号 {sig or 'quiet'}）"
        elif watch is None and hours >= max(2.0, max_hours * 0.25):
            reason = "标的已离监控列表，平仓"

        if not reason:
            continue

        px = fetch_symbol_prices(sym)
        if not px:
            spot_px = float(t["spot_entry"])
            fut_px = float(t["fut_entry"])
            reason = reason + "（无行情，按入场价平）"
        else:
            spot_px, fut_px, _src = px

        try:
            close_paper(
                trade_id=str(t["id"]),
                spot_exit=spot_px,
                fut_exit=fut_px,
                close_reason=reason,
            )
            closed.append({"id": t["id"], "symbol": sym, "reason": reason})
        except Exception as e:
            logger.warning("alpha auto close %s failed: %s", t.get("id"), e)

    book = list_paper()
    opens = [t for t in (book.get("trades") or []) if t.get("status") == "open"]
    open_syms = {str(t.get("symbol") or "").upper() for t in opens}

    # —— 再开 ——
    candidates = []
    for w in watches or []:
        sig = _watch_signal(w)
        if sig not in WINDOW_SIGNALS:
            continue
        sym = str(w.get("symbol") or "").strip().upper()
        if not sym or sym in open_syms:
            continue
        candidates.append(w)

    # 优先更狠的信号
    rank = {"airdrop_dump": 0, "multi_whale_outflow": 1, "airdrop_watch": 2}
    candidates.sort(key=lambda w: rank.get(_watch_signal(w), 9))

    for w in candidates:
        if len(opens) + len(opened) >= max_open:
            break
        sym = str(w.get("symbol") or "").strip().upper()
        sig = _watch_signal(w)
        px = fetch_symbol_prices(sym)
        if not px:
            logger.info("alpha paper skip %s: no binance price", sym)
            continue
        spot_px, fut_px, src = px
        if spot_px <= 0 or fut_px <= 0:
            continue
        try:
            out = open_paper(
                symbol=sym,
                spot_entry=spot_px,
                fut_entry=fut_px,
                size_usd=size,
                signal=sig,
                coingecko_id=str(w.get("coingecko_id") or ""),
                note=f"auto·{sig}·{src}",
                auto=True,
                price_source=src,
            )
            if out.get("ok") and out.get("trade"):
                opened.append({"id": out["trade"]["id"], "symbol": sym, "signal": sig})
                opens.append(out["trade"])
                open_syms.add(sym)
        except Exception as e:
            logger.warning("alpha auto open %s failed: %s", sym, e)

    return {
        "ok": True,
        "enabled": True,
        "opened": opened,
        "closed": closed,
        "size_usd": size,
        "max_open": max_open,
        "max_hold_hours": max_hours,
    }
