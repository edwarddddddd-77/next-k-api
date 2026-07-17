#!/usr/bin/env python3
"""
跨所纸面对锁账本：多腿开平 + 扣除手续费（可选估算持仓期费率收益）。

默认按各所 taker 费率计开/平各一次（双边 × 开平 = 4 笔费率）。
不下真单。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
PAPER_NAME = "xarb_paper_trades.json"
_lock = threading.Lock()

# 默认 taker 费率（小数）；可用 XARB_FEE_<EX> 覆盖，如 XARB_FEE_HYPERLIQUID=0.00025
DEFAULT_TAKER_FEES = {
    "binance": 0.0004,  # 0.04%
    "hyperliquid": 0.00035,  # 0.035%
    "backpack": 0.0002,  # 0.02%
    "okx": 0.0005,  # 0.05%
}

VALID_EX = tuple(DEFAULT_TAKER_FEES.keys())


def _now_cst() -> datetime:
    return datetime.now(CST)


def _path() -> Path:
    return resolve_data_dir() / PAPER_NAME


def fee_rate(exchange: str) -> float:
    ex = str(exchange or "").strip().lower()
    env_key = f"XARB_FEE_{ex.upper()}"
    raw = (os.getenv(env_key) or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except Exception:
            pass
    return float(DEFAULT_TAKER_FEES.get(ex, 0.0004))


def fee_table() -> Dict[str, float]:
    return {ex: fee_rate(ex) for ex in VALID_EX}


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
        logger.warning("xarb paper read failed: %s", e)
        return {"ok": True, "trades": [], "updated_at_cst": None, "error": str(e)}


def _save(data: Dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["ok"] = True
    data["updated_at_cst"] = _now_cst().isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _calc_pnl(trade: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """价差盈亏 + 费率估算 − 手续费。"""
    try:
        size = float(trade.get("size_usd") or 0)
        long_in = float(trade.get("long_entry"))
        short_in = float(trade.get("short_entry"))
        long_out = trade.get("long_exit")
        short_out = trade.get("short_exit")
        if size <= 0 or long_in <= 0 or short_in <= 0 or long_out is None or short_out is None:
            return {
                "price_pnl_usd": None,
                "funding_pnl_usd": None,
                "fees_usd": None,
                "net_pnl_usd": None,
            }
        long_out_f = float(long_out)
        short_out_f = float(short_out)
        if long_out_f <= 0 or short_out_f <= 0:
            return {
                "price_pnl_usd": None,
                "funding_pnl_usd": None,
                "fees_usd": None,
                "net_pnl_usd": None,
            }

        long_pnl = size * (long_out_f - long_in) / long_in
        short_pnl = size * (short_in - short_out_f) / short_in
        price_pnl = round(long_pnl + short_pnl, 4)

        ex_l = str(trade.get("ex_long") or "")
        ex_s = str(trade.get("ex_short") or "")
        fee_l = float(trade.get("fee_rate_long") if trade.get("fee_rate_long") is not None else fee_rate(ex_l))
        fee_s = float(trade.get("fee_rate_short") if trade.get("fee_rate_short") is not None else fee_rate(ex_s))
        # 开仓双边 + 平仓双边
        fees = round(size * (fee_l + fee_s) * 2.0, 4)

        funding_pnl = 0.0
        hours = trade.get("hours_held")
        fr_l = trade.get("funding_8h_long")
        fr_s = trade.get("funding_8h_short")
        if hours is not None and fr_l is not None and fr_s is not None:
            # 空腿收/付 funding_8h_short，多腿付/收 funding_8h_long；按持有小时折算
            h = max(0.0, float(hours))
            funding_pnl = round(size * (float(fr_s) - float(fr_l)) * (h / 8.0), 4)

        net = round(price_pnl + funding_pnl - fees, 4)
        return {
            "price_pnl_usd": price_pnl,
            "funding_pnl_usd": funding_pnl,
            "fees_usd": fees,
            "net_pnl_usd": net,
        }
    except Exception:
        return {
            "price_pnl_usd": None,
            "funding_pnl_usd": None,
            "fees_usd": None,
            "net_pnl_usd": None,
        }


def list_paper() -> Dict[str, Any]:
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        for t in trades:
            if t.get("status") == "closed" and t.get("net_pnl_usd") is None:
                t.update(_calc_pnl(t))
        open_n = sum(1 for t in trades if t.get("status") == "open")
        closed = [t for t in trades if t.get("status") == "closed" and t.get("net_pnl_usd") is not None]
        total_net = round(sum(float(t["net_pnl_usd"]) for t in closed), 4) if closed else 0.0
        total_fees = round(sum(float(t.get("fees_usd") or 0) for t in closed), 4) if closed else 0.0
        return {
            "ok": True,
            "trades": trades,
            "open_count": open_n,
            "closed_count": len(closed),
            "total_net_pnl_usd": total_net,
            "total_fees_usd": total_fees,
            "fee_table": fee_table(),
            "fee_note": "每腿按 taker 计开+平；双边对锁共扣 4 笔手续费。可用 XARB_FEE_<EX> 覆盖。",
            "updated_at_cst": data.get("updated_at_cst"),
        }


def open_paper(
    *,
    base: str,
    ex_long: str,
    ex_short: str,
    long_entry: float,
    short_entry: float,
    size_usd: float,
    funding_8h_long: Optional[float] = None,
    funding_8h_short: Optional[float] = None,
    note: str = "",
    pair: str = "",
) -> Dict[str, Any]:
    base_u = str(base or "").strip().upper()
    el = str(ex_long or "").strip().lower()
    es = str(ex_short or "").strip().lower()
    if not base_u:
        raise ValueError("base_required")
    if el not in VALID_EX or es not in VALID_EX:
        raise ValueError(f"exchange_must_be_one_of:{','.join(VALID_EX)}")
    if el == es:
        raise ValueError("ex_long_and_short_must_differ")
    if float(long_entry) <= 0 or float(short_entry) <= 0:
        raise ValueError("entry_price_required")
    if float(size_usd) <= 0:
        raise ValueError("size_usd_required")

    fee_l = fee_rate(el)
    fee_s = fee_rate(es)
    # 开仓时先记下预计全周期手续费（开+平）
    est_fees = round(float(size_usd) * (fee_l + fee_s) * 2.0, 4)

    trade = {
        "id": str(uuid.uuid4())[:8],
        "status": "open",
        "base": base_u,
        "pair": str(pair or f"{el}/{es}"),
        "ex_long": el,
        "ex_short": es,
        "long_entry": float(long_entry),
        "short_entry": float(short_entry),
        "long_exit": None,
        "short_exit": None,
        "size_usd": float(size_usd),
        "fee_rate_long": fee_l,
        "fee_rate_short": fee_s,
        "est_fees_usd": est_fees,
        "funding_8h_long": float(funding_8h_long) if funding_8h_long is not None else None,
        "funding_8h_short": float(funding_8h_short) if funding_8h_short is not None else None,
        "hours_held": None,
        "price_pnl_usd": None,
        "funding_pnl_usd": None,
        "fees_usd": None,
        "net_pnl_usd": None,
        "note": str(note or "").strip(),
        "opened_at_cst": _now_cst().isoformat(),
        "closed_at_cst": None,
        "legs": f"long@{el} + short@{es}",
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
    long_exit: float,
    short_exit: float,
    hours_held: Optional[float] = None,
) -> Dict[str, Any]:
    tid = str(trade_id or "").strip()
    if not tid:
        raise ValueError("trade_id_required")
    if float(long_exit) <= 0 or float(short_exit) <= 0:
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
        found["long_exit"] = float(long_exit)
        found["short_exit"] = float(short_exit)
        if hours_held is not None:
            found["hours_held"] = max(0.0, float(hours_held))
        else:
            # 自动按开仓时间估算
            try:
                opened = datetime.fromisoformat(str(found.get("opened_at_cst")))
                found["hours_held"] = round(max(0.0, (_now_cst() - opened).total_seconds() / 3600.0), 4)
            except Exception:
                found["hours_held"] = 0.0
        found["status"] = "closed"
        found["closed_at_cst"] = _now_cst().isoformat()
        found.update(_calc_pnl(found))
        _save(data)
        return {"ok": True, "trade": found}
