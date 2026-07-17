#!/usr/bin/env python3
"""
跨所纸面对锁账本：多腿开平 + 扣除手续费（可选估算持仓期费率收益）。

默认按各所 taker 费率计开/平各一次（双边 × 开平 = 4 笔费率）。
不下真单。程序可自动开平（XARB_PAPER_AUTO=1）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _env_truthy(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _paper_size() -> float:
    try:
        return max(10.0, float(os.getenv("XARB_PAPER_SIZE_USD", "1000")))
    except Exception:
        return 1000.0


def _max_open() -> int:
    try:
        return max(1, int(os.getenv("XARB_PAPER_MAX_OPEN", "5")))
    except Exception:
        return 5


def _max_hold_hours() -> float:
    try:
        return max(1.0, float(os.getenv("XARB_PAPER_MAX_HOLD_HOURS", "24")))
    except Exception:
        return 24.0


def _close_funding_ratio() -> float:
    """当前费率差 < 开仓阈值 × 该比例 → 平仓。"""
    try:
        return min(1.0, max(0.05, float(os.getenv("XARB_PAPER_CLOSE_FR_RATIO", "0.4"))))
    except Exception:
        return 0.4


def _hours_held(opened_at_cst: Any) -> float:
    try:
        opened = datetime.fromisoformat(str(opened_at_cst))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=CST)
        return max(0.0, (_now_cst() - opened).total_seconds() / 3600.0)
    except Exception:
        return 0.0


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
        dirty = False
        for t in trades:
            if t.get("status") == "closed" and t.get("net_pnl_usd") is None:
                t.update(_calc_pnl(t))
                dirty = True
        if dirty:
            data["trades"] = trades
            _save(data)
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
            "auto": {
                "enabled": _env_truthy("XARB_PAPER_AUTO", default=True),
                "size_usd": _paper_size(),
                "max_open": _max_open(),
                "max_hold_hours": _max_hold_hours(),
                "primary_only": not _env_truthy("XARB_PAPER_AUTO_SECONDARY", default=False),
                "hint": "程序自动开平 HL↔BP 费率差纸面仓；人工只看结果。",
            },
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
    auto: bool = False,
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
        "auto": bool(auto),
        "close_reason": None,
    }
    with _lock:
        data = _load()
        trades = list(data.get("trades") or [])
        # 自动模式：同 base+多空所 已有开仓则拒绝
        if auto:
            for t in trades:
                if (
                    t.get("status") == "open"
                    and str(t.get("base") or "").upper() == base_u
                    and str(t.get("ex_long") or "").lower() == el
                    and str(t.get("ex_short") or "").lower() == es
                ):
                    return {"ok": False, "skipped": "already_open", "trade": t}
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
    close_reason: str = "",
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
            found["hours_held"] = round(_hours_held(found.get("opened_at_cst")), 4)
        found["status"] = "closed"
        found["closed_at_cst"] = _now_cst().isoformat()
        found["close_reason"] = str(close_reason or "").strip() or None
        found.update(_calc_pnl(found))
        _save(data)
        return {"ok": True, "trade": found}


def _find_row(
    rows: list,
    *,
    base: str,
    ex_long: str,
    ex_short: str,
) -> Optional[Dict[str, Any]]:
    want = {str(ex_long).lower(), str(ex_short).lower()}
    base_u = str(base or "").upper()
    for r in rows or []:
        if str(r.get("base") or "").upper() != base_u:
            continue
        if {str(r.get("ex_a") or "").lower(), str(r.get("ex_b") or "").lower()} == want:
            return r
    return None


def _safe_px(row: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = float(row.get(key))
        return v if v > 0 else None
    except Exception:
        return None


def _leg_prices(
    row: Dict[str, Any],
    *,
    ex_long: str,
    ex_short: str,
) -> Optional[Tuple[float, float, Optional[float], Optional[float]]]:
    """返回 long_px, short_px, funding_8h_long, funding_8h_short。"""
    el = str(ex_long).lower()
    es = str(ex_short).lower()
    a = str(row.get("ex_a") or "").lower()
    b = str(row.get("ex_b") or "").lower()
    if a == el and b == es:
        long_px, short_px = _safe_px(row, "mark_a"), _safe_px(row, "mark_b")
        fr_l, fr_s = row.get("funding_8h_a"), row.get("funding_8h_b")
    elif b == el and a == es:
        long_px, short_px = _safe_px(row, "mark_b"), _safe_px(row, "mark_a")
        fr_l, fr_s = row.get("funding_8h_b"), row.get("funding_8h_a")
    else:
        return None
    if long_px is None or short_px is None:
        return None
    try:
        fr_l_f = float(fr_l) if fr_l is not None else None
        fr_s_f = float(fr_s) if fr_s is not None else None
    except Exception:
        fr_l_f, fr_s_f = None, None
    return long_px, short_px, fr_l_f, fr_s_f


def auto_manage_from_board(board: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据扫描结果自动开/平纸面仓（默认只做 HL↔BP 主对）。
    人工不干预；可用 XARB_PAPER_AUTO=0 关闭。
    """
    if not _env_truthy("XARB_PAPER_AUTO", default=True):
        return {"ok": True, "enabled": False, "opened": [], "closed": []}

    from xarb_radar import _funding_alert_8h  # noqa: PLC0415

    rows = list(board.get("rows") or [])
    fr_alerts = list(board.get("funding_alerts") or [])
    th = board.get("thresholds") or {}
    try:
        open_th = float(th.get("funding_alert_8h") or 0) or _funding_alert_8h()
    except Exception:
        open_th = _funding_alert_8h()
    close_th = open_th * _close_funding_ratio()
    size = _paper_size()
    max_open = _max_open()
    max_hours = _max_hold_hours()
    allow_secondary = _env_truthy("XARB_PAPER_AUTO_SECONDARY", default=False)

    book = list_paper()
    opens = [t for t in (book.get("trades") or []) if t.get("status") == "open"]
    opened: list = []
    closed: list = []

    # —— 先平：费率差收敛 / 超时 ——
    for t in opens:
        base = str(t.get("base") or "").upper()
        el = str(t.get("ex_long") or "").lower()
        es = str(t.get("ex_short") or "").lower()
        row = _find_row(rows, base=base, ex_long=el, ex_short=es)
        hours = _hours_held(t.get("opened_at_cst"))
        reason = None

        if hours >= max_hours:
            reason = f"持仓满 {max_hours:g}h 超时平仓"
        elif row is not None:
            try:
                fr_diff = abs(float(row.get("funding_diff_8h") or 0))
            except Exception:
                fr_diff = 0.0
            if fr_diff < close_th:
                reason = f"费率差收敛至 {fr_diff * 100:.4f}% < 平仓阈值 {close_th * 100:.4f}%"

        if not reason:
            continue

        legs = _leg_prices(row, ex_long=el, ex_short=es) if row else None
        if legs:
            long_px, short_px, t_fr_l, t_fr_s = legs
            try:
                if t_fr_l is not None and t_fr_s is not None:
                    with _lock:
                        data = _load()
                        for x in data.get("trades") or []:
                            if str(x.get("id")) == str(t["id"]) and x.get("status") == "open":
                                x["funding_8h_long"] = t_fr_l
                                x["funding_8h_short"] = t_fr_s
                                break
                        _save(data)
                close_paper(
                    trade_id=str(t["id"]),
                    long_exit=long_px,
                    short_exit=short_px,
                    hours_held=hours,
                    close_reason=reason,
                )
                closed.append({"id": t["id"], "base": base, "reason": reason})
            except Exception as e:
                logger.warning("auto close %s failed: %s", t.get("id"), e)
        else:
            try:
                close_paper(
                    trade_id=str(t["id"]),
                    long_exit=float(t["long_entry"]),
                    short_exit=float(t["short_entry"]),
                    hours_held=hours,
                    close_reason=reason + "（无行情，按入场价平）",
                )
                closed.append({"id": t["id"], "base": base, "reason": reason})
            except Exception as e:
                logger.warning("auto close %s failed: %s", t.get("id"), e)

    # 刷新开仓数
    book = list_paper()
    opens = [t for t in (book.get("trades") or []) if t.get("status") == "open"]

    # —— 再开：主对费率警报 ——
    candidates = []
    for r in fr_alerts:
        if not r.get("funding_alert"):
            continue
        if r.get("primary"):
            candidates.append(r)
        elif allow_secondary:
            candidates.append(r)
    # 无 primary 标记时退回 HL/BP
    if not candidates:
        for r in fr_alerts:
            pair = {str(r.get("ex_a") or "").lower(), str(r.get("ex_b") or "").lower()}
            if pair == {"hyperliquid", "backpack"} and r.get("funding_alert"):
                candidates.append(r)

    for r in candidates:
        if len(opens) + len(opened) >= max_open:
            break
        base = str(r.get("base") or "").upper()
        if not base:
            continue
        try:
            a_higher = float(r.get("funding_8h_a") or 0) >= float(r.get("funding_8h_b") or 0)
            ex_short = str(r["ex_a"] if a_higher else r["ex_b"]).lower()
            ex_long = str(r["ex_b"] if a_higher else r["ex_a"]).lower()
            short_px = float(r["mark_a"] if a_higher else r["mark_b"])
            long_px = float(r["mark_b"] if a_higher else r["mark_a"])
            fr_s = float(r["funding_8h_a"] if a_higher else r["funding_8h_b"])
            fr_l = float(r["funding_8h_b"] if a_higher else r["funding_8h_a"])
        except Exception as e:
            logger.warning("auto open skip bad row %s: %s", base, e)
            continue
        if long_px <= 0 or short_px <= 0:
            continue

        # 已有同向开仓则跳过
        if any(
            str(t.get("base") or "").upper() == base
            and str(t.get("ex_long") or "").lower() == ex_long
            and str(t.get("ex_short") or "").lower() == ex_short
            for t in opens
        ):
            continue

        try:
            out = open_paper(
                base=base,
                ex_long=ex_long,
                ex_short=ex_short,
                long_entry=long_px,
                short_entry=short_px,
                size_usd=size,
                funding_8h_long=fr_l,
                funding_8h_short=fr_s,
                pair=str(r.get("pair") or f"{ex_long}/{ex_short}"),
                note=f"auto·费率差{float(r.get('funding_diff_8h_pct') or 0):.4f}%",
                auto=True,
            )
            if out.get("ok") and out.get("trade"):
                opened.append({"id": out["trade"]["id"], "base": base, "pair": out["trade"].get("pair")})
                opens.append(out["trade"])
        except Exception as e:
            logger.warning("auto open %s failed: %s", base, e)

    return {
        "ok": True,
        "enabled": True,
        "opened": opened,
        "closed": closed,
        "open_th_8h": open_th,
        "close_th_8h": close_th,
        "size_usd": size,
        "max_open": max_open,
        "max_hold_hours": max_hours,
    }
