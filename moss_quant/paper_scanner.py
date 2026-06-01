"""Moss 量化纸面 — 每 profile 单 symbol 单仓。"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.decision import DecisionParams, compute_last_composite, compute_signals
from moss_quant.core.indicators import atr as compute_atr
from moss_quant.core.regime import classify_regime
from moss_quant.db import (
    _utc_now,
    insert_open_signal_from_protocol,
    list_profiles_for_paper_scan,
    settle_profile_external_closed,
    profile_wallet_balance,
    sync_moss_wallet_from_settlements,
)
from moss_quant.kline_cache import load_cached
from moss_quant.params import cap_leverage_for_symbol, resolve_params_dict

logger = logging.getLogger(__name__)

# 实盘信号发送器（延迟导入，结果缓存到进程生命周期结束）
_sender = None


def _get_sender():
    global _sender
    if _sender is None:
        from moss_quant.signal_sender import is_real_mode
        if is_real_mode():
            import moss_quant.signal_sender as s
            _sender = s
            logger.info("[moss_quant] signal sender initialized (REAL MODE)")
        else:
            _sender = False
    return _sender if _sender is not False else None


def _py_scalar(v: Any) -> Any:
    """numpy 标量 → Python float/int，便于 JSON 日志。"""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _verbose() -> bool:
    return bool(getattr(cfg, "MOSS_QUANT_VERBOSE_LOG", True))


def _vlog(msg: str, *args: Any) -> None:
    if _verbose():
        logger.info("[moss] " + msg, *args)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _profile_label(profile: Dict[str, Any]) -> str:
    return "p%d:%s:%s" % (
        int(profile["id"]),
        str(profile.get("symbol") or "").upper(),
        str(profile.get("template") or ""),
    )


def format_scan_detail_message(label: str, detail: Dict[str, Any]) -> str:
    """与 Railway 日志同风格的单行摘要（供前端展示）。"""
    act = str(detail.get("action") or "")
    if act == "error":
        return f"[moss] {label} ERROR {detail.get('error', '')}"
    if act == "wait":
        return (
            f"[moss] {label} WAIT composite={detail.get('composite')} "
            f"thresh=±{detail.get('entry_threshold')} regime={detail.get('regime')} "
            f"reason={detail.get('reason')}"
        )
    if act == "hold":
        return (
            f"[moss] {label} HOLD {detail.get('side', '')} "
            f"entry={detail.get('entry_price')} mark={detail.get('mark_price')} "
            f"upnl={detail.get('upnl')}U pnl%={detail.get('pnl_pct')} "
            f"sl<={detail.get('sl_thresh_pct')}% tp>={detail.get('tp_thresh_pct')}% "
            f"sig={detail.get('signal')}"
        )
    if act == "close":
        return (
            f"[moss] {label} CLOSE {detail.get('side', '')} {detail.get('rule', '')} "
            f"pnl={detail.get('pnl')}U"
        )
    if act == "open":
        return (
            f"[moss] {label} OPEN {detail.get('side', '')} "
            f"entry={detail.get('entry_price')} notional={detail.get('notional')}U "
            f"upnl={detail.get('upnl', 0)}U composite={detail.get('composite')} "
            f"regime={detail.get('regime', '')}"
        )
    return f"[moss] {label} {act}"


def _scan_detail(
    label: str, profile: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    d = {"profile_id": int(profile["id"]), "label": label, **payload}
    d["message"] = format_scan_detail_message(label, d)
    return d


def _margin_pnl_pct(side: str, entry: float, mark: float, leverage: float) -> float:
    """相对保证金的收益率 %（与 hold 日志 pnl% 一致）。"""
    if entry <= 0 or mark <= 0 or leverage <= 0:
        return 0.0
    side_u = side.upper()
    if side_u in ("LONG", "BUY"):
        return (mark - entry) / entry * leverage * 100.0
    return (entry - mark) / entry * leverage * 100.0


def _position_fields(
    *,
    side: str,
    entry: float,
    mark: float,
    notional: float,
    upnl: float,
    leverage: float = 10.0,
) -> Dict[str, Any]:
    return {
        "side": side,
        "entry_price": round(entry, 8),
        "mark_price": round(mark, 8),
        "notional": round(notional, 2),
        "upnl": round(upnl, 4),
        "leverage": round(float(leverage), 2),
        "pnl_pct": round(_margin_pnl_pct(side, entry, mark, leverage), 3),
    }


def protocol_open_positions_by_symbol(
    positions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pos in positions or []:
        symbol = str(pos.get("symbol") or "").upper()
        if not symbol:
            continue
        out.setdefault(symbol, []).append(dict(pos))
    return out


def _protocol_position_notional(pos: Dict[str, Any]) -> float:
    n = pos.get("notional_usdt")
    if n is not None:
        return float(n or 0)
    qty = float(pos.get("quantity") or 0)
    entry = float(pos.get("entry_price") or 0)
    return abs(qty * entry)


def _protocol_position_mark_price(pos: Dict[str, Any]) -> float:
    for key in ("mark_price", "close_price", "entry_price"):
        if key in pos and pos.get(key) is not None:
            return float(pos.get(key) or 0)
    return 0.0


def _protocol_position_unrealized_pnl(pos: Dict[str, Any]) -> float:
    for key in ("unrealized_pnl_usdt", "upnl"):
        if key in pos and pos.get(key) is not None:
            return float(pos.get(key) or 0)
    if str(pos.get("status") or "").lower() == "open" and "pnl_usdt" in pos:
        return float(pos.get("pnl_usdt") or 0)
    return 0.0


def _protocol_position_has_unrealized_pnl(pos: Dict[str, Any]) -> bool:
    return any(
        key in pos and pos.get(key) is not None
        for key in ("unrealized_pnl_usdt", "upnl", "pnl_usdt")
    )


def latest_protocol_open_positions() -> Optional[List[Dict[str, Any]]]:
    from moss_quant.protocol_client import ProtocolClient

    protocol = ProtocolClient.from_env()
    if not protocol.enabled():
        return None
    profile_map: Dict[str, int] = {}
    try:
        from accumulation_radar import init_db

        real_conn = init_db()
        real_conn.row_factory = sqlite3.Row
        try:
            for profile in list_profiles_for_paper_scan(real_conn):
                symbol = str(profile.get("symbol") or "").upper()
                if symbol and symbol not in profile_map:
                    profile_map[symbol] = int(profile["id"])
        finally:
            real_conn.close()
    except Exception:
        profile_map = {}
    out: List[Dict[str, Any]] = []
    for pos in protocol.get_moss_positions(status="open", limit=500):
        entry = float(pos.get("entry_price") or 0)
        mark = _protocol_position_mark_price(pos) or entry
        leverage = float(pos.get("leverage") or 10)
        symbol = str(pos.get("symbol") or "").upper()
        row = {
            "profile_id": profile_map.get(symbol),
            "side": pos.get("side"),
            "symbol": symbol,
            "entry_price": round(entry, 8),
            "mark_price": round(mark, 8),
            "notional": round(_protocol_position_notional(pos), 2),
            "upnl": round(_protocol_position_unrealized_pnl(pos), 4),
            "leverage": round(leverage, 2),
        }
        row["pnl_pct"] = round(
            _margin_pnl_pct(str(row.get("side") or ""), entry, mark, leverage),
            3,
        )
        out.append(row)
    return out


def can_send_live_open(sender: Any, live_opens_allowed: bool) -> bool:
    return sender is None or bool(live_opens_allowed)


@dataclass(frozen=True)
class ProtocolActionResult:
    ok: bool
    error: str = ""
    position_id: Optional[int] = None
    client_ref: str = ""
    entry_price: Optional[float] = None


def _protocol_ingest_traded(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    for detail in resp.get("details") or []:
        if isinstance(detail, dict) and detail.get("action") == "traded":
            return True
    return False


def protocol_ingest_action_result(
    resp: Any,
    *,
    fallback_error: str,
) -> ProtocolActionResult:
    if not isinstance(resp, dict):
        return ProtocolActionResult(ok=False, error="invalid_protocol_response")
    details = resp.get("details") or []
    for detail in details:
        if isinstance(detail, dict) and detail.get("action") == "traded":
            pid = detail.get("position_id")
            entry_raw = detail.get("entry_price")
            entry_price = float(entry_raw) if entry_raw is not None else None
            return ProtocolActionResult(
                ok=True,
                position_id=int(pid) if pid is not None else None,
                client_ref=str(
                    detail.get("client_ref")
                    or detail.get("api_signal_id")
                    or ""
                ),
                entry_price=entry_price,
            )
    first = details[0] if details and isinstance(details[0], dict) else {}
    error = (
        resp.get("error")
        or first.get("error")
        or first.get("reason")
        or first.get("action")
        or fallback_error
    )
    return ProtocolActionResult(ok=False, error=str(error))


def protocol_ingest_open_result(resp: Any) -> ProtocolActionResult:
    return protocol_ingest_action_result(
        resp,
        fallback_error="protocol_open_not_traded",
    )


def _reconcile_orphan_protocol_closes(
    conn: sqlite3.Connection,
    protocol_client: Any,
    protocol_open_by_symbol: Dict[str, List[Dict[str, Any]]],
) -> None:
    """补记：Protocol 已成交开仓但本地无 row、且交易所已无仓的历史单。"""
    try:
        signals = protocol_client.get_signals(source="moss_quant", limit=300)
    except Exception as e:
        logger.warning("[moss] orphan protocol reconcile skipped: %s", e)
        return
    if not isinstance(signals, list):
        return

    for sig in signals:
        if str(sig.get("status") or "") != "traded":
            continue
        action = str(sig.get("action") or "open").lower()
        if action not in ("open", ""):
            continue
        pid = sig.get("profile_id")
        symbol = str(sig.get("symbol") or "").upper()
        if pid is None or not symbol:
            continue
        pid_i = int(pid)
        if protocol_open_by_symbol.get(symbol):
            continue
        has_open = conn.execute(
            """SELECT 1 FROM moss_signals
               WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')
               LIMIT 1""",
            (pid_i,),
        ).fetchone()
        if has_open:
            continue
        already_settled = conn.execute(
            """SELECT 1 FROM moss_settlements
               WHERE profile_id=? AND symbol=? AND exit_rule LIKE 'external_closed%'
               LIMIT 1""",
            (pid_i, symbol),
        ).fetchone()
        if already_settled:
            continue
        api_id = str(sig.get("api_signal_id") or sig.get("client_ref") or "")
        if api_id:
            settled = conn.execute(
                """SELECT 1 FROM moss_signals
                   WHERE profile_id=? AND meta_json LIKE ?
                   LIMIT 1""",
                (pid_i, f"%{api_id}%"),
            ).fetchone()
            if settled:
                continue
        entry = float(sig.get("entry_price") or 0)
        if entry <= 0:
            continue
        side = str(sig.get("side") or "LONG").upper()
        notional = float(sig.get("notional_usdt") or sig.get("margin_usdt") or 0)
        if notional <= 0:
            lev = float(sig.get("leverage") or 1)
            margin = float(sig.get("margin_usdt") or 1000)
            notional = margin * max(lev, 1.0)
        meta = json.dumps({"protocol_client_ref": api_id, "source": "orphan_reconcile"})
        insert_open_signal_from_protocol(
            conn,
            profile_id=pid_i,
            symbol=symbol,
            side=side,
            entry_price=entry,
            virtual_notional_usdt=notional,
            mark_price=entry,
            regime=str(sig.get("regime") or ""),
            meta_json=meta,
        )
        settle_profile_external_closed(
            conn,
            pid_i,
            exit_price=entry,
            exit_rule="external_closed_reconcile",
        )
        logger.info(
            "[moss] reconciled orphan protocol close profile=%s symbol=%s ref=%s",
            pid_i,
            symbol,
            api_id,
        )


def fetch_open_positions_map(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """profile_id → 当前未平仓纸面单（每 profile 仅最新一条）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.profile_id, s.side, s.symbol, s.entry_price, s.mark_price,
                  s.unrealized_pnl_usdt, s.virtual_notional_usdt
           FROM moss_signals s
           INNER JOIN (
               SELECT profile_id, MAX(id) AS max_id
               FROM moss_signals
               WHERE outcome IS NULL AND side IN ('LONG','SHORT')
               GROUP BY profile_id
           ) t ON s.id = t.max_id"""
    ).fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        pid = int(row["profile_id"])
        entry = float(row["entry_price"] or 0)
        mark = float(row["mark_price"] or 0)
        notional = float(row["virtual_notional_usdt"] or 0)
        side = str(row["side"])
        upnl = float(row["unrealized_pnl_usdt"] or 0)
        if mark > 0 and entry > 0 and notional > 0:
            upnl = pnl_usdt(side, entry, mark, notional)
        out[pid] = {
            "profile_id": pid,
            "side": side,
            "symbol": str(row["symbol"] or "").upper(),
            "entry_price": round(entry, 8),
            "mark_price": round(mark, 8),
            "notional": round(notional, 2),
            "upnl": round(upnl, 4),
        }
    return out


def refresh_open_map_marks(
    open_map: Dict[int, Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None,
    *,
    persist: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """拉最新 K 线收盘价刷新现价、浮盈；可选写回 moss_signals。"""
    from moss_quant.db import get_profile

    now = _utc_now()
    for pid, pos in open_map.items():
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        lev = float(pos.get("leverage") or 10.0)
        if conn is not None:
            prof = get_profile(conn, int(pid))
            if prof:
                lev = float(_effective_params(prof).get("base_leverage", lev))
        try:
            df = load_cached(sym, refresh=True)
            mark = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("[moss] refresh mark %s failed: %s", sym, e)
            continue
        entry = float(pos.get("entry_price") or 0)
        notional = float(pos.get("notional") or 0)
        side = str(pos.get("side") or "")
        if mark <= 0 or entry <= 0 or notional <= 0:
            continue
        upnl = round(pnl_usdt(side, entry, mark, notional), 4)
        pos.update(
            {
                "leverage": round(lev, 2),
                "mark_price": round(mark, 8),
                "upnl": upnl,
                "pnl_pct": round(_margin_pnl_pct(side, entry, mark, lev), 3),
            }
        )
        if persist and conn is not None:
            conn.execute(
                """UPDATE moss_signals SET mark_price=?, unrealized_pnl_usdt=?,
                   updated_at_utc=?
                   WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
                (pos["mark_price"], upnl, now, int(pid)),
            )
    if persist and conn is not None:
        conn.commit()
    return open_map


def append_missing_open_position_details(
    conn: sqlite3.Connection,
    details: List[Dict[str, Any]],
    open_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """上次扫描未包含的持仓（如当次 K 线失败）补一条 hold 明细。"""
    if not open_map:
        return details
    from moss_quant.db import get_profile

    seen = {
        int(d["profile_id"])
        for d in details
        if isinstance(d, dict) and d.get("profile_id") is not None
    }
    out = list(details)
    for pid, pos in open_map.items():
        if pid in seen:
            continue
        prof = get_profile(conn, int(pid))
        if not prof:
            continue
        label = _profile_label(prof)
        out.append(
            _scan_detail(
                label,
                prof,
                {
                    "symbol": pos.get("symbol") or prof.get("symbol"),
                    "action": "hold",
                    "template": prof.get("template"),
                    **_position_fields(
                        side=str(pos["side"]),
                        entry=float(pos["entry_price"]),
                        mark=float(pos["mark_price"]),
                        notional=float(pos["notional"]),
                        upnl=float(pos["upnl"]),
                    ),
                },
            )
        )
    return out


def enrich_scan_details_with_positions(
    details: List[Dict[str, Any]],
    open_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """用库内最新持仓覆盖/补全扫描明细中的开仓价与浮盈。"""
    if not open_map:
        return details
    enriched: List[Dict[str, Any]] = []
    for d in details:
        if not isinstance(d, dict):
            enriched.append(d)
            continue
        row = dict(d)
        pid = row.get("profile_id")
        if pid is None:
            enriched.append(row)
            continue
        pos = open_map.get(int(pid))
        if not pos:
            enriched.append(row)
            continue
        act = str(row.get("action") or "").lower()
        if act not in ("hold", "open", "close"):
            row["action"] = "hold"
        row.update(
            {
                "side": pos.get("side") or row.get("side"),
                "entry_price": pos["entry_price"],
                "mark_price": pos["mark_price"],
                "notional": pos.get("notional", row.get("notional")),
                "upnl": pos["upnl"],
                "leverage": pos.get("leverage", row.get("leverage")),
                "pnl_pct": pos.get("pnl_pct", row.get("pnl_pct")),
            }
        )
        label = str(
            row.get("label")
            or ("p%s:%s" % (pid, pos.get("symbol") or row.get("symbol") or ""))
        )
        row["label"] = label
        row["message"] = format_scan_detail_message(label, row)
        enriched.append(row)
    return enriched


def refresh_live_open_signals(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """刷新所有未平仓纸面单的标记价与浮盈并写库。"""
    return refresh_open_map_marks(
        fetch_open_positions_map(conn), conn, persist=True
    )


def serialize_signal_rows(
    conn: sqlite3.Connection, rows: List[Any]
) -> List[Dict[str, Any]]:
    """将 moss_signals 行转为 API dict，并为持仓补充 leverage / pnl_pct。"""
    from moss_quant.db import get_profile

    out: List[Dict[str, Any]] = []
    prof_cache: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        if d.get("outcome"):
            out.append(d)
            continue
        side = str(d.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            out.append(d)
            continue
        pid = int(d["profile_id"])
        if pid not in prof_cache:
            prof_cache[pid] = get_profile(conn, pid) or {}
        lev = 10.0
        if prof_cache[pid]:
            lev = float(_effective_params(prof_cache[pid]).get("base_leverage", 10))
        entry = float(d.get("entry_price") or 0)
        mark = float(d.get("mark_price") or 0)
        d["leverage"] = round(lev, 2)
        d["pnl_pct"] = round(_margin_pnl_pct(side, entry, mark, lev), 3)
        out.append(d)
    return out


def scan_detail_lines(details: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        label = str(
            d.get("label")
            or ("p%s:%s" % (d.get("profile_id", "?"), d.get("symbol", "")))
        )
        lines.append(format_scan_detail_message(label, d))
    return lines


def compute_current_signal(df: pd.DataFrame, params: DecisionParams) -> int:
    regime = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
    signals = compute_signals(df, params, regime)
    return int(signals.iloc[-1])


def exit_snapshot(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    leverage: float,
    prev_regime: str = "",
    current_regime: str = "",
) -> Dict[str, Any]:
    side_u = side.upper()
    if entry <= 0 or mark <= 0:
        return {"exit_rule": None}

    if side_u in ("LONG", "BUY"):
        pnl_pct = (mark - entry) / entry * leverage
    else:
        pnl_pct = (entry - mark) / entry * leverage

    atr_series = compute_atr(df, 14)
    atr_val = float(atr_series.iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = mark * 0.02
    sl_dist = params.sl_atr_mult * atr_val / entry
    tp_dist = sl_dist * params.tp_rr_ratio
    sl_thresh = -sl_dist * leverage
    tp_thresh = tp_dist * leverage
    sig = compute_current_signal(df, params)

    exit_rule: Optional[str] = None
    if pnl_pct <= sl_thresh:
        exit_rule = "stop_loss"
    elif pnl_pct >= tp_thresh:
        exit_rule = "take_profit"
    elif params.trailing_enabled:
        # 移动止损：用当前 bar 的 high/low 作为极值近似
        bar_high = float(df["high"].iloc[-1])
        bar_low = float(df["low"].iloc[-1])
        trail_dist = params.trailing_distance_atr * atr_val
        if side_u in ("LONG", "BUY"):
            if bar_high > entry * (1 + params.trailing_activation_pct):
                trail_sl = bar_high - trail_dist
                if bar_low <= trail_sl:
                    exit_rule = "trailing_stop"
        else:
            if bar_low < entry * (1 - params.trailing_activation_pct):
                trail_sl = bar_low + trail_dist
                if bar_high >= trail_sl:
                    exit_rule = "trailing_stop"
    if exit_rule is None and params.exit_on_regime_change and current_regime and prev_regime != current_regime:
        exit_rule = "regime_change"
    if exit_rule is None and side_u in ("LONG", "BUY") and sig == -1:
        exit_rule = "signal_reverse"
    elif exit_rule is None and side_u in ("SHORT", "SELL") and sig == 1:
        exit_rule = "signal_reverse"

    return {
        "exit_rule": exit_rule,
        "pnl_pct": round(pnl_pct * 100, 3),
        "sl_thresh_pct": round(sl_thresh * 100, 3),
        "tp_thresh_pct": round(tp_thresh * 100, 3),
        "signal": sig,
        "atr": round(atr_val, 6),
    }


def _atr_from_df(df: pd.DataFrame, mark: float) -> float:
    atr_series = compute_atr(df, 14)
    atr_val = float(atr_series.iloc[-1])
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = mark * 0.02
    return atr_val


def compute_paper_protective_prices(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    """纸面等价绝对 SL/TP（与 exit_snapshot / 开仓逻辑一致）。"""
    side_u = side.upper()
    if entry <= 0 or mark <= 0:
        return {"sl_price": None, "tp_price": None, "atr_sl": None, "trailing_sl": None}

    atr_val = _atr_from_df(df, mark)
    sl_dist = params.sl_atr_mult * atr_val
    tp_dist = sl_dist * params.tp_rr_ratio
    bar_high = float(df["high"].iloc[-1])
    bar_low = float(df["low"].iloc[-1])
    trailing_sl: Optional[float] = None

    if side_u in ("LONG", "BUY"):
        atr_sl = entry - sl_dist
        tp_price = entry + tp_dist
        if params.trailing_enabled and bar_high > entry * (1 + params.trailing_activation_pct):
            trail_dist = params.trailing_distance_atr * atr_val
            trailing_sl = bar_high - trail_dist
            sl_price = max(atr_sl, trailing_sl)
        else:
            sl_price = atr_sl
    else:
        atr_sl = entry + sl_dist
        tp_price = entry - tp_dist
        if params.trailing_enabled and bar_low < entry * (1 - params.trailing_activation_pct):
            trail_dist = params.trailing_distance_atr * atr_val
            trailing_sl = bar_low + trail_dist
            sl_price = min(atr_sl, trailing_sl)
        else:
            sl_price = atr_sl

    return {
        "sl_price": round(float(sl_price), 8),
        "tp_price": round(float(tp_price), 8),
        "atr_sl": round(float(atr_sl), 8),
        "trailing_sl": round(float(trailing_sl), 8) if trailing_sl is not None else None,
    }


def _protective_price_changed(
    old: Optional[float],
    new: Optional[float],
    *,
    min_rel: float = 0.0002,
) -> bool:
    if new is None:
        return False
    if old is None:
        return True
    if old <= 0:
        return old != new
    return abs(new - old) / old >= min_rel


def sync_live_protective_orders(
    *,
    sender: Any,
    symbol: str,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    profile_id: int,
    meta: Dict[str, Any],
    has_live_position: bool,
    label: str = "",
) -> Dict[str, Any]:
    """每轮 hold 将纸面等价 SL/TP 同步到 Protocol/Binance。"""
    if not sender or not has_live_position:
        return meta

    prices = compute_paper_protective_prices(
        side=side,
        entry=entry,
        mark=mark,
        params=params,
        df=df,
    )
    sl_price = prices.get("sl_price")
    tp_price = prices.get("tp_price")
    if sl_price is None or tp_price is None:
        return meta

    last_sl = meta.get("last_synced_sl")
    last_tp = meta.get("last_synced_tp")
    sl_changed = _protective_price_changed(
        float(last_sl) if last_sl is not None else None,
        float(sl_price),
    )
    tp_changed = _protective_price_changed(
        float(last_tp) if last_tp is not None else None,
        float(tp_price),
    )
    meta_changed = False

    if sl_changed:
        try:
            resp = sender.send_update_sl(
                symbol=symbol,
                side=side,
                new_sl_price=round(float(sl_price), 6),
                profile_id=profile_id,
            )
            if _protocol_ingest_traded(resp):
                meta["last_synced_sl"] = round(float(sl_price), 6)
                meta_changed = True
                logger.info(
                    "[moss] %s sync SL %.6g (atr=%.6g trail=%s)",
                    label,
                    sl_price,
                    prices.get("atr_sl"),
                    prices.get("trailing_sl"),
                )
            else:
                logger.warning("[moss] %s sync SL skipped: %s", label, resp)
        except Exception as exc:
            logger.warning("[moss] %s sync SL failed: %s", label, exc)

    if tp_changed:
        try:
            resp = sender.send_update_tp(
                symbol=symbol,
                side=side,
                new_tp_price=round(float(tp_price), 6),
                profile_id=profile_id,
            )
            if _protocol_ingest_traded(resp):
                meta["last_synced_tp"] = round(float(tp_price), 6)
                meta_changed = True
                logger.info("[moss] %s sync TP %.6g", label, tp_price)
            else:
                logger.warning("[moss] %s sync TP skipped: %s", label, resp)
        except Exception as exc:
            logger.warning("[moss] %s sync TP failed: %s", label, exc)

    if meta_changed:
        meta["last_synced_at_utc"] = _utc_now()
    return meta


def check_exit(
    *,
    side: str,
    entry: float,
    mark: float,
    params: DecisionParams,
    df: pd.DataFrame,
    leverage: float,
) -> Optional[str]:
    return exit_snapshot(
        side=side,
        entry=entry,
        mark=mark,
        params=params,
        df=df,
        leverage=leverage,
    ).get("exit_rule")


def entry_snapshot(
    df: pd.DataFrame,
    params: DecisionParams,
    regime_s: pd.Series,
) -> Dict[str, Any]:
    sig = compute_current_signal(df, params)
    composite = compute_last_composite(df, params, regime_s)
    th = params.entry_threshold
    regime_label = str(regime_s.iloc[-1]) if len(regime_s) else "SIDEWAYS"
    if sig == 1:
        reason = "signal_long"
    elif sig == -1:
        reason = "signal_short"
    elif abs(composite) <= th:
        reason = "composite_below_threshold"
    else:
        reason = "no_discrete_signal"
    return {
        "signal": int(sig),
        "composite": round(float(composite), 4),
        "entry_threshold": round(float(_py_scalar(th)), 4),
        "regime": regime_label,
        "reason": reason,
        "bars": int(len(df)),
    }


def pnl_usdt(side: str, entry: float, exit_px: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    if side.upper() in ("LONG", "BUY"):
        return notional * (exit_px - entry) / entry
    return notional * (entry - exit_px) / entry


def _effective_params(profile: Dict[str, Any]) -> dict:
    base = dict(profile["initial_params"])
    tactical = profile.get("tactical_params") or {}
    base.update(tactical)
    sym = str(profile.get("symbol") or "").upper()
    return cap_leverage_for_symbol(resolve_params_dict(base), sym)


def _position_margin(notional: float, leverage: float) -> float:
    if leverage <= 0 or notional <= 0:
        return 0.0
    return notional / leverage


def _free_margin(
    wallet_balance: float,
    *,
    side: str,
    entry: float,
    mark: float,
    notional: float,
    leverage: float,
) -> float:
    """与原工程 realtime_incremental / live_runner 一致：权益 − 已占用保证金。"""
    margin = _position_margin(notional, leverage)
    if margin <= 0:
        return max(0.0, wallet_balance)
    upnl = pnl_usdt(side, entry, mark, notional)
    equity = wallet_balance + upnl
    return max(0.0, equity - margin)


def _open_notional_from_free_margin(free_margin: float, params: dict) -> float:
    """margin = min(free*risk, free*max_pct, free)；notional = margin * lev（factory 同款）。"""
    if free_margin <= 0:
        return 0.0
    lev = min(float(params.get("base_leverage", 10)), float(params.get("max_leverage", 10)))
    risk = float(params.get("risk_per_trade", 0.1))
    max_pct = float(params.get("max_position_pct", 0.5))
    margin = min(free_margin * risk, free_margin * max_pct, free_margin)
    return max(margin * lev, 10.0)


def _notional_for_profile(
    conn: sqlite3.Connection,
    profile_id: int,
    params: dict,
    *,
    open_row: Optional[sqlite3.Row] = None,
    mark: float = 0.0,
    leverage: float = 10.0,
) -> float:
    wallet_balance = profile_wallet_balance(conn, profile_id)
    if open_row is not None and mark > 0:
        free = _free_margin(
            wallet_balance,
            side=str(open_row["side"]),
            entry=float(open_row["entry_price"] or 0),
            mark=mark,
            notional=float(open_row["virtual_notional_usdt"] or 0),
            leverage=leverage,
        )
    else:
        free = max(0.0, wallet_balance)
    return _open_notional_from_free_margin(free, params)


def live_notional_from_account(
    *,
    wallet_balance_usdt: float,
    enabled_profile_count: int,
    leverage: float,
    params: Dict[str, Any],
) -> float:
    wallet = float(wallet_balance_usdt)
    leverage = float(leverage)
    if not math.isfinite(wallet) or wallet <= 0:
        raise ValueError("wallet_balance_usdt must be positive")
    if enabled_profile_count <= 0:
        raise ValueError("enabled_profile_count must be positive")
    if not math.isfinite(leverage) or leverage <= 0:
        raise ValueError("leverage must be positive")
    risk = float(params.get("risk_per_trade", 0.1))
    max_pct = float(params.get("max_position_pct", 0.5))
    if not math.isfinite(risk) or risk <= 0:
        raise ValueError("risk_per_trade must be positive")
    if not math.isfinite(max_pct) or max_pct <= 0:
        raise ValueError("max_position_pct must be positive")
    per_robot_equity = wallet / int(enabled_profile_count)
    margin = min(per_robot_equity * risk, per_robot_equity * max_pct)
    return round(max(margin * leverage, 0.0), 2)


def run_paper_scan(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    profiles = list_profiles_for_paper_scan(conn)
    stats: Dict[str, Any] = {
        "profiles_scanned": len(profiles),
        "opens": 0,
        "closes": 0,
        "details": [],
    }
    now = _utc_now()

    logger.info(
        "[moss] paper scan start enabled=%s verbose=%s",
        len(profiles),
        _verbose(),
    )
    if not profiles:
        logger.info("[moss] paper scan skip: no enabled profiles")
        conn.execute(
            """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
               VALUES (?,?,?,?,?)""",
            (now, 0, 0, 0, "[]"),
        )
        conn.commit()
        return stats

    sender = _get_sender()
    live_account_summary: Optional[Dict[str, Any]] = None
    protocol_client: Any = None
    protocol_open_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    protocol_truth_loaded = False
    live_opens_allowed = True
    enabled_profile_count = sum(1 for p in profiles if bool(p.get("enabled")))

    if sender:
        try:
            from moss_quant.protocol_client import ProtocolClient

            protocol_client = ProtocolClient.from_env()
            live_account_summary = protocol_client.get_account_summary()
            protocol_open_by_symbol = protocol_open_positions_by_symbol(
                protocol_client.get_moss_positions(status="open", limit=500)
            )
            protocol_truth_loaded = True
            _reconcile_orphan_protocol_closes(conn, protocol_client, protocol_open_by_symbol)
        except Exception as e:
            logger.error("[moss] protocol truth load failed: %s", e)
            stats["protocol_error"] = str(e)
            live_opens_allowed = False

    for profile in profiles:
        pid = int(profile["id"])
        symbol = str(profile["symbol"]).upper()
        label = _profile_label(profile)
        params_d = _effective_params(profile)
        params = DecisionParams.from_dict(params_d)
        lev = float(params_d.get("base_leverage", 10))
        real_positions = protocol_open_by_symbol.get(symbol, [])
        protocol_pos = real_positions[0] if real_positions else None

        try:
            # HL：缓存过期时 load_cached 会自动 ccxt 更新，无需每次 refresh=True
            df = load_cached(symbol, refresh=False)
        except Exception as e:
            logger.warning("[moss] %s kline failed: %s", label, e)
            stats["details"].append(
                _scan_detail(label, profile, {"symbol": symbol, "action": "error", "error": str(e)})
            )
            continue

        mark = float(df["close"].iloc[-1])
        regime_s = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
        regime_label = str(regime_s.iloc[-1]) if len(regime_s) else "SIDEWAYS"

        row = conn.execute(
            """SELECT * FROM moss_signals
               WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')
               LIMIT 1""",
            (pid,),
        ).fetchone()

        if sender and protocol_truth_loaded and protocol_pos and not row:
            entry_px = float(protocol_pos.get("entry_price") or mark)
            notional_pb = _protocol_position_notional(protocol_pos)
            side_pb = str(protocol_pos.get("side") or "LONG").upper()
            mark_pb = _protocol_position_mark_price(protocol_pos) or mark
            insert_open_signal_from_protocol(
                conn,
                profile_id=pid,
                symbol=symbol,
                side=side_pb,
                entry_price=entry_px,
                virtual_notional_usdt=notional_pb,
                mark_price=mark_pb,
                regime=regime_label,
                meta_json=json.dumps({"source": "protocol_backfill"}),
            )
            row = conn.execute(
                """SELECT * FROM moss_signals
                   WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')
                   LIMIT 1""",
                (pid,),
            ).fetchone()
            logger.info("[moss] %s backfilled local signal from protocol_open", label)

        if row:
            if sender and protocol_truth_loaded and not real_positions:
                settled = settle_profile_external_closed(
                    conn, pid, exit_price=mark, exit_rule="external_closed"
                )
                stats["closes"] += int(settled.get("closed") or 0)
                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "close",
                            "side": str(row["side"]),
                            "rule": "external_closed",
                            "pnl": float(settled.get("pnl_usdt") or 0),
                        },
                    )
                )
                logger.info(
                    "[moss] %s CLOSE external_closed pnl=%s",
                    label,
                    settled.get("pnl_usdt"),
                )
                continue

            side = str((protocol_pos or {}).get("side") or row["side"])
            entry = float((protocol_pos or {}).get("entry_price") or row["entry_price"] or 0)
            if protocol_pos:
                notional = _protocol_position_notional(protocol_pos)
                lev = float(protocol_pos.get("leverage") or lev)
            else:
                notional = float(row["virtual_notional_usdt"] or 0)
            snap = exit_snapshot(
                side=side,
                entry=entry,
                mark=mark,
                params=params,
                df=df,
                leverage=lev,
                prev_regime=str(row["regime"] or ""),
                current_regime=regime_label,
            )
            exit_rule = snap.get("exit_rule")
            if exit_rule:
                if sender and real_positions:
                    close_resp = sender.send_close(
                        symbol=symbol,
                        side=side,
                        exit_rule=exit_rule,
                        close_price=mark,
                        profile_id=pid,
                    )
                    close_result = protocol_ingest_action_result(
                        close_resp,
                        fallback_error="protocol_close_not_traded",
                    )
                    if not close_result.ok:
                        stats["details"].append(
                            _scan_detail(
                                label,
                                profile,
                                {
                                    "symbol": symbol,
                                    "action": "error",
                                    "error": f"protocol_close_failed: {close_result.error}",
                                },
                            )
                        )
                        logger.error("[moss] %s protocol close failed: %s", label, close_result.error)
                        continue

                pnl = pnl_usdt(side, entry, mark, notional)
                outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")

                if sender and real_positions:
                    try:
                        close_resp = sender.send_close(
                            symbol=symbol,
                            side=side,
                            exit_rule=exit_rule,
                            close_price=mark,
                            profile_id=pid,
                        )
                        if not _protocol_ingest_traded(close_resp):
                            err = (
                                (close_resp or {}).get("error")
                                or "protocol_close_not_traded"
                            )
                            logger.error(
                                "[moss] %s protocol close failed: %s",
                                label,
                                err,
                            )
                            stats["details"].append(
                                _scan_detail(
                                    label,
                                    profile,
                                    {
                                        "symbol": symbol,
                                        "action": "error",
                                        "error": f"protocol_close_failed: {err}",
                                        "rule": exit_rule,
                                    },
                                )
                            )
                            continue
                    except Exception as exc:
                        logger.error(
                            "[moss] %s protocol close error: %s",
                            label,
                            exc,
                        )
                        stats["details"].append(
                            _scan_detail(
                                label,
                                profile,
                                {
                                    "symbol": symbol,
                                    "action": "error",
                                    "error": f"protocol_close_failed: {exc}",
                                    "rule": exit_rule,
                                },
                            )
                        )
                        continue

                conn.execute(
                    """UPDATE moss_signals SET outcome=?, outcome_at_utc=?, exit_price=?,
                       pnl_usdt=?, exit_rule=?, updated_at_utc=?, unrealized_pnl_usdt=0
                       WHERE id=?""",
                    (outcome, now, mark, pnl, exit_rule, now, row["id"]),
                )
                conn.execute(
                    """INSERT INTO moss_settlements(
                        settled_at_utc, signal_id, profile_id, symbol, side, outcome,
                        entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        now,
                        row["id"],
                        pid,
                        symbol,
                        side,
                        outcome,
                        entry,
                        mark,
                        pnl,
                        notional,
                        exit_rule,
                    ),
                )
                sync_moss_wallet_from_settlements(conn)
                profile_wallet_balance(conn, pid)
                stats["closes"] += 1

                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "close",
                            "side": side,
                            "rule": exit_rule,
                            "pnl": round(pnl, 4),
                        },
                    )
                )
                logger.info(
                    "[moss] %s CLOSE %s %s pnl=%.4fU mark=%.6g entry=%.6g",
                    label,
                    side,
                    exit_rule,
                    pnl,
                    mark,
                    entry,
                )
                _vlog("%s exit_diag %s", label, snap)
            else:
                upnl = pnl_usdt(side, entry, mark, notional)
                detail_mark = mark
                detail_upnl = upnl
                if protocol_pos:
                    detail_mark = _protocol_position_mark_price(protocol_pos) or mark
                    if _protocol_position_has_unrealized_pnl(protocol_pos):
                        detail_upnl = _protocol_position_unrealized_pnl(protocol_pos)
                conn.execute(
                    """UPDATE moss_signals SET mark_price=?, unrealized_pnl_usdt=?,
                       updated_at_utc=? WHERE id=?""",
                    (detail_mark, detail_upnl, now, row["id"]),
                )
                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "hold",
                            "template": profile.get("template"),
                            "sl_thresh_pct": snap.get("sl_thresh_pct"),
                            "tp_thresh_pct": snap.get("tp_thresh_pct"),
                            "signal": snap.get("signal"),
                            **_position_fields(
                                side=side,
                                entry=entry,
                                mark=detail_mark,
                                notional=notional,
                                upnl=detail_upnl,
                                leverage=lev,
                            ),
                        },
                    )
                )
                logger.info(
                    "[moss] %s HOLD %s upnl=%.4fU pnl%%=%s sl<=%s%% tp>=%s%% sig=%s",
                    label,
                    side,
                    upnl,
                    snap.get("pnl_pct"),
                    snap.get("sl_thresh_pct"),
                    snap.get("tp_thresh_pct"),
                    snap.get("signal"),
                )
                _vlog("%s hold_diag %s", label, snap)

                # 滚仓检测（与原工程：浮盈×reinvest_pct 作新保证金，受 free_margin 约束）
                if params.rolling_enabled:
                    try:
                        meta = json.loads(row["meta_json"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                    roll_count = int(meta.get("rolling_count") or 0)
                    pnl_pct_display = float(snap.get("pnl_pct") or 0)
                    unrealized_ratio = pnl_pct_display / 100.0
                    if (
                        roll_count < params.rolling_max_times
                        and unrealized_ratio >= params.rolling_trigger_pct
                    ):
                        margin = _position_margin(notional, lev)
                        float_profit = margin * unrealized_ratio
                        new_margin = float_profit * params.rolling_reinvest_pct
                        wallet_balance = profile_wallet_balance(conn, pid, sync=False)
                        free_margin = _free_margin(
                            wallet_balance,
                            side=side,
                            entry=entry,
                            mark=mark,
                            notional=notional,
                            leverage=lev,
                        )
                        if sender and not can_send_live_open(sender, live_opens_allowed):
                            logger.warning(
                                "[moss] %s rolling skipped: protocol truth unavailable",
                                label,
                            )
                        elif new_margin > 0 and free_margin >= new_margin:
                            add_notional = new_margin * lev
                            rolling_ok = True
                            if sender:
                                prot_prices = compute_paper_protective_prices(
                                    side=side,
                                    entry=mark,
                                    mark=mark,
                                    params=params,
                                    df=df,
                                )
                                sender.send_rolling(
                                    symbol=symbol,
                                    side=side,
                                    margin_usdt=round(new_margin, 6),
                                    leverage=round(lev, 6),
                                    profile_id=pid,
                                    play=profile.get("template", ""),
                                    sl_price=round(prot_prices["sl_price"] or mark, 6),
                                    tp_price=round(prot_prices["tp_price"] or mark, 6),
                                    rolling_count=roll_count + 1,
                                )
                                rolling_result = protocol_ingest_action_result(
                                    rolling_resp,
                                    fallback_error="protocol_rolling_not_traded",
                                )
                                rolling_ok = rolling_result.ok
                                if not rolling_ok:
                                    logger.error(
                                        "[moss] %s protocol rolling failed: %s",
                                        label,
                                        rolling_result.error,
                                    )
                            if rolling_ok:
                                notional = round(notional + add_notional, 2)
                                conn.execute(
                                    """UPDATE moss_signals SET virtual_notional_usdt=?, updated_at_utc=?
                                       WHERE id=?""",
                                    (notional, now, row["id"]),
                                )
                                if params.rolling_move_stop:
                                    move_stop_ok = True
                                    if sender and real_positions:
                                        move_stop_resp = sender.send_update_sl(
                                            symbol=symbol,
                                            side=side,
                                            new_sl_price=round(entry, 6),
                                            profile_id=pid,
                                        )
                                        move_stop_result = protocol_ingest_action_result(
                                            move_stop_resp,
                                            fallback_error="protocol_move_stop_not_traded",
                                        )
                                        move_stop_ok = move_stop_result.ok
                                        if not move_stop_ok:
                                            logger.error(
                                                "[moss] %s protocol move_stop failed: %s",
                                                label,
                                                move_stop_result.error,
                                            )
                                    if move_stop_ok:
                                        meta["stop_moved_to_entry"] = True
                                meta["rolling_count"] = roll_count + 1
                                conn.execute(
                                    "UPDATE moss_signals SET meta_json=? WHERE id=?",
                                    (json.dumps(meta), row["id"]),
                                )
                                logger.info(
                                    "[moss] %s ROLLING #%s add_notional=%.2fU total=%.2fU pnl%%=%.2f",
                                    label,
                                    roll_count + 1,
                                    add_notional,
                                    notional,
                                    pnl_pct_display,
                                )

                if sender and real_positions:
                    try:
                        hold_meta = json.loads(row["meta_json"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        hold_meta = {}
                    hold_meta = sync_live_protective_orders(
                        sender=sender,
                        symbol=symbol,
                        side=side,
                        entry=entry,
                        mark=mark,
                        params=params,
                        df=df,
                        profile_id=pid,
                        meta=hold_meta,
                        has_live_position=True,
                        label=label,
                    )
                    if hold_meta.get("last_synced_at_utc"):
                        conn.execute(
                            "UPDATE moss_signals SET meta_json=? WHERE id=?",
                            (json.dumps(hold_meta), row["id"]),
                        )
            continue

        ent = entry_snapshot(df, params, regime_s)
        if ent["signal"] == 0:
            stats["details"].append(
                _scan_detail(
                    label,
                    profile,
                    {
                        "symbol": symbol,
                        "action": "wait",
                        "composite": ent["composite"],
                        "entry_threshold": ent["entry_threshold"],
                        "reason": ent["reason"],
                        "regime": regime_label,
                    },
                )
            )
            logger.info(
                "[moss] %s WAIT composite=%s thresh=±%s regime=%s reason=%s mark=%.6g",
                label,
                ent["composite"],
                ent["entry_threshold"],
                regime_label,
                ent["reason"],
                mark,
            )
            continue

        side = "LONG" if ent["signal"] == 1 else "SHORT"
        composite = ent["composite"]
        if sender:
            try:
                if not live_opens_allowed:
                    raise ValueError(str(stats.get("protocol_error") or "protocol_unavailable"))
                if live_account_summary is None:
                    raise ValueError("protocol_account_unavailable")
                notional = live_notional_from_account(
                    wallet_balance_usdt=live_account_summary["wallet_balance_usdt"],
                    enabled_profile_count=enabled_profile_count,
                    leverage=lev,
                    params=params_d,
                )
            except (KeyError, TypeError, ValueError) as e:
                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "error",
                            "error": f"live_notional_unavailable: {e}",
                        },
                    )
                )
                logger.error("[moss] %s live open sizing failed: %s", label, e)
                continue
        else:
            notional = _notional_for_profile(conn, pid, params_d, leverage=lev)

        if notional <= 0:
            stats["details"].append(
                _scan_detail(
                    label,
                    profile,
                    {
                        "symbol": symbol,
                        "action": "wait",
                        "reason": "insufficient_free_margin",
                        "regime": regime_label,
                    },
                )
            )
            logger.info("[moss] %s SKIP OPEN: insufficient free margin", label)
            continue

        if sender:
            prot_prices = compute_paper_protective_prices(
                side=side,
                entry=mark,
                mark=mark,
                params=params,
                df=df,
            )
            sl_price = prot_prices["sl_price"]
            tp_price = prot_prices["tp_price"]
            open_resp = sender.send_open(
                symbol=symbol,
                side=side,
                entry_price=mark,
                sl_price=round(sl_price or mark, 6),
                tp_price=round(tp_price or mark, 6) if tp_price else None,
                margin_usdt=round(notional / lev, 6),
                leverage=round(lev, 6),
                profile_id=pid,
                play=profile.get("template", ""),
                composite=composite,
                regime=regime_label,
            )
            open_result = protocol_ingest_open_result(open_resp)
            if not open_result.ok:
                stats["details"].append(
                    _scan_detail(
                        label,
                        profile,
                        {
                            "symbol": symbol,
                            "action": "error",
                            "error": f"protocol_open_failed: {open_result.error}",
                            "template": profile.get("template"),
                            "composite": composite,
                            "regime": regime_label,
                        },
                    )
                )
                logger.error("[moss] %s protocol open failed: %s", label, open_result.error)
                continue

            entry_px = open_result.entry_price or mark
            if protocol_client:
                try:
                    fresh = protocol_open_positions_by_symbol(
                        protocol_client.get_moss_positions(status="open")
                    )
                    if fresh.get(symbol):
                        protocol_open_by_symbol[symbol] = fresh[symbol]
                        entry_px = float(
                            fresh[symbol][0].get("entry_price") or entry_px
                        )
                except Exception as exc:
                    logger.warning(
                        "[moss] %s refresh entry after open failed: %s",
                        label,
                        exc,
                    )
            elif protocol_open_by_symbol.get(symbol):
                proto_after = protocol_open_by_symbol[symbol][0]
                entry_px = float(proto_after.get("entry_price") or entry_px)
            meta = json.dumps(
                {
                    "protocol_client_ref": open_result.client_ref,
                    "protocol_position_id": open_result.position_id,
                    "last_synced_sl": round(float(sl_price or mark), 6),
                    "last_synced_tp": round(float(tp_price or mark), 6),
                    "last_synced_at_utc": now,
                }
            )
        else:
            open_result = None
            meta = None
            entry_px = mark

        conn.execute(
            """INSERT INTO moss_signals(
                profile_id, recorded_at_utc, side, symbol, entry_price,
                virtual_notional_usdt, mark_price, composite, regime,
                unrealized_pnl_usdt, meta_json, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,0,?,?)""",
            (
                pid,
                now,
                side,
                symbol,
                entry_px,
                notional,
                entry_px,
                composite,
                regime_label,
                meta,
                now,
            ),
        )
        stats["opens"] += 1

        stats["details"].append(
            _scan_detail(
                label,
                profile,
                {
                    "symbol": symbol,
                    "action": "open",
                    "template": profile.get("template"),
                    "composite": composite,
                    "regime": regime_label,
                    "position_id": (
                        open_result.position_id if open_result else None
                    ),
                    **_position_fields(
                        side=side,
                        entry=entry_px,
                        mark=entry_px,
                        notional=notional,
                        upnl=0.0,
                        leverage=lev,
                    ),
                },
            )
        )
        logger.info(
            "[moss] %s OPEN %s notional=%.2fU composite=%s regime=%s mark=%.6g",
            label,
            side,
            notional,
            composite,
            regime_label,
            mark,
        )

    detail_json = json.dumps(stats["details"], ensure_ascii=False, default=_py_scalar)
    conn.execute(
        """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
           VALUES (?,?,?,?,?)""",
        (now, stats["profiles_scanned"], stats["opens"], stats["closes"], detail_json),
    )
    conn.commit()

    logger.info(
        "[moss] paper scan done profiles=%s opens=%s closes=%s",
        stats["profiles_scanned"],
        stats["opens"],
        stats["closes"],
    )
    if _verbose() and stats["details"]:
        logger.info("[moss] paper details %s", detail_json)

    return stats
