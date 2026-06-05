"""Moss2 纸面钱包、浮盈刷新（对齐 moss_quant.paper_scanner 纸面语义）。"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional

from moss2 import config as cfg
from moss2.db import _utc_now, get_profile
from moss2.kline_loader import load_market_df
from moss2.params import merge_profile_params

logger = logging.getLogger(__name__)


def paper_source_of_truth() -> bool:
    return bool(cfg.MOSS2_PAPER_SOURCE_OF_TRUTH)


def paper_trading_leverage() -> float:
    return 1.0


def pnl_usdt(side: str, entry: float, mark: float, notional: float) -> float:
    if entry <= 0 or notional <= 0:
        return 0.0
    side_u = str(side).upper()
    if side_u in ("LONG", "BUY"):
        return notional * (mark - entry) / entry
    return notional * (entry - mark) / entry


def margin_pnl_pct(side: str, entry: float, mark: float, leverage: float) -> float:
    if entry <= 0 or mark <= 0 or leverage <= 0:
        return 0.0
    side_u = str(side).upper()
    if side_u in ("LONG", "BUY"):
        return (mark - entry) / entry * leverage * 100.0
    return (entry - mark) / entry * leverage * 100.0


def _leverage_for_profile(profile: Optional[dict]) -> float:
    if not profile:
        return paper_trading_leverage()
    params = merge_profile_params(profile)
    lev = min(
        float(params.get("base_leverage", 10)),
        float(params.get("max_leverage", 10)),
    )
    return paper_trading_leverage() if paper_source_of_truth() else lev


def fetch_open_positions_map(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.id, s.profile_id, s.side, s.symbol, s.entry_price, s.mark_price,
                  s.unrealized_pnl_usdt, s.virtual_notional_usdt, s.meta_json, p.variant
           FROM moss2_signals s
           JOIN moss2_profiles p ON p.id = s.profile_id
           WHERE s.outcome IS NULL AND s.side IN ('LONG','SHORT')"""
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
        prof = get_profile(conn, pid)
        lev = _leverage_for_profile(prof)
        out[pid] = {
            "signal_id": int(row["id"]),
            "profile_id": pid,
            "side": side,
            "symbol": str(row["symbol"] or "").upper(),
            "variant": str(row["variant"] or cfg.MOSS2_OPS_VARIANT),
            "entry_price": round(entry, 8),
            "mark_price": round(mark, 8),
            "notional": round(notional, 2),
            "upnl": round(upnl, 4),
            "unrealized_pnl_usdt": round(upnl, 4),
            "leverage": round(lev, 2),
            "meta_json": row["meta_json"] if "meta_json" in row.keys() else None,
        }
    return out


def refresh_open_map_marks(
    open_map: Dict[int, Dict[str, Any]],
    conn: sqlite3.Connection,
    *,
    persist: bool = True,
) -> Dict[int, Dict[str, Any]]:
    now = _utc_now()
    for pid, pos in open_map.items():
        sym = str(pos.get("symbol") or "").upper()
        variant = str(pos.get("variant") or cfg.MOSS2_OPS_VARIANT)
        if not sym:
            continue
        prof = get_profile(conn, int(pid))
        lev = _leverage_for_profile(prof)
        try:
            df = load_market_df(sym, variant, limit=cfg.MOSS2_KLINE_LIMIT)
            mark = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("[moss2] refresh mark %s failed: %s", sym, e)
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
                "unrealized_pnl_usdt": upnl,
                "pnl_pct": round(margin_pnl_pct(side, entry, mark, lev), 3),
            }
        )
        from moss2.exit_levels import enrich_position_exit_levels

        enrich_position_exit_levels(
            conn,
            pos,
            df=df,
            at_utc=now,
            persist_meta=persist,
            signal_id=int(pos.get("signal_id") or 0) or None,
            meta_json=pos.get("meta_json"),
        )
        if persist:
            conn.execute(
                """UPDATE moss2_signals SET mark_price=?, unrealized_pnl_usdt=?,
                   updated_at_utc=?
                   WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
                (pos["mark_price"], upnl, now, int(pid)),
            )
    if persist:
        conn.commit()
    return open_map


def refresh_live_open_signals(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    return refresh_open_map_marks(fetch_open_positions_map(conn), conn, persist=True)


def serialize_signal_rows(
    conn: sqlite3.Connection, rows: list
) -> list:
    out = []
    prof_cache: Dict[int, dict] = {}
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
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
        lev = _leverage_for_profile(prof_cache[pid])
        entry = float(d.get("entry_price") or 0)
        mark = float(d.get("mark_price") or 0)
        notional = float(d.get("virtual_notional_usdt") or 0)
        upnl = float(d.get("unrealized_pnl_usdt") or 0)
        if mark > 0 and entry > 0 and notional > 0:
            upnl = pnl_usdt(side, entry, mark, notional)
        d["unrealized_pnl_usdt"] = round(upnl, 4)
        d["leverage"] = round(lev, 2)
        d["pnl_pct"] = round(margin_pnl_pct(side, entry, mark, lev), 3)
        from moss2.exit_levels import enrich_position_exit_levels, parse_exit_levels_from_meta

        levels = parse_exit_levels_from_meta(d.get("meta_json"))
        if levels.get("stop_loss") is None and entry > 0:
            prof = prof_cache[pid]
            tmp = {
                "profile_id": pid,
                "symbol": str(prof.get("symbol") or d.get("symbol") or "").upper(),
                "variant": str(prof.get("variant") or cfg.MOSS2_OPS_VARIANT),
                "side": side,
                "entry_price": entry,
                "mark_price": mark if mark > 0 else entry,
            }
            enrich_position_exit_levels(conn, tmp, persist_meta=False)
            for key in ("stop_loss", "take_profit", "atr14"):
                if tmp.get(key) is not None:
                    levels[key] = tmp[key]
        d.update(levels)
        out.append(d)
    return out
