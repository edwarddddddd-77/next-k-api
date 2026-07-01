"""King Keltner 看板数据（summary / trades）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

def _db_conn():
    import accumulation_radar

    return accumulation_radar.init_db()
from orb.core.kline_cache import norm_symbol
from orb.core.paper import _session_date_now
from orb.core.session_today import build_session_today
from orb.kk.config import KKConfig
from orb.kk.db import load_state_json, load_wallet, migrate_kk_tables
from orb.kk.equity import symbol_equity_usdt
from orb.kk.live_exec import live_enabled


def _bot_label(symbol: str) -> str:
    sym = norm_symbol(symbol)
    return sym[:-4] if sym.endswith("USDT") else sym


def _side_name(side_val: Any) -> str:
    if str(side_val).upper() in ("LONG", "SHORT"):
        return str(side_val).upper()
    n = int(side_val or 0)
    if n > 0:
        return "LONG"
    if n < 0:
        return "SHORT"
    return ""


def _pos_from_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = payload.get("ctx") if isinstance(payload.get("ctx"), dict) else payload
    pos = ctx.get("pos") if isinstance(ctx.get("pos"), dict) else {}
    return pos


def build_kk_summary(*, kk: Optional[KKConfig] = None) -> Dict[str, Any]:
    kk = kk or KKConfig.from_env()
    orb_cfg = kk.orb_session_cfg()
    session_date = _session_date_now(orb_cfg)
    today = build_session_today(orb_cfg)
    symbols = kk.symbol_list()
    base_eq = float(kk.equity_usdt or 14.0)

    robots: List[Dict[str, Any]] = []
    open_positions = 0
    sum_pnl = 0.0
    settled_trades = 0

    conn = _db_conn()
    try:
        cur = conn.cursor()
        migrate_kk_tables(cur)

        cur.execute(
            """
            SELECT symbol, SUM(COALESCE(pnl_usdt, 0)) AS pnl, COUNT(*) AS n
            FROM kk_trades
            WHERE event = 'close' AND pnl_usdt IS NOT NULL
            GROUP BY symbol
            """
        )
        pnl_by_sym = {
            norm_symbol(str(r[0])): (float(r[1] or 0.0), int(r[2] or 0))
            for r in cur.fetchall()
        }

        for sym in symbols:
            sym = norm_symbol(sym)
            wallet = symbol_equity_usdt(kk, sym, cur=cur)
            state = load_state_json(cur, sym, session_date)
            pos = _pos_from_state(state)
            side_i = int(pos.get("side") or 0)
            open_side = _side_name(side_i)
            if open_side:
                open_positions += 1
            sym_pnl, sym_settled = pnl_by_sym.get(sym, (0.0, 0))
            settled_trades += sym_settled
            sum_pnl += sym_pnl
            robots.append(
                {
                    "symbol": sym,
                    "label": _bot_label(sym),
                    "wallet_balance_usdt": round(wallet, 4),
                    "realized_pnl_usdt": round(sym_pnl, 4),
                    "open_side": open_side,
                    "entry": float(pos.get("entry") or 0.0) or None,
                    "notional_usdt": float(pos.get("notional") or 0.0) or None,
                    "enabled": True,
                }
            )
    finally:
        conn.close()

    out: Dict[str, Any] = {
        "ok": True,
        "lane": kk.lane,
        "engine": kk.engine,
        "session_date": session_date,
        "today": today,
        "robot_count": len(symbols),
        "robot_equity_usdt": base_eq,
        "compound": bool(kk.compound),
        "live_enabled": bool(kk.live_enabled),
        "live_active": live_enabled(kk),
        "robot_bound": True,
        "open_positions": open_positions,
        "settled_trades": settled_trades,
        "sum_pnl_usdt": round(sum_pnl, 4),
        "robots": robots,
        "max_open_positions": int(kk.max_open_positions or 0),
        "eod_flat": bool(kk.eod_flat),
        "rth_only": bool(kk.rth_only),
    }
    try:
        from orb.kk.vnpy.supervisor import kk_vnpy_supervisor

        out["vnpy"] = {
            "running": kk_vnpy_supervisor.is_running,
            "bootstrap": kk_vnpy_supervisor.last_status,
        }
    except Exception as exc:
        out["vnpy"] = {"running": False, "error": str(exc)}
    return out


def fetch_kk_trades(*, limit: int = 200, kk: Optional[KKConfig] = None) -> Dict[str, Any]:
    kk = kk or KKConfig.from_env()
    orb_cfg = kk.orb_session_cfg()
    session_date = _session_date_now(orb_cfg)
    lim = max(1, min(int(limit or 200), 500))
    rows: List[Dict[str, Any]] = []
    open_syms: set[str] = set()

    conn = _db_conn()
    try:
        cur = conn.cursor()
        migrate_kk_tables(cur)

        for sym in kk.symbol_list():
            sym = norm_symbol(sym)
            state = load_state_json(cur, sym, session_date)
            pos = _pos_from_state(state)
            side = _side_name(int(pos.get("side") or 0))
            if not side:
                continue
            open_syms.add(sym)
            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "outcome": None,
                    "entry_price": float(pos.get("entry") or 0.0) or None,
                    "exit_price": None,
                    "notional_usdt": float(pos.get("notional") or 0.0) or None,
                    "pnl_usdt": None,
                    "recorded_at_utc": None,
                    "session_date": session_date,
                    "event": "open",
                    "robot_label": _bot_label(sym),
                }
            )

        cur.execute(
            """
            SELECT session_date, symbol, event, side, entry, exit_px, notional_usdt,
                   pnl_usdt, outcome, created_at_utc
            FROM kk_trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (lim,),
        )
        for r in cur.fetchall():
            sym = norm_symbol(str(r[1]))
            event = str(r[2] or "")
            if event == "open" and sym in open_syms:
                continue
            side = _side_name(r[3])
            outcome = str(r[8] or "").strip().lower() or None
            if event == "close" and not outcome:
                outcome = "close"
            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "outcome": outcome if event == "close" else None,
                    "entry_price": float(r[4] or 0.0) or None,
                    "exit_price": float(r[5] or 0.0) or None if event == "close" else None,
                    "notional_usdt": float(r[6] or 0.0) or None,
                    "pnl_usdt": float(r[7]) if r[7] is not None else None,
                    "recorded_at_utc": str(r[9] or ""),
                    "session_date": str(r[0] or ""),
                    "event": event,
                    "robot_label": _bot_label(sym),
                }
            )
    finally:
        conn.close()

    return {"ok": True, "trades": rows, "session_date": session_date}


def clear_kk_db() -> Dict[str, Any]:
    conn = _db_conn()
    try:
        cur = conn.cursor()
        migrate_kk_tables(cur)
        counts: Dict[str, int] = {}
        for table in (
            "kk_trades",
            "kk_runs",
            "kk_symbol_state",
            "kk_session_opens",
            "kk_symbol_bots",
        ):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int(cur.fetchone()[0] or 0)
            cur.execute(f"DELETE FROM {table}")
        conn.commit()
        return {"ok": True, "deleted": counts}
    finally:
        conn.close()
