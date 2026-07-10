"""vnpy lane 复利钱包与成交持久化（crypto + 通用 lane）。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from quant.common.fees import fee_maker_bps_from_env, fee_taker_bps_from_env, trade_fee_usdt
from quant.common.kline_cache import norm_symbol


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate_vnpy_lane_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS vnpy_lane_symbol_bots (
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            wallet_usdt REAL NOT NULL,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (lane, symbol)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS vnpy_lane_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane TEXT NOT NULL,
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event TEXT NOT NULL,
            side TEXT,
            entry REAL,
            exit_px REAL,
            notional_usdt REAL,
            pnl_usdt_gross REAL,
            fee_usdt REAL,
            pnl_usdt REAL,
            outcome TEXT,
            detail_json TEXT,
            bar_ms INTEGER,
            created_at_utc TEXT NOT NULL
        )
        """
    )


def load_lane_wallet(
    cur: sqlite3.Cursor,
    *,
    lane: str,
    symbol: str,
    default: float,
) -> float:
    sym = norm_symbol(symbol)
    cur.execute(
        "SELECT wallet_usdt FROM vnpy_lane_symbol_bots WHERE lane = ? AND symbol = ?",
        (lane, sym),
    )
    row = cur.fetchone()
    if row is None:
        return float(default)
    return float(row[0] or default)


def save_lane_wallet(
    cur: sqlite3.Cursor,
    *,
    lane: str,
    symbol: str,
    wallet: float,
    now_utc: str,
) -> None:
    sym = norm_symbol(symbol)
    cur.execute(
        """
        INSERT INTO vnpy_lane_symbol_bots (lane, symbol, wallet_usdt, updated_at_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(lane, symbol) DO UPDATE SET
            wallet_usdt = excluded.wallet_usdt,
            updated_at_utc = excluded.updated_at_utc
        """,
        (lane, sym, round(float(wallet), 4), now_utc),
    )


def lane_equity_usdt(cfg: Any, symbol: str, *, cur: sqlite3.Cursor | None = None) -> float:
    base = float(getattr(cfg, "equity_usdt", 0.0) or 100.0)
    if not getattr(cfg, "compound", False) or cur is None:
        return base
    lane = str(getattr(cfg, "lane", "") or "")
    if not lane:
        return base
    return load_lane_wallet(cur, lane=lane, symbol=symbol, default=base)


def estimate_lane_close_pnl(
    *,
    side: str,
    entry: float,
    exit_px: float,
    notional_usdt: float,
) -> tuple[float, float, float]:
    entry_v = float(entry or 0.0)
    exit_v = float(exit_px or 0.0)
    notion = float(notional_usdt or 0.0)
    if entry_v <= 0 or exit_v <= 0 or notion <= 0:
        return 0.0, 0.0, 0.0
    side_u = str(side).upper()
    if side_u == "LONG":
        gross = (exit_v - entry_v) / entry_v * notion
    else:
        gross = (entry_v - exit_v) / entry_v * notion
    fee = trade_fee_usdt(
        notion,
        entry_mode="signal",
        maker_bps=fee_maker_bps_from_env(),
        taker_bps=fee_taker_bps_from_env(),
    )
    net = round(float(gross) - float(fee), 4)
    return round(float(gross), 4), round(float(fee), 4), net


def record_lane_vnpy_fill(
    *,
    lane: str,
    symbol: str,
    event: str,
    side: str,
    price: float,
    volume: float,
    notional_usdt: float,
    session_date: str,
    bar_ms: int,
    cfg: Any,
    outcome: str = "",
    pnl_usdt: Optional[float] = None,
    pnl_gross: Optional[float] = None,
    fee_usdt: Optional[float] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> float:
    sym = norm_symbol(symbol)
    from accumulation_radar import init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        migrate_vnpy_lane_tables(cur)
        now_utc = _utc_now()
        cur.execute(
            """
            INSERT INTO vnpy_lane_trades (
                lane, session_date, symbol, event, side, entry, exit_px, notional_usdt,
                pnl_usdt_gross, fee_usdt, pnl_usdt, outcome, detail_json, bar_ms, created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                lane,
                session_date,
                sym,
                event,
                side,
                float(price) if event == "open" else 0.0,
                float(price) if event != "open" else 0.0,
                float(notional_usdt),
                float(pnl_gross or 0.0),
                float(fee_usdt or 0.0),
                float(pnl_usdt or 0.0),
                outcome,
                json.dumps(detail or {}, ensure_ascii=False),
                int(bar_ms),
                now_utc,
            ),
        )
        wallet = load_lane_wallet(
            cur,
            lane=lane,
            symbol=sym,
            default=float(getattr(cfg, "equity_usdt", 0.0) or 100.0),
        )
        if getattr(cfg, "compound", False) and event != "open" and pnl_usdt is not None:
            wallet = round(wallet + float(pnl_usdt), 4)
            save_lane_wallet(cur, lane=lane, symbol=sym, wallet=wallet, now_utc=now_utc)
        conn.commit()
        return wallet
    finally:
        conn.close()
