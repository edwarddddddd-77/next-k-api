"""Trading ORB vnpy 成交持久化与复利钱包。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from quant.common.fees import trade_fee_usdt
from quant.common.kline_cache import norm_symbol
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.db import insert_trade, load_wallet, migrate_orb_vnpy_tables, save_wallet


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def estimate_close_pnl(
    *,
    side: str,
    entry: float,
    exit_px: float,
    notional_usdt: float,
    cfg: OrbVnpyConfig,
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
        maker_bps=cfg.fee_maker_bps,
        taker_bps=cfg.fee_taker_bps,
    )
    net = round(float(gross) - float(fee), 4)
    return round(float(gross), 4), round(float(fee), 4), net


def record_vnpy_fill(
    *,
    symbol: str,
    event: str,
    side: str,
    price: float,
    volume: float,
    notional_usdt: float,
    session_date: str,
    bar_ms: int,
    cfg: OrbVnpyConfig,
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
        migrate_orb_vnpy_tables(cur)
        now_utc = _utc_now()
        insert_trade(
            cur,
            session_date=session_date,
            symbol=sym,
            event=event,
            side=side,
            entry=float(price) if event == "open" else 0.0,
            exit_px=float(price) if event != "open" else 0.0,
            notional_usdt=float(notional_usdt),
            pnl_usdt_gross=float(pnl_gross or 0.0),
            fee_usdt=float(fee_usdt or 0.0),
            pnl_usdt=float(pnl_usdt or 0.0),
            outcome=outcome,
            detail=detail,
            bar_ms=int(bar_ms),
            now_utc=now_utc,
        )
        wallet = load_wallet(cur, sym, default=float(cfg.equity_usdt or 14.0))
        if cfg.compound and event != "open" and pnl_usdt is not None:
            wallet = round(wallet + float(pnl_usdt), 4)
            save_wallet(cur, sym, wallet, now_utc=now_utc)
        conn.commit()
        return wallet
    finally:
        conn.close()
