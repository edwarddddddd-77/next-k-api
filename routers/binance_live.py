"""币安 U 本位实盘视图（仓位 / 成交），供前端直连 next-k-api。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from orb.vnpy.binance_gateway import binance_credentials_configured
from orb.trading_orb.config import OrbVnpyConfig
from orb.trading_orb.live_exec import live_enabled

router = APIRouter(prefix="/api/binance", tags=["binance"])


def _require_credentials() -> None:
    if not binance_credentials_configured():
        raise HTTPException(status_code=503, detail="binance_credentials_missing")


@router.get("/status")
async def binance_status() -> Dict[str, Any]:
    orb = OrbVnpyConfig.from_env()
    open_count = 0
    if binance_credentials_configured():
        from orb.vnpy.binance_account import list_all_open_positions

        open_count = len(list_all_open_positions(symbols=orb.symbol_list() or None))
    vnpy: Dict[str, Any] = {"running": False}
    try:
        from orb.trading_orb.vnpy.supervisor import orb_vnpy_supervisor

        vnpy = {
            "running": orb_vnpy_supervisor.is_running,
            "bootstrap": orb_vnpy_supervisor.last_status,
        }
    except Exception as exc:
        vnpy = {"running": False, "error": str(exc)}
    return {
        "ok": True,
        "lane": orb.lane,
        "live_enabled": orb.live_enabled,
        "live_active": live_enabled(orb),
        "api_key_set": binance_credentials_configured(),
        "open_positions": open_count,
        "symbols": orb.symbol_list(),
        "vnpy": vnpy,
    }


@router.get("/positions")
async def binance_positions(
    status: str = Query("open", description="仅支持 open（实时持仓）"),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    if status != "open":
        raise HTTPException(status_code=410, detail="historical_positions_removed")
    _require_credentials()
    from orb.vnpy.binance_account import list_all_open_positions

    rows = list_all_open_positions(symbols=None)
    return rows[: int(limit)]


@router.get("/account/summary")
async def binance_account_summary() -> Dict[str, Any]:
    _require_credentials()
    from orb.vnpy.binance_account import fetch_account_summary

    return fetch_account_summary()


@router.get("/trades")
async def binance_trades(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(200, ge=1, le=1000),
    symbol: Optional[str] = Query(None, description="可选，筛选单标的"),
) -> Dict[str, Any]:
    _require_credentials()
    from orb.vnpy.binance_account import fetch_user_trades

    orb = OrbVnpyConfig.from_env()
    symbols = list(orb.symbol_list() or [])
    if symbol:
        symbols = [symbol]
    else:
        from orb.vnpy.binance_account import list_all_open_positions

        for p in list_all_open_positions(symbols=None):
            s = str(p.get("symbol") or "")
            if s and s not in symbols:
                symbols.append(s)
    if not symbols:
        return {"ok": True, "trades": [], "days": days}
    per_sym = max(20, int(limit // max(1, len(symbols))) + 5)
    rows = fetch_user_trades(symbols, days=days, limit_per_symbol=per_sym)
    return {"ok": True, "trades": rows[: int(limit)], "days": days}
