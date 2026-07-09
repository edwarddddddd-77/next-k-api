"""Trading ORB vnpy API。"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.live_exec import live_enabled

router = APIRouter(prefix="/api/trading-orb", tags=["trading_orb"])


@router.get("/status")
async def trading_orb_status() -> Dict[str, Any]:
    orb = OrbVnpyConfig.from_env()
    out: Dict[str, Any] = {
        "lane": orb.lane,
        "engine": orb.engine,
        "exchange": orb.live_exchange,
        "market_data_exchange": orb.market_data_exchange,
        "enabled": orb.enabled,
        "live_enabled": orb.live_enabled,
        "live_active": live_enabled(orb),
        "symbols": orb.symbol_list(),
        "equity_usdt": orb.equity_usdt,
        "risk_pct": orb.risk_pct,
        "risk_per_trade_usdt": orb.risk_per_trade_usdt,
        "max_open_positions": orb.max_open_positions,
        "compound": orb.compound,
        "shadow": orb.shadow,
        "or_minutes": orb.or_minutes,
        "vol_thresh": orb.vol_thresh,
        "stop_or_mult": orb.stop_or_mult,
        "target_or_mult": orb.target_or_mult,
        "entry_window": f"{orb.entry_start_hour:02d}:{orb.entry_start_minute:02d}-"
        f"{orb.entry_end_hour:02d}:{orb.entry_end_minute:02d}",
    }
    try:
        from quant.engine.combined_supervisor import combined_vnpy_supervisor

        out["vnpy"] = {
            "running": combined_vnpy_supervisor.is_running,
            "bootstrap": combined_vnpy_supervisor.last_status,
        }
    except Exception as exc:
        out["vnpy"] = {"running": False, "error": str(exc)}
    return out
