"""官方 vnpy_binance BinanceLinearGateway（ORB / ICT 实盘）。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Set

from orb.core.kline_cache import norm_symbol
from orb.core.session_paper import _session_date_now
from orb.ict.config import IctVnpyConfig
from orb.trading_orb.config import OrbVnpyConfig
from orb.trading_orb.wallet_sync import estimate_close_pnl as orb_estimate_close_pnl
from orb.trading_orb.wallet_sync import record_vnpy_fill as orb_record_vnpy_fill
from orb.vnpy.bootstrap import ensure_vnpy_path
from orb.vnpy.lane import cfg_for_symbol, get_enabled_vnpy_lanes, lane_live_enabled

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Offset  # noqa: E402
from vnpy.trader.object import OrderRequest, PositionData, TradeData  # noqa: E402
from vnpy_binance.linear_gateway import BinanceLinearGateway  # noqa: E402

logger = logging.getLogger(__name__)

GATEWAY_NAME = BinanceLinearGateway.default_name


def binance_credentials_configured() -> bool:
    return bool(
        (os.getenv("BINANCE_API_KEY") or "").strip()
        and (os.getenv("BINANCE_API_SECRET") or "").strip()
    )


def binance_connect_setting() -> dict:
    server = (os.getenv("BINANCE_SERVER") or "REAL").strip().upper()
    if server not in ("REAL", "TESTNET"):
        server = "REAL"
    kline = (os.getenv("BINANCE_KLINE_STREAM") or "").strip()
    if not kline:
        lanes = get_enabled_vnpy_lanes()
        has_stream = any(name in ("trading_orb", "ict_2022") for name, _ in lanes)
        kline = "True" if has_stream else "False"
    if kline.lower() in ("1", "true", "yes", "on"):
        kline = "True"
    else:
        kline = "False"
    proxy_port = int((os.getenv("BINANCE_PROXY_PORT") or "0").strip() or 0)
    return {
        "API Key": (os.getenv("BINANCE_API_KEY") or "").strip(),
        "API Secret": (os.getenv("BINANCE_API_SECRET") or "").strip(),
        "Server": server,
        "Kline Stream": kline,
        "Proxy Host": (os.getenv("BINANCE_PROXY_HOST") or "").strip(),
        "Proxy Port": proxy_port,
    }


def vnpy_vt_symbol(symbol: str) -> str:
    """官方合约 vt_symbol：ETHUSDT_SWAP_BINANCE.GLOBAL"""
    sym = norm_symbol(symbol)
    return f"{sym}_SWAP_BINANCE.GLOBAL"


def symbol_from_vt(vt_symbol: str) -> str:
    raw = str(vt_symbol or "").split(".", 1)[0]
    if raw.endswith("_SWAP_BINANCE"):
        raw = raw[: -len("_SWAP_BINANCE")]
    return norm_symbol(raw)


class VnpyBinanceLinearGateway(BinanceLinearGateway):
    """官方 Gateway + vnpy 实盘守卫与复利记账。"""

    def __init__(self, event_engine, gateway_name: str = GATEWAY_NAME) -> None:
        super().__init__(event_engine, gateway_name)
        self._open_lots: Dict[str, Dict[str, Any]] = {}
        self._active_symbols: Set[str] = set()

    def _lane_cfg(self, symbol: str, *, lane: str | None = None):
        return cfg_for_symbol(symbol, lane=lane)

    def hydrate_from_exchange(self, symbols: list[str]) -> None:
        """启动时从交易所恢复持仓集合与开仓 lot（重启后平仓 PnL 正确）。"""
        if not binance_credentials_configured():
            return
        try:
            from orb.vnpy.binance_account import fetch_position_snapshots

            snaps = fetch_position_snapshots(symbols)
        except Exception as exc:
            logger.warning("[vnpy] hydrate exchange state failed: %s", exc)
            return
        for sym, snap in snaps.items():
            amt = float(snap.get("amount") or 0.0)
            if abs(amt) < 1e-12:
                continue
            self._active_symbols.add(sym)
            entry = float(snap.get("entry") or 0.0)
            vol = abs(amt)
            if entry > 0:
                self._open_lots[sym] = {
                    "side": "LONG" if amt > 0 else "SHORT",
                    "entry": entry,
                    "notional_usdt": entry * vol,
                    "volume": vol,
                }

    def send_order(self, req: OrderRequest) -> str:
        sym = symbol_from_vt(req.symbol)
        cfg = self._lane_cfg(sym)
        if getattr(cfg, "shadow", False):
            self.write_log(f"SHADOW=1 跳过实盘下单 {sym}")
            return ""
        if not lane_live_enabled(cfg):
            self.write_log(f"LIVE_ENABLED=0 或未配置币安 Key，拒单 {sym}")
            return ""
        vol = float(req.volume or 0.0)
        if vol <= 0:
            self.write_log(
                f"拒单 volume<=0 {sym} {req.direction.value} {req.offset.value} price={req.price}"
            )
            return ""
        if req.offset == Offset.OPEN:
            max_pos = int(getattr(cfg, "max_open_positions", 0) or 0)
            if max_pos > 0 and sym not in self._active_symbols:
                if self._open_position_count(cfg) >= max_pos:
                    self.write_log(f"已达最大持仓数 {max_pos}，拒单 {sym}")
                    return ""
        return super().send_order(req)

    def on_position(self, position: PositionData) -> None:
        sym = symbol_from_vt(position.symbol)
        if abs(float(position.volume or 0.0)) > 0:
            self._active_symbols.add(sym)
        else:
            self._active_symbols.discard(sym)
        super().on_position(position)

    def on_trade(self, trade: TradeData) -> None:
        super().on_trade(trade)
        try:
            self._persist_trade(trade)
        except Exception as exc:
            self.write_log(f"trade persist failed {trade.symbol}: {exc}")

    def _lane_pool(self, cfg) -> Set[str]:
        return {norm_symbol(s) for s in cfg.symbol_list()}

    def _open_position_count(self, cfg) -> int:
        pool = self._lane_pool(cfg)
        return sum(1 for sym in self._active_symbols if sym in pool)

    def _session_date(self, cfg) -> str:
        return _session_date_now(cfg.orb_session_cfg())

    def _is_eod_close(self, cfg) -> bool:
        if isinstance(cfg, IctVnpyConfig):
            return False
        if not getattr(cfg, "eod_flat", False):
            return False
        import pandas as pd

        from orb.vnpy.eod import effective_eod_hm

        sess = cfg.orb_session_cfg()
        now_ms = int(time.time() * 1000)
        ts = pd.Timestamp(now_ms, unit="ms", tz=sess.session_tz)
        eh, em = effective_eod_hm(
            bar_ms=now_ms,
            session_tz=sess.session_tz,
            session_open_time=sess.session_open_time,
            session_close_time=sess.session_close_time,
            market=sess.market,
            exit_hour=int(getattr(cfg, "exit_hour", 15)),
            exit_minute=int(getattr(cfg, "exit_minute", 55)),
        )
        return ts.hour > eh or (ts.hour == eh and ts.minute >= em)

    def _wallet_cfg_for_persist(self, cfg):
        if isinstance(cfg, OrbVnpyConfig):
            return cfg
        if isinstance(cfg, IctVnpyConfig):
            orb = OrbVnpyConfig.from_env()
            return OrbVnpyConfig(
                equity_usdt=float(cfg.equity_usdt),
                compound=bool(cfg.compound),
                fee_maker_bps=orb.fee_maker_bps,
                fee_taker_bps=orb.fee_taker_bps,
            )
        return OrbVnpyConfig.from_env()

    def _persist_trade(self, trade: TradeData) -> None:
        sym = symbol_from_vt(trade.symbol)
        cfg = self._lane_cfg(sym)
        if not lane_live_enabled(cfg) or getattr(cfg, "shadow", False):
            return
        bar_ms = int(time.time() * 1000)
        session_date = self._session_date(cfg)
        px = float(trade.price or 0.0)
        vol = float(trade.volume or 0.0)
        if px <= 0 or vol <= 0:
            return

        if trade.offset == Offset.OPEN:
            side = "LONG" if trade.direction == Direction.LONG else "SHORT"
            notional = px * vol
            self._open_lots[sym] = {
                "side": side,
                "entry": px,
                "notional_usdt": notional,
                "volume": vol,
            }
            wallet_cfg = self._wallet_cfg_for_persist(cfg)
            detail = {"lane": "ict_2022"} if isinstance(cfg, IctVnpyConfig) else None
            orb_record_vnpy_fill(
                symbol=sym,
                event="open",
                side=side,
                price=px,
                volume=vol,
                notional_usdt=notional,
                session_date=session_date,
                bar_ms=bar_ms,
                cfg=wallet_cfg,
                detail=detail,
            )
            return

        lot = self._open_lots.get(sym, {})
        pos_side = str(lot.get("side") or ("LONG" if trade.direction == Direction.SHORT else "SHORT"))
        entry_px = float(lot.get("entry") or px)
        notion = float(lot.get("notional_usdt") or 0.0)
        if notion <= 0:
            notion = px * vol
        outcome = "eod" if self._is_eod_close(cfg) else "close"
        wallet_cfg = self._wallet_cfg_for_persist(cfg)
        gross, fee, net = orb_estimate_close_pnl(
            side=pos_side,
            entry=entry_px,
            exit_px=px,
            notional_usdt=notion,
            cfg=wallet_cfg,
        )
        detail = {"lane": "ict_2022"} if isinstance(cfg, IctVnpyConfig) else None
        orb_record_vnpy_fill(
            symbol=sym,
            event="close",
            side=pos_side,
            price=px,
            volume=vol,
            notional_usdt=notion,
            session_date=session_date,
            bar_ms=bar_ms,
            cfg=wallet_cfg,
            outcome=outcome,
            pnl_usdt=net,
            pnl_gross=gross,
            fee_usdt=fee,
            detail=detail,
        )
        self._open_lots.pop(sym, None)


__all__ = [
    "GATEWAY_NAME",
    "VnpyBinanceLinearGateway",
    "binance_connect_setting",
    "binance_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
