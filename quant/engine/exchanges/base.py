"""vnpy Gateway 共用实盘守卫与复利记账（交易所无关）。

Strategy lane packages removed — persist via generic vnpy_wallet only.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Set

from quant.common.kline_cache import norm_symbol
from quant.common.session_paper import _session_date_now
from quant.common.vnpy_wallet import estimate_lane_close_pnl, record_lane_vnpy_fill
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.lane import cfg_for_symbol, lane_live_enabled

ensure_vnpy_path()

from vnpy.trader.constant import Direction, Offset  # noqa: E402
from vnpy.trader.object import OrderRequest, PositionData, TradeData  # noqa: E402

logger = logging.getLogger(__name__)


class VnpyLiveGatewayMixin:
    """混入官方 vnpy Gateway，统一 shadow / 持仓上限 / 钱包同步。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._open_lots: Dict[str, Dict[str, Any]] = {}
        self._active_symbols: Set[str] = set()

    def _lane_cfg(self, symbol: str, *, lane: str | None = None):
        return cfg_for_symbol(symbol, lane=lane)

    def hydrate_from_exchange(self, symbols: list[str]) -> None:
        from quant.engine.exchanges.registry import get_live_adapter

        adapter = get_live_adapter()
        if not adapter.credentials_configured():
            return
        try:
            snaps = adapter.fetch_position_snapshots(symbols)
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
        from quant.engine.exchanges.registry import get_live_adapter

        sym = get_live_adapter().symbol_from_vt(req.symbol)
        cfg = self._lane_cfg(sym)
        if getattr(cfg, "shadow", False):
            self.write_log(f"SHADOW=1 跳过实盘下单 {sym}")
            return ""
        if not lane_live_enabled(cfg):
            self.write_log(f"LIVE_ENABLED=0 或未配置 API Key，拒单 {sym}")
            return ""
        vol = float(req.volume or 0.0)
        if vol <= 0:
            self.write_log(
                f"拒单 volume<=0 {sym} {req.direction.value} {req.offset.value} price={req.price}"
            )
            return ""
        if req.offset == Offset.OPEN:
            from quant.common.portfolio_guard import portfolio_allows_open

            ok, reason = portfolio_allows_open(sym, cfg, active_symbols=self._active_symbols)
            if not ok:
                self.write_log(reason)
                return ""
        return super().send_order(req)

    def on_position(self, position: PositionData) -> None:
        from quant.engine.exchanges.registry import get_live_adapter

        sym = get_live_adapter().symbol_from_vt(position.symbol)
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

    def _session_date(self, cfg) -> str:
        return _session_date_now(cfg.orb_session_cfg())

    def _lane_detail(self, cfg) -> dict | None:
        lane = getattr(cfg, "lane", None)
        if lane:
            return {"lane": lane}
        return None

    def _persist_trade(self, trade: TradeData) -> None:
        from quant.engine.exchanges.registry import get_live_adapter

        sym = get_live_adapter().symbol_from_vt(trade.symbol)
        cfg = self._lane_cfg(sym)
        if not lane_live_enabled(cfg) or getattr(cfg, "shadow", False):
            return
        lane = str(getattr(cfg, "lane", "") or "")
        if not lane:
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
                "lane": lane,
            }
            record_lane_vnpy_fill(
                lane=lane,
                symbol=sym,
                event="open",
                side=side,
                price=px,
                volume=vol,
                notional_usdt=notional,
                session_date=session_date,
                bar_ms=bar_ms,
                cfg=cfg,
                detail=self._lane_detail(cfg),
            )
            return

        lot = self._open_lots.get(sym, {})
        pos_side = str(lot.get("side") or ("LONG" if trade.direction == Direction.SHORT else "SHORT"))
        entry_px = float(lot.get("entry") or px)
        notion = float(lot.get("notional_usdt") or 0.0)
        if notion <= 0:
            notion = px * vol
        gross, fee, net = estimate_lane_close_pnl(
            side=pos_side,
            entry=entry_px,
            exit_px=px,
            notional_usdt=notion,
        )
        record_lane_vnpy_fill(
            lane=lane,
            symbol=sym,
            event="close",
            side=pos_side,
            price=px,
            volume=vol,
            notional_usdt=notion,
            session_date=session_date,
            bar_ms=bar_ms,
            cfg=cfg,
            outcome="close",
            pnl_usdt=net,
            pnl_gross=gross,
            fee_usdt=fee,
            detail=self._lane_detail(cfg),
        )
        self._open_lots.pop(sym, None)
