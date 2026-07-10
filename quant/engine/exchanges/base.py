"""vnpy Gateway 共用实盘守卫与复利记账（交易所无关）。"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Set

from quant.common.kline_cache import norm_symbol
from quant.common.session_paper import _session_date_now
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.lane import cfg_for_symbol, lane_live_enabled
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.wallet_sync import estimate_close_pnl as orb_estimate_close_pnl
from quant.trading_orb.wallet_sync import record_vnpy_fill as orb_record_vnpy_fill
from quant.common.vnpy_wallet import estimate_lane_close_pnl, record_lane_vnpy_fill

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
            max_pos = int(getattr(cfg, "max_open_positions", 0) or 0)
            if max_pos > 0 and sym not in self._active_symbols:
                if self._open_position_count(cfg) >= max_pos:
                    self.write_log(f"已达最大持仓数 {max_pos}，拒单 {sym}")
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

    def _open_position_count(self, cfg) -> int:
        pool = self._lane_pool(cfg)
        return sum(1 for sym in self._active_symbols if sym in pool)

    def _session_date(self, cfg) -> str:
        return _session_date_now(cfg.orb_session_cfg())

    def _is_eod_close(self, cfg) -> bool:
        if not isinstance(cfg, OrbVnpyConfig):
            return False
        if not getattr(cfg, "eod_flat", False):
            return False
        import pandas as pd

        from quant.engine.eod import effective_eod_hm

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
        return OrbVnpyConfig.from_env()

    def _lane_detail(self, cfg) -> dict | None:
        lane = getattr(cfg, "lane", None)
        if lane:
            return {"lane": lane}
        return None

    def _persist_lane_trade(self, trade: TradeData, *, cfg, sym: str) -> None:
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

    def _persist_trade(self, trade: TradeData) -> None:
        from quant.engine.exchanges.registry import get_live_adapter

        sym = get_live_adapter().symbol_from_vt(trade.symbol)
        cfg = self._lane_cfg(sym)
        lane = getattr(cfg, "lane", None)
        if lane in ("mtfmomo", "kama_trend", "squeeze_breakout"):
            if not lane_live_enabled(cfg) or getattr(cfg, "shadow", False):
                return
            self._persist_lane_trade(trade, cfg=cfg, sym=sym)
            return
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
            detail = self._lane_detail(cfg)
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
        detail = self._lane_detail(cfg)
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
