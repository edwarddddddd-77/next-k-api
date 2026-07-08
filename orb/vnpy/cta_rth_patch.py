"""vnpy CTA 引擎补丁：RTH 外不处理 tick/stop；volume<=0 拒发。"""

from __future__ import annotations

import logging
import time
from typing import Any

from orb.vnpy.bootstrap import ensure_vnpy_path
from orb.vnpy.lane import (
    active_lane_session_cfg,
    lane_eod_flat_and_enabled,
    lane_rth_only,
    lane_vnpy_idle_outside_rth,
)

ensure_vnpy_path()

logger = logging.getLogger(__name__)
_PATCHED = False


def tick_in_lane_rth(tick: Any) -> bool:
    if not lane_rth_only():
        return True
    from orb.core.session_paper import in_regular_session

    dt = getattr(tick, "datetime", None)
    if dt is not None:
        ms = int(dt.timestamp() * 1000)
    else:
        ms = int(time.time() * 1000)
    return bool(in_regular_session(active_lane_session_cfg(), now_ms=ms))


def _ict_strategy(strategy: Any) -> bool:
    cn = type(strategy).__name__
    sn = getattr(strategy, "strategy_name", "") or ""
    return cn == "TradingIct2022VnpyStrategy" or str(sn).startswith("ict2022_")


def _dispatch_tick_to_ict_only(self, tick: Any) -> None:
    for strategy in self.strategies.values():
        if not getattr(strategy, "inited", False) or not getattr(strategy, "trading", False):
            continue
        if strategy.vt_symbol != tick.vt_symbol:
            continue
        if _ict_strategy(strategy):
            strategy.on_tick(tick)


def apply_cta_engine_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vnpy.trader.utility import round_to
    from vnpy_ctastrategy.engine import CtaEngine

    _orig_process_tick = CtaEngine.process_tick_event
    _orig_send_order = CtaEngine.send_order
    _orig_check_stop = CtaEngine.check_stop_order

    def process_tick_event(self, event) -> None:
        tick = event.data
        if lane_rth_only() and lane_vnpy_idle_outside_rth() and not tick_in_lane_rth(tick):
            if lane_eod_flat_and_enabled(self):
                return _orig_process_tick(self, event)
            _dispatch_tick_to_ict_only(self, tick)
            return
        return _orig_process_tick(self, event)

    def send_order(
        self,
        strategy,
        direction,
        offset,
        price,
        volume,
        stop,
        lock,
        net,
    ) -> list:
        contract = self.main_engine.get_contract(strategy.vt_symbol)
        if not contract:
            return _orig_send_order(
                self, strategy, direction, offset, price, volume, stop, lock, net
            )
        vol = round_to(float(volume or 0.0), float(contract.min_volume or 0.001))
        if vol <= 0:
            self.write_log(
                f"拒单 volume<=0（舍入后） {strategy.vt_symbol} {offset.value} "
                f"raw={volume} min_vol={contract.min_volume}",
                strategy,
            )
            return []
        return _orig_send_order(self, strategy, direction, offset, price, vol, stop, lock, net)

    def check_stop_order(self, tick) -> None:
        stale: list[str] = []
        for stop_order in list(self.stop_orders.values()):
            if float(stop_order.volume or 0.0) <= 0:
                stale.append(stop_order.stop_orderid)
        for sid in stale:
            so = self.stop_orders.pop(sid, None)
            if so is None:
                continue
            strategy = self.strategies.get(so.strategy_name)
            if strategy is not None:
                vt_set = self.strategy_orderid_map.get(strategy.strategy_name)
                if vt_set and sid in vt_set:
                    vt_set.discard(sid)
            logger.warning(
                "[vnpy] removed zero-volume local stop %s %s",
                so.vt_symbol,
                sid,
            )
        return _orig_check_stop(self, tick)

    CtaEngine.process_tick_event = process_tick_event
    CtaEngine.send_order = send_order
    CtaEngine.check_stop_order = check_stop_order
    _PATCHED = True
    logger.info("[vnpy] CtaEngine patches applied (RTH tick guard, volume<=0)")


def _allow_tick_outside_rth(engine: Any, cfg=None) -> bool:
    """显式传入 cfg 时按该 lane 配置判断 EOD 强平 tick。"""
    if cfg is not None:
        if not getattr(cfg, "enabled", False) or not getattr(cfg, "eod_flat", False):
            return False
        for strategy in getattr(engine, "strategies", {}).values():
            if getattr(strategy, "pos", 0) != 0:
                return True
        return False
    return lane_eod_flat_and_enabled(engine)
