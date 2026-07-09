"""vnpy CTA 引擎补丁：RTH 外不处理 tick/stop；volume<=0 拒发。"""

from __future__ import annotations

import logging
import time
from typing import Any

from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.lane import (
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
    from quant.common.session_paper import in_regular_session

    dt = getattr(tick, "datetime", None)
    if dt is not None:
        ms = int(dt.timestamp() * 1000)
    else:
        ms = int(time.time() * 1000)
    return bool(in_regular_session(active_lane_session_cfg(), now_ms=ms))


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
