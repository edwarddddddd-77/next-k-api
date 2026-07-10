"""vnpy 多 lane 配置与辅助函数（lane 列表见 quant.engine.registry）。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from quant.common.kline_cache import norm_symbol
from quant.trading_orb.config import OrbVnpyConfig
from quant.engine.registry import get_enabled_vnpy_lanes as _registry_enabled_lanes
from quant.engine.registry import plugin_for_lane


def get_enabled_vnpy_lanes() -> List[Tuple[str, Any]]:
    return _registry_enabled_lanes()


def combined_vnpy_enabled() -> bool:
    return bool(get_enabled_vnpy_lanes())


def get_active_vnpy_config() -> Tuple[Optional[str], Any]:
    lanes = get_enabled_vnpy_lanes()
    if not lanes:
        return None, None
    if len(lanes) == 1:
        return lanes[0]
    return "combined", {"lanes": lanes}


def find_symbol_pool_overlaps(lanes: Optional[List[Tuple[str, Any]]] = None) -> List[str]:
    """各 vnpy lane 不得共享同一标的（同一交易所仅一个净持仓）。"""
    seen: dict[str, str] = {}
    overlaps: List[str] = []
    for name, cfg in lanes or get_enabled_vnpy_lanes():
        for raw in cfg.symbol_list():
            sym = norm_symbol(raw)
            owner = seen.get(sym)
            if owner is not None and owner != name:
                if sym not in overlaps:
                    overlaps.append(sym)
            else:
                seen[sym] = name
    return overlaps


def cfg_for_lane(lane_name: str) -> Any:
    plugin = plugin_for_lane(lane_name)
    if plugin is not None:
        return plugin.config()
    if lane_name == "mtfmomo":
        from quant.mtfmomo.config import MtfMomoConfig

        return MtfMomoConfig.from_env()
    if lane_name == "kama_trend":
        from quant.kama_trend.config import KamaTrendConfig

        return KamaTrendConfig.from_env()
    if lane_name == "squeeze_breakout":
        from quant.squeeze_breakout.config import SqueezeBreakoutConfig

        return SqueezeBreakoutConfig.from_env()
    return OrbVnpyConfig.from_env()


def cfg_for_symbol(symbol: str, *, lane: Optional[str] = None) -> Any:
    sym = norm_symbol(symbol)
    if lane:
        return cfg_for_lane(lane)
    for lane_name, cfg in get_enabled_vnpy_lanes():
        if sym in {norm_symbol(s) for s in cfg.symbol_list()}:
            return cfg
    return OrbVnpyConfig.from_env()


def combined_symbol_pool() -> set[str]:
    pool: set[str] = set()
    for _, cfg in get_enabled_vnpy_lanes():
        pool.update(norm_symbol(s) for s in cfg.symbol_list())
    return pool


def lane_live_enabled(cfg: Any) -> bool:
    lane = getattr(cfg, "lane", None)
    if lane == "mtfmomo":
        from quant.mtfmomo.live_exec import live_enabled as momo_live

        return momo_live(cfg)
    if lane == "kama_trend":
        from quant.kama_trend.live_exec import live_enabled as kama_live

        return kama_live(cfg)
    if lane == "squeeze_breakout":
        from quant.squeeze_breakout.live_exec import live_enabled as sqz_live

        return sqz_live(cfg)
    from quant.trading_orb.live_exec import live_enabled as orb_live

    return orb_live(cfg)


def any_lane_live_enabled() -> bool:
    return any(lane_live_enabled(cfg) for _, cfg in get_enabled_vnpy_lanes())


def combined_max_open_positions() -> int:
    total = 0
    for _, cfg in get_enabled_vnpy_lanes():
        total += int(getattr(cfg, "max_open_positions", 0) or 0)
    return total


def active_lane_session_cfg():
    for name, cfg in get_enabled_vnpy_lanes():
        if name == "trading_orb":
            return cfg.orb_session_cfg()
    lanes = get_enabled_vnpy_lanes()
    if lanes:
        return lanes[0][1].orb_session_cfg()
    return OrbVnpyConfig.from_env().orb_session_cfg()


def lane_rth_only() -> bool:
    for name, cfg in get_enabled_vnpy_lanes():
        if name == "trading_orb":
            return bool(getattr(cfg, "rth_only", True))
    return True


def lane_vnpy_idle_outside_rth() -> bool:
    for _, cfg in get_enabled_vnpy_lanes():
        if bool(getattr(cfg, "vnpy_idle_outside_rth", True)):
            return True
    return True


def lane_eod_flat_and_enabled(engine) -> bool:
    for _, cfg in get_enabled_vnpy_lanes():
        if not getattr(cfg, "enabled", False) or not getattr(cfg, "eod_flat", False):
            continue
        for strategy in getattr(engine, "strategies", {}).values():
            if getattr(strategy, "pos", 0) != 0:
                return True
    return False
