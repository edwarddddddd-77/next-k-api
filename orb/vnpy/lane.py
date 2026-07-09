"""vnpy 多 lane 配置（ICT + ORB + Aberration）。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from orb.aberration.config import AberrationVnpyConfig
from orb.core.kline_cache import norm_symbol
from orb.ict.config import IctVnpyConfig
from orb.trading_orb.config import OrbVnpyConfig


def get_enabled_vnpy_lanes() -> List[Tuple[str, Any]]:
    """返回所有已启用的 vnpy lane，顺序：ICT → ORB → Aberration。"""
    lanes: List[Tuple[str, Any]] = []
    ict = IctVnpyConfig.from_env()
    if ict.enabled and ict.is_vnpy_engine():
        lanes.append(("ict_2022", ict))
    orb = OrbVnpyConfig.from_env()
    if orb.enabled and orb.is_vnpy_engine():
        lanes.append(("trading_orb", orb))
    ab = AberrationVnpyConfig.from_env()
    if ab.enabled and ab.is_vnpy_engine():
        lanes.append(("aberration", ab))
    return lanes


def combined_vnpy_enabled() -> bool:
    return bool(get_enabled_vnpy_lanes())


def get_active_vnpy_config() -> Tuple[Optional[str], Any]:
    """兼容旧 API：多 lane 时返回 combined。"""
    lanes = get_enabled_vnpy_lanes()
    if not lanes:
        return None, None
    if len(lanes) == 1:
        return lanes[0]
    return "combined", {"lanes": lanes}


def find_symbol_pool_overlaps(lanes: Optional[List[Tuple[str, Any]]] = None) -> List[str]:
    """各 vnpy lane 不得共享同一标的（币安仅一个净持仓）。"""
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
    for name, cfg in get_enabled_vnpy_lanes():
        if name == lane_name:
            return cfg
    return OrbVnpyConfig.from_env()


def cfg_for_symbol(symbol: str, *, lane: Optional[str] = None) -> Any:
    sym = norm_symbol(symbol)
    if lane:
        cfg = cfg_for_lane(lane)
        pool = {norm_symbol(s) for s in cfg.symbol_list()}
        return cfg if sym in pool else OrbVnpyConfig.from_env()
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
    from orb.aberration.config import AberrationVnpyConfig as AbCfg
    from orb.ict.config import IctVnpyConfig as IctCfg
    from orb.trading_orb.live_exec import live_enabled as orb_live

    if isinstance(cfg, (IctCfg, AbCfg)):
        if not cfg.live_enabled:
            return False
        from orb.trading_orb.live_exec import binance_credentials_configured

        return binance_credentials_configured()
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
