"""vnpy lane helpers — strategy plugins removed; stubs keep engine imports alive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from quant.common.config import OrbConfig
from quant.common.kline_cache import norm_symbol
from quant.engine.registry import get_enabled_vnpy_lanes as _registry_enabled_lanes
from quant.engine.registry import plugin_for_lane


@dataclass
class DisabledLaneConfig:
    """Placeholder when no vnpy strategy lanes are registered."""

    lane: str = "disabled"
    enabled: bool = False
    live_enabled: bool = False
    shadow: bool = True
    eod_flat: bool = False
    rth_only: bool = True
    vnpy_idle_outside_rth: bool = True
    max_open_positions: int = 0
    equity_usdt: float = 0.0
    symbols: list[str] = field(default_factory=list)

    def symbol_list(self) -> list[str]:
        return list(self.symbols)

    def is_vnpy_engine(self) -> bool:
        return False

    def orb_session_cfg(self) -> OrbConfig:
        return OrbConfig()


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
    lane_list = lanes or get_enabled_vnpy_lanes()
    seen: dict[str, str] = {}
    overlaps: List[str] = []
    for name, cfg in lane_list:
        for raw in cfg.symbol_list():
            sym = norm_symbol(raw)
            owner = seen.get(sym)
            if owner is not None and owner != name and sym not in overlaps:
                overlaps.append(sym)
            else:
                seen[sym] = name
    return overlaps


def cfg_for_lane(lane_name: str) -> Any:
    plugin = plugin_for_lane(lane_name)
    if plugin is not None:
        return plugin.config()
    return DisabledLaneConfig(lane=str(lane_name or "disabled"))


def cfg_for_symbol(symbol: str, *, lane: Optional[str] = None) -> Any:
    if lane:
        return cfg_for_lane(lane)
    sym = norm_symbol(symbol)
    for lane_name, cfg in get_enabled_vnpy_lanes():
        if sym in {norm_symbol(s) for s in cfg.symbol_list()}:
            return cfg
    return DisabledLaneConfig()


def combined_symbol_pool() -> set[str]:
    pool: set[str] = set()
    for _, cfg in get_enabled_vnpy_lanes():
        pool.update(norm_symbol(s) for s in cfg.symbol_list())
    return pool


def lane_live_enabled(cfg: Any) -> bool:
    if cfg is None:
        return False
    return bool(getattr(cfg, "live_enabled", False)) and bool(getattr(cfg, "enabled", False))


def any_lane_live_enabled() -> bool:
    return any(lane_live_enabled(cfg) for _, cfg in get_enabled_vnpy_lanes())


def combined_max_open_positions() -> int:
    total = 0
    for _, cfg in get_enabled_vnpy_lanes():
        total += int(getattr(cfg, "max_open_positions", 0) or 0)
    return total


def active_lane_session_cfg() -> OrbConfig:
    lanes = get_enabled_vnpy_lanes()
    if lanes:
        cfg = lanes[0][1]
        if hasattr(cfg, "orb_session_cfg"):
            return cfg.orb_session_cfg()
    return OrbConfig()


def lane_rth_only() -> bool:
    return True


def lane_vnpy_idle_outside_rth() -> bool:
    return True


def lane_eod_flat_and_enabled(engine) -> bool:
    return False
