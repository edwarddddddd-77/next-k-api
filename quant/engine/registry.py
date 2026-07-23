"""vnpy lane 插件注册表（当前无实盘策略插件；观盘台仅用 AVAX F-MR 纸面信号）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type

from quant.common.strategy_switch import StrategySwitchSpec, vnpy_master_enabled
from quant.engine.bootstrap import ensure_vnpy_path

ensure_vnpy_path()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VnpyLanePlugin:
    name: str
    load_config: Callable[[], Any]
    strategy_class: Type
    class_name: str
    sync_prefix: str
    register: Callable[[Any, Any, Any], List[str]]
    switch: StrategySwitchSpec
    uses_kline_stream: bool = True

    def is_enabled(self) -> bool:
        if not vnpy_master_enabled():
            return False
        if not self.switch.enabled():
            return False
        cfg = self.load_config()
        return bool(getattr(cfg, "enabled", False) and cfg.is_vnpy_engine())

    def config(self) -> Any:
        return self.load_config()

    def switch_status(self, *, live_active: bool | None = None, running: bool | None = None) -> Dict[str, Any]:
        return self.switch.status(live_active=live_active, running=running)


def _load_plugins() -> List[VnpyLanePlugin]:
    return []


_PLUGINS: List[VnpyLanePlugin] | None = None


def vnpy_lane_plugins() -> List[VnpyLanePlugin]:
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = _load_plugins()
    return _PLUGINS


def get_enabled_vnpy_lanes() -> List[Tuple[str, Any]]:
    lanes: List[Tuple[str, Any]] = []
    for plugin in vnpy_lane_plugins():
        if plugin.is_enabled():
            lanes.append((plugin.name, plugin.config()))
    return lanes


def list_strategy_switch_status() -> Dict[str, Any]:
    """所有已注册量化策略的开关状态（当前为空）。"""
    return {
        "ok": True,
        "vnpy_master_enabled": vnpy_master_enabled(),
        "active_lanes": [],
        "strategies": [],
    }


def plugin_for_lane(name: str) -> VnpyLanePlugin | None:
    for plugin in vnpy_lane_plugins():
        if plugin.name == name:
            return plugin
    return None


def register_strategy_classes(cta_engine) -> None:
    for plugin in vnpy_lane_plugins():
        cta_engine.classes[plugin.class_name] = plugin.strategy_class
