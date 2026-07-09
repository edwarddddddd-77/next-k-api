"""vnpy lane 插件注册表 — 策略注册与执行层解耦。

新 lane 步骤：
1. 在 quant/{lane}/config.py 实现 from_env / symbol_list
2. 在 quant/{lane}/register.py 实现 register_vnpy_strategies
3. 在本文件 _load_plugins() 中追加 VnpyLanePlugin
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Type

from quant.engine.bootstrap import ensure_vnpy_path

ensure_vnpy_path()


@dataclass(frozen=True)
class VnpyLanePlugin:
    name: str
    load_config: Callable[[], Any]
    strategy_class: Type
    class_name: str
    sync_prefix: str
    register: Callable[[Any, Any, Any], List[str]]
    uses_kline_stream: bool = True

    def is_enabled(self) -> bool:
        cfg = self.load_config()
        return bool(getattr(cfg, "enabled", False) and cfg.is_vnpy_engine())

    def config(self) -> Any:
        return self.load_config()


def _load_plugins() -> List[VnpyLanePlugin]:
    from quant.trading_orb.register import TRADING_ORB_VNPY_PLUGIN

    return [TRADING_ORB_VNPY_PLUGIN]


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


def plugin_for_lane(name: str) -> VnpyLanePlugin | None:
    for plugin in vnpy_lane_plugins():
        if plugin.name == name:
            return plugin
    return None


def register_strategy_classes(cta_engine) -> None:
    for plugin in vnpy_lane_plugins():
        cta_engine.classes[plugin.class_name] = plugin.strategy_class
