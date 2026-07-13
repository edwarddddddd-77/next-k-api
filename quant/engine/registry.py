"""vnpy lane 插件注册表 — 策略注册与执行层解耦。

新 lane 步骤：
1. 在 quant/{lane}/config.py 实现 from_env / symbol_list
2. 在 quant/{lane}/switches.py 定义 StrategySwitchSpec
3. 在 quant/{lane}/register.py 实现 register_vnpy_strategies
4. 在本文件 _load_plugins() 中追加 VnpyLanePlugin
"""

from __future__ import annotations

import importlib
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


def _import_plugin(module: str, attr: str) -> VnpyLanePlugin | None:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except ImportError as exc:
        logger.warning("vnpy plugin skipped (missing): %s.%s (%s)", module, attr, exc)
        return None


def _load_plugins() -> List[VnpyLanePlugin]:
    specs = [
        ("quant.trading_orb.register", "TRADING_ORB_VNPY_PLUGIN"),
        ("quant.ib50.register", "IB50_VNPY_PLUGIN"),
        ("quant.anchor_drift.register", "ANCHOR_DRIFT_VNPY_PLUGIN"),
        ("quant.mtfmomo.register", "MTFMOMO_VNPY_PLUGIN"),
        ("quant.kama_trend.register", "KAMA_TREND_VNPY_PLUGIN"),
        ("quant.squeeze_breakout.register", "SQUEEZE_BREAKOUT_VNPY_PLUGIN"),
        ("quant.breakout_donchian.register", "BREAKOUT_DONCHIAN_VNPY_PLUGIN"),
        ("quant.ibs_conservative.register", "IBS_CONSERVATIVE_VNPY_PLUGIN"),
        ("quant.ibs_aggressive.register", "IBS_AGGRESSIVE_VNPY_PLUGIN"),
        ("quant.ibs_tv.register", "IBS_TV_VNPY_PLUGIN"),
    ]
    plugins: List[VnpyLanePlugin] = []
    for module, attr in specs:
        plugin = _import_plugin(module, attr)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


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
    """所有已注册量化策略的开关状态（供 API / 运维）。"""
    from quant.engine.lane import lane_live_enabled

    running_lanes: set[str] = set()
    try:
        from quant.engine.combined_supervisor import combined_vnpy_supervisor

        status = combined_vnpy_supervisor.last_status
        if combined_vnpy_supervisor.is_running and status.get("ok"):
            running_lanes = set(status.get("lanes") or [name for name, _ in get_enabled_vnpy_lanes()])
    except Exception:
        pass

    strategies: List[Dict[str, Any]] = []
    for plugin in vnpy_lane_plugins():
        cfg = plugin.config()
        live_active = lane_live_enabled(cfg) if plugin.switch.enabled() else False
        strategies.append(
            plugin.switch_status(
                live_active=live_active,
                running=plugin.name in running_lanes,
            )
        )
    return {
        "ok": True,
        "vnpy_master_enabled": vnpy_master_enabled(),
        "active_lanes": [name for name, _ in get_enabled_vnpy_lanes()],
        "strategies": strategies,
    }


def plugin_for_lane(name: str) -> VnpyLanePlugin | None:
    for plugin in vnpy_lane_plugins():
        if plugin.name == name:
            return plugin
    return None


def register_strategy_classes(cta_engine) -> None:
    for plugin in vnpy_lane_plugins():
        cta_engine.classes[plugin.class_name] = plugin.strategy_class
