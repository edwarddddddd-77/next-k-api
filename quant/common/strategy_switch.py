"""量化策略开关 — 统一 env 命名与解析。

每个 vnpy lane 应提供 StrategySwitchSpec，并优先读取：
  STRATEGY_{LANE_ID}_ENABLED / STRATEGY_{LANE_ID}_LIVE / STRATEGY_{LANE_ID}_SHADOW

仍兼容各 lane 历史 env（如 ORB_VNPY_*）。
全局 vnpy 总开关：VNPY_ENABLED（默认开）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple


def env_truthy(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def resolve_switch(keys: Sequence[str], *, default: bool = False) -> bool:
    """按顺序读取 env，首个已设置的键生效；均未设置则 default。"""
    for key in keys:
        raw = os.getenv(key)
        if raw is not None and str(raw).strip():
            return env_truthy(raw, default=default)
    return default


def vnpy_master_enabled() -> bool:
    """全局 vnpy 总开关；未设 env 时默认开启。"""
    return resolve_switch(("VNPY_ENABLED",), default=True)


@dataclass(frozen=True)
class StrategySwitchSpec:
    lane: str
    title: str
    enabled_keys: Tuple[str, ...]
    live_keys: Tuple[str, ...] = ()
    shadow_keys: Tuple[str, ...] = ()
    default_enabled: bool = False
    default_live: bool = False
    default_shadow: bool = False

    @property
    def env_enabled(self) -> str:
        return self.enabled_keys[0]

    @property
    def env_live(self) -> str:
        return self.live_keys[0] if self.live_keys else ""

    @property
    def env_shadow(self) -> str:
        return self.shadow_keys[0] if self.shadow_keys else ""

    def enabled(self) -> bool:
        return resolve_switch(self.enabled_keys, default=self.default_enabled)

    def live(self) -> bool:
        if not self.live_keys:
            return self.default_live
        return resolve_switch(self.live_keys, default=self.default_live)

    def shadow(self) -> bool:
        if not self.shadow_keys:
            return self.default_shadow
        return resolve_switch(self.shadow_keys, default=self.default_shadow)

    def status(self, *, live_active: bool | None = None, running: bool | None = None) -> Dict[str, Any]:
        enabled = self.enabled()
        if not vnpy_master_enabled():
            enabled = False
        out: Dict[str, Any] = {
            "id": self.lane,
            "title": self.title,
            "enabled": enabled,
            "live": self.live(),
            "shadow": self.shadow(),
            "vnpy_master_enabled": vnpy_master_enabled(),
            "env": {
                "enabled": self.env_enabled,
                "live": self.env_live or None,
                "shadow": self.env_shadow or None,
                "legacy_enabled": self.enabled_keys[1] if len(self.enabled_keys) > 1 else None,
                "legacy_live": self.live_keys[1] if len(self.live_keys) > 1 else None,
            },
        }
        if live_active is not None:
            out["live_active"] = live_active
        if running is not None:
            out["running"] = running
        return out
