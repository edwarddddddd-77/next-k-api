"""IBS 激进 lane 配置 — entry 0.19 / exit 0.95。"""

from __future__ import annotations

from quant.ibs.config import IbsLaneConfig
from quant.ibs.profile import PROFILE_AGGRESSIVE
from quant.ibs_aggressive.paths import resolve_ibs_aggressive_symbols_path
from quant.ibs_aggressive.switches import IBS_AGGRESSIVE_SWITCH


class IbsAggressiveConfig(IbsLaneConfig):
    lane: str = "ibs_aggressive"
    profile: str = PROFILE_AGGRESSIVE

    @classmethod
    def from_env(cls) -> "IbsAggressiveConfig":
        base = IbsLaneConfig.from_env(
            lane="ibs_aggressive",
            profile=PROFILE_AGGRESSIVE,
            switch=IBS_AGGRESSIVE_SWITCH,
            resolve_symbols_path=resolve_ibs_aggressive_symbols_path,
            env_prefix="IBS_AGGRESSIVE",
        )
        return cls(**base.__dict__)
